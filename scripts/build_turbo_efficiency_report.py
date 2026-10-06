"""Build local audio comparisons and telemetry plots from isolated Turbo runs."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf


def number(value, decimals=2) -> str:
    """Render unavailable metrics distinctly from zero."""
    return "—" if value is None else f"{value:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    reports = [(p.parent, json.loads(p.read_text())) for p in sorted(root.glob("*/report.json"))]
    reference = {}
    for folder, report in reports:
        for row in report.get("modes", {}).get("baseline", {}).get("samples", []):
            reference[(row["text"], row["seed"])] = (folder / "baseline" / row["audio"], row)
    sections, comparisons, plots = [], [], []
    quick_modes = {
        "FP32 baseline": "controlled-precision/baseline",
        "12 ms pacing": "screen-2/pace12",
        "16 ms pacing": "pace16/pace16",
        "Reference encoders on CPU": "screen-2/offload",
        "Native FP16 T3": "native-half/native_half_t3",
        "CPU vocoder": "screen-2/cpu_hift",
    }
    quick = ["<h2>Direct listening comparison</h2><div style='overflow-x:auto'><table><tr>"
             "<th>Sentence</th>" + "".join(f"<th>{name}</th>" for name in quick_modes)
             + "</tr>"]
    for case in ("normal", "technical"):
        quick.append(f"<tr><th>{case}</th>")
        for folder in quick_modes.values():
            path = root / folder / f"{case}.wav"
            href = html.escape(str(path.relative_to(root)))
            control = (f"<audio controls preload='none' src='{href}'></audio>"
                       if path.exists() else "Not completed")
            quick.append(f"<td style='min-width:200px'>{control}</td>")
        quick.append("</tr>")
    quick.append("</table></div>")
    sections.extend(quick)
    for folder, report in reports:
        sections.append(f"<h2>{html.escape(folder.name)}</h2>")
        if report.get("errors"):
            sections.append(f"<pre>{html.escape(json.dumps(report['errors'], indent=2))}</pre>")
        for mode, info in report.get("modes", {}).items():
            sections.append(f"<h3>{html.escape(mode)}</h3>")
            for name in ("word_checks.json", "similarity-chatterbox.json"):
                path = folder / mode / name
                if path.exists():
                    href = html.escape(str(path.relative_to(root)))
                    sections.append(f"<p><a href='{href}'>{html.escape(name)}</a></p>")
            body = []
            for row in info.get("samples", []):
                path = folder / mode / row["audio"]
                original = reference.get((row["text"], row["seed"]))
                diff = {"run": folder.name, "mode": mode, "audio": row["audio"]}
                if original:
                    a, _ = sf.read(original[0], dtype="float64")
                    b, _ = sf.read(path, dtype="float64")
                    diff["baseline"] = str(original[0].relative_to(root))
                    diff["same_tokens"] = original[1]["tokens_sha256"] == row["tokens_sha256"]
                    diff["same_pcm"] = original[1]["pcm_sha256"] == row["pcm_sha256"]
                    diff["same_length"] = len(a) == len(b)
                    if len(a) == len(b):
                        error = float(np.sum((a - b) ** 2))
                        diff["max_absolute_difference"] = float(np.max(np.abs(a - b)))
                        diff["difference_snr_db"] = (
                            float(10 * np.log10(np.sum(a * a) / error)) if error > 0 else None
                        )
                comparisons.append(diff)
                telemetry = row["telemetry"]
                href = html.escape(str(path.relative_to(root)))
                body.append(
                    f"<tr><td>{html.escape(row['text'])}<br>"
                    f"<audio controls preload='none' src='{href}'></audio></td>"
                    f"<td>{number(row['seconds'])}</td><td>{number(row['rtf'])}</td>"
                    f"<td>{number(telemetry.get('power_w', {}).get('mean'))} / "
                    f"{number(telemetry.get('power_w', {}).get('peak'))}</td>"
                    f"<td>{telemetry.get('start_c')} → "
                    f"{telemetry.get('temperature_c', {}).get('peak')}°C</td>"
                    f"<td>{number(row['allocated_peak_mib'], 0)}</td>"
                    f"<td>{diff.get('same_tokens', '—')} / "
                    f"{number(diff.get('difference_snr_db'), 1)} dB</td></tr>"
                )
            sections.append(
                "<table><tr><th>Sample</th><th>Ready s</th><th>RTF</th>"
                "<th>Mean / peak W</th><th>Start → peak temp</th><th>Allocated MiB</th>"
                "<th>Same tokens / PCM difference SNR</th></tr>" + "".join(body) + "</table>"
            )
            repeated = {"repeated speech": info["repeat"]} if "repeat" in info else {}
            repeated.update(info.get("chunks", {}))
            for name, row in repeated.items():
                telemetry = row["telemetry"]
                audio = row["audio_seconds"]
                energy = telemetry.get("energy_j", 0)
                gap = max(row.get("playback_gaps_seconds", []) or [0])
                error = row.get("error") or "Completed"
                start = info.get("repeat_start", row.get("controlled_start"))
                sections.append(
                    f"<h4>{html.escape(name)} — {html.escape(error)}</h4>"
                    f"<p>Start: {html.escape(str(start))}</p>"
                    f"<p>{number(audio)} s audio; RTF {number(row['rtf'])}; "
                    f"first ready {number(row['first_audio_ready_seconds'])} s; "
                    f"largest simulated gap {number(gap, 3)} s; "
                    f"{number(energy / audio if audio else None)} board J/audio-second.</p>"
                    f"<pre>{html.escape(json.dumps(telemetry, indent=2))}</pre>"
                )
                subdir = folder / mode / name if name != "repeated speech" else folder / mode
                for sample in row.get("rows", []):
                    if name == "repeated speech":
                        break  # loop filenames are overwritten; initial A/B samples are above
                    href = html.escape(str((subdir / sample["audio"]).relative_to(root)))
                    sections.append(f"<p>{html.escape(sample['text'])}<br>"
                                    f"<audio controls preload='none' src='{href}'></audio></p>")
        telemetry_path = folder / "telemetry.json"
        if telemetry_path.exists():
            rows = json.loads(telemetry_path.read_text())
            if rows:
                epoch = rows[0]["time"]
                points = [[round(r["time"] - epoch, 2), r["power_w"], r["temperature_c"],
                           r["fan_pct"], r["utilization_pct"]] for r in rows]
                plots.append({"name": folder.name, "points": points})
                sections.append(f"<p><a href='{html.escape(folder.name)}/telemetry.json'>"
                                "Raw timestamped telemetry</a></p>")
    for path in sorted(root.glob("*/runtime.json")):
        report = json.loads(path.read_text())
        sections.append(f"<h2>Actual RAPHAEL playback: {html.escape(report['mode'])}</h2>")
        for row in report["cases"]:
            gaps = row["pipeline"]["playback_gaps"]
            durations = [sf.info(path.parent / name).duration for name in row["audio"]]
            active = row["total_seconds"] - (row["first_audio_seconds"] or 0)
            suspicious = active > sum(durations) * 1.3 + 0.2
            for device, duration in zip(row.get("device_timing", []), durations):
                suspicious |= device.get("active_seconds", 0) > duration * 1.3 + 0.1
            if suspicious:
                sections.append("<p class='notice'>Playback stayed active substantially longer "
                                "than the saved PCM. Small inter-call gaps do not establish "
                                "continuous real-time speech in this run.</p>")
            if row["fallback"]:
                sections.append("<p class='notice'>FAILED Turbo conversation: the benchmark "
                                "thermal guard triggered and the last sentence used Piper. "
                                "This is not a successful quiet-mode result.</p>")
            sections.append(f"<p>{html.escape(row['name'])}: first playback "
                            f"{number(row['first_audio_seconds'])} s, mean gap "
                            f"{number(statistics.mean(gaps) if gaps else None, 3)} s, "
                            f"fallback: {row['fallback']}</p>")
            for name in row["audio"]:
                href = html.escape(str((path.parent / name).relative_to(root)))
                fallback = row["fallback"] and sf.info(path.parent / name).samplerate == 22050
                label = "Piper fallback" if fallback else "Turbo experiment"
                sections.append(f"<p>{html.escape(name)} — {label}<br>"
                                f"<audio controls preload='none' src='{href}'></audio></p>")
    (root / "pcm-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    page = """<!doctype html><meta charset="utf-8"><title>Turbo efficiency experiments</title>
<style>body{font:16px system-ui;margin:2rem;color:#18202a;background:#fafafa}
table{border-collapse:collapse;width:100%}td,th{padding:.7rem;border:1px solid #ccd;text-align:left}
audio{max-width:360px;width:100%}pre{white-space:pre-wrap;font-size:13px}
canvas{width:100%;max-width:1100px;background:white;border:1px solid #ccd}h2{margin-top:3rem}
.notice{padding:1rem;background:#fff0cc;max-width:1000px}</style>
<h1>Chatterbox Turbo: workload and quality experiments</h1>
<p class="notice">Production is unchanged. Same primary reference and sampling defaults.
No sustained quiet mode passed validation. Planned 120-second tests hit the thermal guard.
Temperature-limited runs are incomplete, not sustained thermal successes. Whole-board power
includes the desktop. Differing fan/start temperatures and aborted durations invalidate simple
temperature or average-energy rankings. Stage timings are instrumented; real playback appears
separately. PCM difference SNR is a numerical comparison, not a speaker score.
Listen before considering any precision or placement change.</p>
<div id="plots"></div>""" + "\n".join(sections)
    page += "<script>const plots=" + json.dumps(plots) + ";" + """
for (const plot of plots) {
 const section=document.createElement('section');
 const title=document.createElement('h2');title.textContent=plot.name+' — full telemetry timeline';
 section.append(title);const label=document.createElement('p');
 label.textContent='Blue: watts · Red: °C · Green: fan % · Gray: GPU utilization %. '
 +'Cooldown and load are included. Hover for values.';section.append(label);
 const c=document.createElement('canvas');c.width=1100;c.height=260;section.append(c);
 document.querySelector('#plots').append(section);const ctx=c.getContext('2d');
 const end=plot.points.at(-1)[0]; const colors=['#2463c8','#ce3434','#258d59','#91969f'];
 ctx.font='12px system-ui';ctx.fillStyle='#333';
 for(let v=0;v<=150;v+=25){let y=240-v/150*220;ctx.fillText(v,2,y);
 ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(1090,y);ctx.stroke();}
 colors.forEach((color,k)=>{ctx.beginPath();ctx.strokeStyle=color;
 plot.points.forEach((p,i)=>{const x=35+p[0]/end*1055,y=240-p[k+1]/150*220;
 i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
 c.onmousemove=e=>{const t=(e.offsetX/c.clientWidth*1100-35)/1055*end;
 const p=plot.points.reduce((a,b)=>Math.abs(a[0]-t)<Math.abs(b[0]-t)?a:b);
 c.title=`${p[0]}s: ${p[1]}W, ${p[2]}°C, fan ${p[3]}%, utilization ${p[4]}%`;};
}
</script>"""
    (root / "index.html").write_text(page)
    print(root / "index.html")


if __name__ == "__main__":
    main()
