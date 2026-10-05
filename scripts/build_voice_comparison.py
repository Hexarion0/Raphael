"""Build an offline A/B listening page from completed native cloning benchmarks."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

LABELS = {
    "chatterbox-turbo": "Chatterbox Turbo",
    "cosyvoice3": "CosyVoice3 0.5B",
    "qwen06": "Qwen3-TTS 0.6B Base",
}


def relative(path: Path, output: Path) -> str:
    """Make media links portable together with their parent experiment directory."""
    return html.escape(os.path.relpath(path.resolve(), output.parent.resolve()), quote=True)


def main() -> None:
    """Keep every repeated take accessible, without selecting flattering model samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    models = []
    for directory in args.directories:
        summary = json.loads((directory / "summary.json").read_text())
        rows = json.loads((directory / "measurements.json").read_text())
        checks = directory / "word_checks.json"
        checked = {r["audio"]: r for r in json.loads(checks.read_text())} if checks.exists() else {}
        for encoder in ["chatterbox", "campplus"]:
            similarity = directory / f"similarity-{encoder}.json"
            if similarity.exists():
                for result in json.loads(similarity.read_text()):
                    checked.setdefault(result["audio"], {})[encoder] = result["reference_cosine"]
        models.append((directory, summary, rows, checked))
    references = json.loads(args.references.read_text())
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAPHAEL — native voice cloning comparison</title>
<style>
body{font:16px system-ui;background:#101722;color:#e7edf5;max-width:1500px;margin:auto;padding:24px}
h1{font-size:28px}h2{font-size:21px}p{line-height:1.6}a{color:#a0ccff}
table{width:100%;border-collapse:collapse}th,td{padding:16px;vertical-align:top;text-align:left;
border:1px solid #334256}th{background:#1c293b}td{min-width:230px}audio{width:100%}
.scroll{overflow:auto}.ref{background:#192537;padding:16px;margin:12px 0;border-radius:8px}
small{display:block;color:#b9c7d8;margin:8px 0}button,select,input{font:inherit;padding:7px;
background:#23354d;color:#fff;border:1px solid #56708e;border-radius:5px;margin:4px}
.notes{width:90%}.warning{color:#ffc88c}summary{cursor:pointer}code{word-break:break-all}
</style><h1>RAPHAEL · native voice cloning</h1>
<p>Compare speaker similarity, naturalness, warmth and consistency first. Play the source,
then the same sentence across models. Every take is available. Audio is unprocessed model output.
This page works offline; nothing is uploaded. No training or voice conversion was used.</p>
<p class="warning">References remain provisional: ASR agreement and speaker consistency do not
prove speaker identity, absence of overlap, or perfect transcription.
Final selection needs listening. Latency measures when waveform data reached the CPU,
excluding playback. ASR checks are fallible.</p>
<button onclick="stopAll()">Stop all audio</button>
<button onclick="downloadRatings()">Download listening notes</button>
<h2>Source references</h2>"""
    ]
    for index, reference in enumerate(references, 1):
        start, end = reference["start"], reference["end"]
        url = reference["source_url"] + f"&t={int(start)}s"
        parts.append(
            f'<div class="ref"><b>Reference {index}'
            + (" · benchmark reference" if index == 1 else "")
            + f'</b><small><a href="{html.escape(url, quote=True)}">'
            f"{html.escape(reference['video_id'])} · {start:.2f}–{end:.2f}s</a> "
            f"· {end - start:.2f}s · {html.escape(reference.get('transcript_status', 'pending'))}"
            f'</small><audio controls preload="none" src="'
            f'{relative(Path(reference["reference_wav"]), output)}"></audio>'
            f"<p>{html.escape(reference['text'])}</p><small>Flags: "
            f"{html.escape(', '.join(reference['reasons']) or 'none detected')}</small>"
            f"<small>Music tag {reference.get('music_score', 0):.3f} · "
            f"source-speaker cosine {reference.get('source_speaker_consistency_cosine', 0):.3f} · "
            f"clipping fraction {reference.get('clipping_fraction', 0):.5f}</small>"
            f"<small>{html.escape(reference.get('selection_reason', ''))}</small></div>"
        )
    parts.append(
        '<h2>Measured performance</h2><div class="scroll"><table><tr><th>Model</th>'
        "<th>Load / reference cache</th><th>Median warm audio ready / RTF</th>"
        "<th>Peak process VRAM / RAM</th><th>Output API / stability</th></tr>"
    )
    for _, summary, rows, _ in models:
        warm = [r for r in rows if r["phase"] == "warm"]
        parts.append(
            f"<tr><td>{html.escape(LABELS.get(summary['candidate'], summary['candidate']))}"
            f"<small>{html.escape(summary.get('precision', ''))}</small></td><td>"
            f"{summary.get('model_load_seconds', 0):.2f}s / "
            f"{summary.get('conditioning_seconds', 0):.2f}s</td><td>"
            f"{summary.get('warm_median_audio_ready_seconds', 0):.2f}s / "
            f"{summary.get('warm_median_rtf', 0):.3f}</td><td>"
            f"{summary.get('sampled_process_vram_mib', 0):.0f} MiB / "
            f"{summary.get('peak_rss_mib', 0):.0f} MiB</td><td>"
            f"{html.escape(summary.get('streaming', 'unknown'))}<br>{len(warm)} warm requests; "
            f"{html.escape(str(summary['errors']))}</td></tr>"
        )
    parts.append("</table></div>")
    parts.append(
        '<p class="warning">These WAVs are already generated, so playback here has no synthesis '
        "gaps. CosyVoice emits native chunks, but their arrivals can be slower than playback. "
        "Qwen's tested Python wrapper returns the complete waveform. Speaker cosine checks "
        "are uncalibrated; each encoder is also used by one of the compared systems.</p>"
    )
    diagnostics = output.parent / "diagnostics.json"
    if diagnostics.exists():
        parts.append("<h2>Observed problems and configuration changes</h2>")
        for model, issues in json.loads(diagnostics.read_text()).items():
            parts.append(f"<b>{html.escape(LABELS.get(model, model))}</b><ul>")
            for issue in issues:
                parts.append(
                    f"<li>{html.escape(issue['configuration'])}: {html.escape(issue['issue'])}. "
                    f"{html.escape(issue['outcome'])}.</li>"
                )
            parts.append("</ul>")
    parts.append(
        "<h2>Same sentences, same primary reference</h2>"
        '<div class="scroll"><table><tr><th>Sentence</th>'
    )
    for _, summary, _, _ in models:
        parts.append(
            f"<th>{html.escape(LABELS.get(summary['candidate'], summary['candidate']))}</th>"
        )
    parts.append("</tr>")
    suite = {r["id"]: r["text"] for _, _, rows, _ in models for r in rows}
    for sentence_id, text in suite.items():
        parts.append(f"<tr><td><b>{html.escape(sentence_id)}</b><p>{html.escape(text)}</p></td>")
        for directory, summary, rows, checks in models:
            takes = [r for r in rows if r["id"] == sentence_id and r["phase"] == "warm"]
            takes += [r for r in rows if r["id"] == sentence_id and r["phase"] == "first_inference"]
            key = summary["candidate"] + "-" + sentence_id
            parts.append("<td>")
            if takes:
                parts.append('<select aria-label="Take" onchange="changeTake(this)">')
                for take in takes:
                    label = (
                        "First request"
                        if take["phase"] == "first_inference"
                        else f"Take {take['repeat']}"
                    )
                    info = f"ready {take['audio_ready_seconds']:.2f}s · "
                    info += f"total {take['generation_seconds']:.2f}s · RTF {take['rtf']:.3f}"
                    info += f" · audio {take.get('duration_seconds', 0):.2f}s"
                    info += f" · {len(take.get('chunk_arrival_seconds', []))} chunks"
                    check = checks.get(take["audio"])
                    if check and "word_edits" in check:
                        info += f" · ASR word edits {check['word_edits']}"
                    if take.get("duration_seconds", 1) < 0.5:
                        info += " · REVIEW: unusually short output"
                    parts.append(
                        f'<option value="{relative(directory / take["audio"], output)}" '
                        f'data-info="{html.escape(info, quote=True)}">'
                        f"{html.escape(label)}</option>"
                    )
                first = takes[0]
                parts.append(
                    f'</select><audio controls preload="none" src="'
                    f'{relative(directory / first["audio"], output)}"></audio>'
                    '<small class="metric"></small><button onclick="playFromStart(this)">'
                    "Play from start</button>"
                )
                if checks:
                    parts.append("<details><summary>Word / speaker proxy checks</summary>")
                    for take in takes:
                        check = checks.get(take["audio"])
                        if check and "observed" in check:
                            parts.append(
                                f"<small>Take {take['repeat']}: "
                                f"{html.escape(check['observed'])}</small>"
                            )
                        if check:
                            for encoder in ["chatterbox", "campplus"]:
                                if encoder in check:
                                    parts.append(
                                        f"<small>Take {take['repeat']} · {encoder} "
                                        f"speaker cosine {check[encoder]:.3f}"
                                        " · uncalibrated proxy</small>"
                                    )
                    parts.append("</details>")
                parts.append(
                    f'<input class="notes" data-key="{html.escape(key, quote=True)}" '
                    'placeholder="Similarity, warmth, artifacts, preference…" '
                    'aria-label="Listening notes">'
                )
            else:
                parts.append("No completed warm sample. See recorded errors above.")
            parts.append("</td>")
        parts.append("</tr>")
    parts.append("""</table></div><script>
function stopAll(){document.querySelectorAll('audio').forEach(a=>a.pause());}
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>{
 document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause();});}));
function changeTake(s){let td=s.closest('td');td.querySelector('audio').src=s.value;
 td.querySelector('.metric').textContent=s.selectedOptions[0].dataset.info;}
document.querySelectorAll('select').forEach(changeTake);
function playFromStart(b){let a=b.closest('td').querySelector('audio');stopAll();
 a.currentTime=0;a.play();}
let notes={};try{notes=JSON.parse(localStorage.getItem('raphael-voice-notes')||'{}');}catch(e){}
document.querySelectorAll('.notes').forEach(i=>{i.value=notes[i.dataset.key]||'';
 i.addEventListener('input',()=>{notes[i.dataset.key]=i.value;
 try{localStorage.setItem('raphael-voice-notes',JSON.stringify(notes));}catch(e){}});});
function downloadRatings(){let a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));
 a.download='raphael-listening-notes.json';a.click();URL.revokeObjectURL(a.href);}
</script></html>""")
    output.write_text("\n".join(parts), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
