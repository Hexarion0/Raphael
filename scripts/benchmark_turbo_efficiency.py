"""Isolated Turbo workload experiments; never changes RAPHAEL or GPU settings.

Run with data/voice/envs/clone/bin/python. Generated audio and telemetry stay in data/.
The playback timeline is simulated at the exact WAV duration, unless a separate
RAPHAEL runtime test is used. Temperature is guarded between transformer steps.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import queue
import random
import re
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXTS = [
    ("short", "Of course."),
    ("normal", "Your server is online and everything looks normal."),
    ("caring", "You're still awake? You should probably get some rest soon."),
    ("serious", "Something isn't right. I'm checking the system now."),
    ("question", "Would you like me to take care of that for you?"),
    ("technical", "Your GPU is at 63 degrees and memory usage is 4.2 gigabytes."),
    ("chuckle", "[chuckle] I had a feeling you would say that."),
    ("long", "I have checked the system, and nothing needs your attention right now. "
     "The server is online, the important services are responding, and storage has enough "
     "free space for normal use. I will keep monitoring the machine and let you know if "
     "anything changes. You can continue with what you were doing."),
]


def write_json(path: Path, value) -> None:
    """Keep an interrupted benchmark readable."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Telemetry:
    """Sample NVML without repeatedly spawning nvidia-smi."""

    def __init__(self, cutoff: int = 85):
        import pynvml as nv

        nv.nvmlInit()
        self.nv, self.handle = nv, nv.nvmlDeviceGetHandleByIndex(0)
        self.cutoff = cutoff
        self.rows: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def sample(self) -> dict:
        nv, handle = self.nv, self.handle
        process_vram = sum(p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                           if p.pid == os.getpid() and p.usedGpuMemory < 2**60) / 2**20
        rss = next((int(line.split()[1]) / 1024
                    for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:')), 0)
        return {
            "time": time.perf_counter(),
            "temperature_c": nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
            "power_w": nv.nvmlDeviceGetPowerUsage(handle) / 1000,
            "utilization_pct": nv.nvmlDeviceGetUtilizationRates(handle).gpu,
            "fan_pct": nv.nvmlDeviceGetFanSpeed(handle),
            "graphics_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_GRAPHICS),
            "memory_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
            "gpu_used_mib": nv.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
            "process_vram_mib": process_vram, "rss_mib": rss,
            "pstate": nv.nvmlDeviceGetPerformanceState(handle),
        }

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.rows.append(self.sample())
            except self.nv.NVMLError:
                pass
            self.stop.wait(0.1)

    def guard(self) -> None:
        if not self.rows or time.perf_counter() - self.rows[-1]["time"] > 2:
            raise RuntimeError("GPU telemetry unavailable; refusing unmonitored workload")
        if self.rows[-1]["temperature_c"] >= self.cutoff:
            raise RuntimeError(f"Thermal stop at {self.rows[-1]['temperature_c']} C")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def summarize(rows: list[dict], start: float, end: float) -> dict:
    """Integrate board power over a named interval, including any pacing/idle."""
    selected = [r for r in rows if start <= r["time"] <= end]
    if not selected:
        return {}
    result = {"seconds": end - start, "started_at": start, "ended_at": end,
              "samples": len(selected),
              "start_c": selected[0]["temperature_c"], "end_c": selected[-1]["temperature_c"]}
    for key in ("power_w", "utilization_pct", "fan_pct", "graphics_mhz", "gpu_used_mib",
                "temperature_c", "process_vram_mib", "rss_mib"):
        if key not in selected[0]:
            continue
        values = [r[key] for r in selected]
        result[key] = {"mean": statistics.mean(values), "peak": max(values), "min": min(values)}
    result["energy_j"] = sum(
        (b["time"] - a["time"]) * (a["power_w"] + b["power_w"]) / 2
        for a, b in zip(selected, selected[1:])
    )
    result["pstates"] = dict(Counter(r["pstate"] for r in selected))
    return result


def cool_down(monitor: Telemetry, ceiling: int, seconds: int = 180) -> dict:
    """Require five seconds below the ceiling; do not count cooldown as workload."""
    start, stable = time.perf_counter(), None
    while time.perf_counter() - start < seconds:
        row = monitor.sample()
        if (row["temperature_c"] <= ceiling
                and row["fan_pct"] <= getattr(monitor, "start_fan_ceiling", 100)):
            stable = stable or time.perf_counter()
            if time.perf_counter() - stable >= 5:
                return row
        else:
            stable = None
        time.sleep(0.5)
    raise RuntimeError(f"Could not reach controlled start <= {ceiling} C in {seconds}s")


@contextlib.contextmanager
def experiment(model, mode: str, monitor: Telemetry):
    """Temporary hooks on this isolated model only; defaults and watermark preserved."""
    import torch

    handles, restorations = [], []

    def hook(_module, _args, _output):
        monitor.guard()
        if mode.startswith("pace"):
            torch.cuda.synchronize()
            time.sleep(float(mode.removeprefix("pace")) / 1000)

    handles.append(model.t3.tfmr.register_forward_hook(hook))
    for module in (model.s3gen.flow.decoder.estimator, model.s3gen.mel2wav):
        handles.append(module.register_forward_pre_hook(lambda *_: monitor.guard()))
    if mode == "offload":
        # S3Gen.device normally follows the reference tokenizer, although generation
        # needs the flow device. This override is confined to the benchmark instance.
        original_class = model.s3gen.__class__
        model.s3gen.__class__ = type("CachedReferenceS3Gen", (original_class,), {
            "device": property(lambda self: next(self.flow.parameters()).device)
        })
        restorations.append(lambda: setattr(model.s3gen, "__class__", original_class))
        for module in (model.ve, model.s3gen.tokenizer, model.s3gen.speaker_encoder):
            module.cpu()
            restorations.append(lambda module=module: module.cuda())
        torch.cuda.empty_cache()  # once after offload; never per sentence
    if mode in {"amp_t3", "half_t3", "native_half_t3"}:
        original = model.t3.inference_turbo
        if mode in {"half_t3", "native_half_t3"}:
            model.t3.half()

            def restore_t3():
                from safetensors.torch import load_file

                model.t3.float()
                weights = load_file(
                    ROOT / "data/voice/models/chatterbox-turbo/t3_turbo_v1.safetensors"
                )
                model.t3.load_state_dict(weights, strict=False)

            restorations.append(restore_t3)
        if mode == "native_half_t3":
            conditionals = vars(model.conds.t3).copy()
            for name, tensor in conditionals.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    setattr(model.conds.t3, name, tensor.half())
            restorations.append(lambda: vars(model.conds.t3).update(conditionals))

        def mixed(*args, **kwargs):
            if mode == "native_half_t3":
                return original(*args, **kwargs)
            with torch.autocast("cuda", dtype=torch.float16):
                return original(*args, **kwargs)

        model.t3.inference_turbo = mixed
        restorations.append(lambda: setattr(model.t3, "inference_turbo", original))
    if mode == "cpu_hift":
        original = model.s3gen.hift_inference
        model.s3gen.mel2wav.cpu()

        def cpu_hift(speech_feat, cache_source=None):
            output = model.s3gen.mel2wav.inference(speech_feat.cpu(), torch.zeros(1, 1, 0))
            return tuple(v.cuda() for v in output)

        model.s3gen.hift_inference = cpu_hift
        restorations.extend([
            lambda: setattr(model.s3gen, "hift_inference", original),
            lambda: model.s3gen.mel2wav.cuda(),
        ])
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for restore in reversed(restorations):
            restore()


def inventory(model) -> dict:
    """Record actual device and dtype; never infer them from checkpoint filenames."""
    modules = {"t3": model.t3, "voice_encoder": model.ve,
               "s3_tokenizer": model.s3gen.tokenizer,
               "speaker_encoder": model.s3gen.speaker_encoder,
               "flow": model.s3gen.flow, "vocoder": model.s3gen.mel2wav}
    return {name: dict(Counter({
        f"{device}/{dtype}": sum(p.numel() * p.element_size() for p in module.parameters()
                                  if str(p.device) == device and str(p.dtype) == dtype) / 2**20
        for device, dtype in {(str(p.device), str(p.dtype)) for p in module.parameters()}
    })) for name, module in modules.items()}


def generate(model, text: str, seed: int, path: Path, monitor: Telemetry) -> dict:
    """Match production inference, record stage timings and reproducible float PCM."""
    import numpy as np
    import soundfile as sf
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parts, restore = {}, []
    token_hash = []
    for obj, method, label in (
        (model.t3, "inference_turbo", "t3"),
        (model.s3gen, "flow_inference", "flow"),
        (model.s3gen, "hift_inference", "vocoder"),
        (model.watermarker, "apply_watermark", "watermark"),
    ):
        original = getattr(obj, method)

        def measured(*args, _original=original, _label=label, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = _original(*args, **kwargs)
            torch.cuda.synchronize()
            parts[_label] = time.perf_counter() - started
            if _label == "t3":
                token_hash.append(hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest())
            return out

        setattr(obj, method, measured)
        restore.append((obj, method, original))
    monitor.guard()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            audio = model.generate(text)
        torch.cuda.synchronize()
        end = time.perf_counter()
    finally:
        for obj, method, original in restore:
            setattr(obj, method, original)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Non-finite or empty PCM")
    sf.write(path, audio, model.sr, subtype="FLOAT")
    return {
        "audio": path.name, "text": text, "seed": seed,
        "started_at": start, "ended_at": end,
        "seconds": end - start, "audio_seconds": len(audio) / model.sr,
        "rtf": (end - start) / (len(audio) / model.sr), "stages": parts,
        "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "tokens_sha256": token_hash[0], "pcm_peak": float(np.max(np.abs(audio))),
        "allocated_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "reserved_peak_mib": torch.cuda.max_memory_reserved() / 2**20,
        "telemetry": summarize(monitor.rows, start, end),
    }


def repeat_pipeline(model, mode: str, monitor: Telemetry, out: Path, seconds: float,
                    sequence: list[tuple[str, str]] | None = None) -> dict:
    """Two waiting WAVs, synthesis and real-time playback overlap; no actual sound device."""
    pending: queue.Queue = queue.Queue(maxsize=2)
    finished = threading.Event()
    starts, gaps, rows = [], [], []
    epoch = time.perf_counter()

    def player():
        previous_end = None
        while True:
            item = pending.get()
            if item is None:
                return
            now = time.perf_counter()
            starts.append(now - epoch)
            if previous_end is not None:
                gaps.append(now - previous_end)
            finished.wait(item["audio_seconds"])
            previous_end = time.perf_counter()

    thread = threading.Thread(target=player, daemon=True)
    thread.start()
    max_queue = 0
    error = None
    try:
        for i in range(len(sequence) if sequence else 1000):
            if seconds and time.perf_counter() - epoch >= seconds:
                break
            # Fixed sentence loop, identical seeds/text for every configuration.
            key, text = sequence[i] if sequence else TEXTS[1 + i % 5]
            row = generate(model, text, 2000 + i % 5, out / f"loop-{key}.wav", monitor)
            rows.append(row)
            pending.put(row)
            max_queue = max(max_queue, pending.qsize())
    except Exception as err:
        error = f"{type(err).__name__}: {err}"
    finally:
        pending.put(None)
        thread.join()
    end = time.perf_counter()
    return {"mode": mode, "rows": rows, "error": error,
            "first_audio_ready_seconds": starts[0] if starts else None,
            "playback": "duration-matched simulator; no output device latency",
            "playback_gaps_seconds": gaps, "max_queue_depth": max_queue,
            "audio_seconds": sum(r["audio_seconds"] for r in rows),
            "synthesis_seconds": sum(r["seconds"] for r in rows),
            "rtf": (sum(r["seconds"] for r in rows) /
                    sum(r["audio_seconds"] for r in rows)) if rows else None,
            "telemetry": summarize(monitor.rows, epoch, end)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["baseline", "pace12", "offload"],
                        choices=["baseline", "pace8", "pace12", "pace16", "offload",
                                 "amp_t3", "half_t3", "native_half_t3", "cpu_hift"])
    parser.add_argument("--cases", nargs="+", default=["short", "normal", "technical"])
    parser.add_argument("--repeat-seconds", type=float, default=0)
    parser.add_argument("--start-ceiling", type=int, default=60)
    parser.add_argument("--cutoff", type=int, default=85)
    parser.add_argument("--idle-seconds", type=float, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cool-each-case", action="store_true")
    parser.add_argument("--fan-ceiling", type=int, default=100)
    parser.add_argument("--chunk-study", action="store_true")
    args = parser.parse_args()
    if not 60 <= args.cutoff <= 85:
        parser.error("--cutoff must be between 60 and 85 C")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    torch.set_num_threads(4)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    monitor = Telemetry(args.cutoff)
    monitor.start_fan_ceiling = args.fan_ceiling
    report: dict = {"modes": {}, "errors": {}, "settings": vars(args) | {"output": str(out)},
                   "notes": ["Isolated experiments; production source/config unchanged.",
                             "Stage synchronization instrumentation used in all modes.",
                             "Board telemetry includes desktop applications."]}
    profile = json.loads((ROOT / "voice_profiles/raphael/voice.json").read_text())
    reference = ROOT / profile["reference_audio"]
    report["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    try:
        start = time.perf_counter()
        time.sleep(args.idle_seconds)
        report["idle_before_model"] = summarize(monitor.rows, start, time.perf_counter())
        if torch.cuda.mem_get_info()[0] / 2**20 < 3000:
            raise RuntimeError("Need at least 3000 MiB free VRAM before loading Turbo")
        t0 = time.perf_counter()
        model = ChatterboxTurboTTS.from_local(str(ROOT / "data/voice/models/chatterbox-turbo"),
                                            device="cuda")
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.prepare_conditionals(str(reference), exaggeration=0.0)
        torch.cuda.synchronize()
        report["conditioning_seconds"] = time.perf_counter() - t0
        report["parameters_mib"] = inventory(model)
        report["attention"] = model.t3.tfmr.config._attn_implementation
        report["torch"] = str(torch.__version__)
        report["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        report["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        report["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        report["idle_before_start"] = monitor.sample()
        start = time.perf_counter()
        time.sleep(args.idle_seconds)
        report["idle_loaded"] = summarize(monitor.rows, start, time.perf_counter())
        for mode in args.modes:
            folder = out / mode
            folder.mkdir(exist_ok=True)
            try:
                with experiment(model, mode, monitor):
                    row = {"start": cool_down(monitor, args.start_ceiling), "samples": [],
                           "parameters_mib": inventory(model)}
                    report["modes"][mode] = row
                    for i, (key, text) in enumerate(TEXTS):
                        if key not in args.cases:
                            continue
                        if args.cool_each_case:
                            cool_down(monitor, args.start_ceiling)
                        row["samples"].append(generate(
                            model, text, 1729 + i, folder / f"{key}.wav", monitor
                        ))
                        print(mode, key, round(row["samples"][-1]["rtf"], 3), flush=True)
                        write_json(folder / "measurements.json", row["samples"])
                        write_json(out / "report.json", report)
                    if args.repeat_seconds:
                        row["repeat_start"] = cool_down(monitor, args.start_ceiling)
                        write_json(out / "report.json", report)
                        row["repeat"] = repeat_pipeline(model, mode, monitor, folder,
                                                        args.repeat_seconds)
                    if args.chunk_study:
                        paragraph = TEXTS[-1][1]
                        groups = {
                            "paragraph": [paragraph],
                            "sentences": re.split(r"(?<=[.!?])\s+", paragraph),
                            "clauses": re.split(r"(?<=[.!?,])\s+", paragraph),
                        }
                        row["chunks"] = {}
                        for name, pieces in groups.items():
                            start = cool_down(monitor, args.start_ceiling)
                            subdir = folder / name
                            subdir.mkdir(exist_ok=True)
                            row["chunks"][name] = repeat_pipeline(
                                model, mode, monitor, subdir, 0,
                                [(f"{i:02d}", text) for i, text in enumerate(pieces)],
                            )
                            row["chunks"][name]["controlled_start"] = start
                            write_json(out / "report.json", report)
                    write_json(out / "report.json", report)
            except Exception as err:
                report["errors"][mode] = f"{type(err).__name__}: {err}"
                print(mode, report["errors"][mode], flush=True)
                write_json(out / "report.json", report)
        if args.profile and not report["errors"]:
            cool_down(monitor, args.start_ceiling)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as p:
                generate(model, TEXTS[1][1], 1730, out / "profile.wav", monitor)
            (out / "profile.txt").write_text(
                p.key_averages().table(sort_by="self_cuda_time_total", row_limit=35)
            )
            p.export_chrome_trace(str(out / "profile-trace.json"))
        start = time.perf_counter()
        time.sleep(args.idle_seconds)
        report["idle_after_loaded"] = summarize(monitor.rows, start, time.perf_counter())
    finally:
        monitor.close()
        write_json(out / "report.json", report)
        write_json(out / "telemetry.json", monitor.rows)


if __name__ == "__main__":
    main()
