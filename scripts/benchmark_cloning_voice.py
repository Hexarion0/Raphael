"""Measure one warm, local voice-cloning model without training or RAPHAEL integration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "data/voice/cache/huggingface"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "data/voice/cache/numba"))
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "data/voice/cache/modelscope"))


def deny_network(event, args):
    """Fail closed on Python TCP/UDP connections; GPU and local file APIs stay available."""
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        raise RuntimeError("Network name resolution is disabled during the local benchmark")
    if event == "socket.connect" and isinstance(args[1], tuple):
        raise RuntimeError("Network access is disabled during the local benchmark")


def file_hash(path: Path) -> str:
    """Hash a reference or suite without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    """Persist completed measurements atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class MemorySampler:
    """Sample RSS and NVML memory; keep allocator peaks as separate exact metrics."""

    def __init__(self) -> None:
        import psutil
        import pynvml

        self.psutil, self.nvml = psutil, pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.process = psutil.Process()
        self.halted = threading.Event()
        self.rss_peak = 0
        self.gpu_peak: int | None = None
        self.whole_gpu_peak = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self) -> None:
        """Observe this process and its children; never attribute desktop VRAM to TTS."""
        try:
            processes = [self.process, *self.process.children(recursive=True)]
            ids = {p.pid for p in processes}
            self.rss_peak = max(self.rss_peak, sum(p.memory_info().rss for p in processes))
            self.whole_gpu_peak = max(
                self.whole_gpu_peak,
                self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used,
            )
            gpu = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            used = sum(p.usedGpuMemory for p in gpu if p.pid in ids and p.usedGpuMemory < 2**60)
            self.gpu_peak = max(self.gpu_peak or 0, used)
        except Exception:
            pass  # Missing samples remain distinct from the PyTorch allocator observations.

    def run(self) -> None:
        """Use short, low-overhead samples; interval is recorded in the report."""
        while not self.halted.wait(0.1):
            self.sample()

    def start(self) -> None:
        """Include imports and model loading in process peaks."""
        self.sample()
        self.thread.start()

    def stop(self) -> dict:
        """Return process peaks and release the sampling thread."""
        self.sample()
        self.halted.set()
        self.thread.join(timeout=2)
        return {
            "peak_rss_mib": self.rss_peak / 2**20,
            "sampled_process_vram_mib": (
                self.gpu_peak / 2**20 if self.gpu_peak is not None else None
            ),
            "sampled_whole_gpu_mib": self.whole_gpu_peak / 2**20,
            "memory_sample_interval_seconds": 0.1,
        }


def load_adapter(args):
    """Load only local files and expose waveform chunks with cached conditioning."""
    import torch

    if args.candidate == "chatterbox500":
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_local(args.model, device="cuda")

        def condition():
            model.prepare_conditionals(
                str(args.reference), exaggeration=args.generation.get("exaggeration", 0.5)
            )

        def generate(text):
            yield model.generate(text, **args.generation)

        return model.sr, condition, generate, model, "complete-waveform API", "float32"
    if args.candidate.startswith("chatterbox"):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_local(
            args.model,
            device="cuda",
            nano=args.candidate == "chatterbox-nano",
        )

        def condition():
            model.prepare_conditionals(str(args.reference), exaggeration=0.0)

        def generate(text):
            yield model.generate(text, **args.generation)

        return model.sr, condition, generate, model, "complete-waveform API", "float32"
    if args.candidate == "cosyvoice3":
        sys.path.insert(0, str(args.vendor.resolve()))
        sys.path.insert(0, str(args.vendor.resolve() / "third_party/Matcha-TTS"))
        from unittest.mock import patch

        import onnxruntime
        from cosyvoice.cli.cosyvoice import AutoModel
        from transformers import Qwen2Config, Qwen2ForCausalLM

        onnxruntime.preload_dlls()

        # The stock loader initializes a blank backbone, then overwrites every
        # parameter from llm.pt with strict=True. Construct that same backbone
        # from its official config to avoid downloading duplicate blank weights.
        blank = args.model.resolve() / "CosyVoice-BlankEN"

        def initialize_blank(path, *unused_args, **unused_kwargs):
            if Path(path).resolve() != blank:
                raise ValueError("Unexpected CosyVoice backbone path")
            return Qwen2ForCausalLM(Qwen2Config.from_pretrained(blank, local_files_only=True))

        with patch.object(Qwen2ForCausalLM, "from_pretrained", side_effect=initialize_blank):
            model = AutoModel(
                model_dir=str(args.model.resolve()), fp16=True, load_trt=False, load_vllm=False
            )
        if "CUDAExecutionProvider" not in model.frontend.speech_tokenizer_session.get_providers():
            raise RuntimeError("CosyVoice speech tokenizer fell back to CPU unexpectedly")
        # fp16=True enables autocast but stock loading keeps FP32 weights resident.
        # Half LLM weights make room on this board. The flow produced non-finite
        # output in FP16 here, so retain FP32 flow/HiFT and disable their autocast.
        model.model.llm.half()
        model.model.fp16 = False
        torch.cuda.empty_cache()

        def condition():
            model.add_zero_shot_spk(
                "You are a helpful assistant.<|endofprompt|>"
                + args.reference_text.read_text().strip(),
                str(args.reference),
                "raphael_benchmark",
            )

        def generate(text):
            for chunk in model.inference_zero_shot(
                text,
                "",
                "",
                zero_shot_spk_id="raphael_benchmark",
                stream=True,
            ):
                yield chunk["tts_speech"]

        return (
            model.sample_rate,
            condition,
            generate,
            model,
            "stream=True audio chunks",
            "LLM float16 resident; flow/HiFT float32; AMP disabled",
        )
    if args.candidate == "qwen06":
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            str(args.model.resolve()),
            device_map="cuda:0",
            dtype=getattr(torch, args.qwen_precision),
            attn_implementation="sdpa",
        )
        rate = model.model.speech_tokenizer.get_output_sample_rate()
        prompts = []

        def condition():
            prompts.extend(
                model.create_voice_clone_prompt(
                    ref_audio=str(args.reference),
                    ref_text=args.reference_text.read_text().strip(),
                    x_vector_only_mode=False,
                )
            )

        def generate(text):
            waves, generated_rate = model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=prompts,
            )
            if generated_rate != rate:
                raise ValueError("Qwen's returned sample rate changed during inference")
            yield waves[0]

        return (
            rate,
            condition,
            generate,
            model,
            "complete-waveform Python wrapper",
            args.qwen_precision + " SDPA",
        )
    raise ValueError("Unknown candidate")


def main() -> None:
    """Record hardware compatibility, loading, conditioning and repeated warm requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=["chatterbox500", "chatterbox-turbo", "chatterbox-nano", "cosyvoice3", "qwen06"],
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-text", type=Path)
    parser.add_argument("--suite", type=Path, default=ROOT / "docs/voice-evaluation.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--generation-config", type=Path)
    parser.add_argument("--label", help="Human-readable experiment label")
    parser.add_argument(
        "--qwen-precision", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    args = parser.parse_args()
    args.generation = (
        json.loads(args.generation_config.read_text()) if args.generation_config else {}
    )
    if args.candidate in {"chatterbox-turbo", "chatterbox-nano"}:
        if set(args.generation) & {"exaggeration", "cfg_weight", "min_p"}:
            parser.error("Turbo ignores exaggeration, cfg_weight and min_p; do not benchmark them")
    elif args.generation and args.candidate != "chatterbox500":
        parser.error("Generation overrides are implemented only for Chatterbox")
    sys.addaudithook(deny_network)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently benchmark CPU as GTX performance")
    torch.set_num_threads(4)
    torch.cuda.init()
    device = torch.cuda.get_device_properties(0)
    torch.zeros(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    hardware = {
        "gpu": device.name,
        "compute_capability": [device.major, device.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compiled_architectures": torch.cuda.get_arch_list(),
        "bf16_native_supported": torch.cuda.is_bf16_supported(including_emulation=False),
        "bf16_including_emulation_supported": torch.cuda.is_bf16_supported(),
        "free_vram_mib_before_load": free / 2**20,
        "total_vram_mib": total / 2**20,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    print(json.dumps(hardware, indent=2), flush=True)
    if args.preflight:
        dtype = getattr(torch, args.qwen_precision)
        test = torch.randn(1, 2, 16, 32, device="cuda", dtype=dtype)
        torch.nn.functional.scaled_dot_product_attention(test, test, test)
        test.flatten(0, 2) @ test.flatten(0, 2).T
        torch.nn.functional.conv1d(
            torch.randn(1, 8, 64, device="cuda", dtype=dtype),
            torch.randn(8, 8, 3, device="cuda", dtype=dtype),
        )
        torch.cuda.synchronize()
        print("Precision kernel preflight passed:", args.qwen_precision, flush=True)
        return
    if not all([args.candidate, args.model, args.reference, args.reference_text, args.output]):
        parser.error("A benchmark requires candidate/model/reference/reference-text/output")
    if not args.model.is_dir() or not args.reference.is_file() or not args.reference_text.is_file():
        parser.error("All model/reference inputs must already exist locally")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if free < 3.5 * 2**30:
        raise RuntimeError(
            "Less than 3.5 GiB free; defer GPU loading until other jobs release memory"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        **hardware,
        "candidate": args.candidate,
        "reference": str(args.reference),
        "reference_sha256": file_hash(args.reference),
        "reference_text_sha256": file_hash(args.reference_text),
        "suite_sha256": file_hash(args.suite),
        "errors": [],
        "playback_measured": False,
        "training": False,
        "display_name": args.label or args.candidate,
        "generation_config": args.generation,
        "reference_path": str(args.reference.resolve()),
    }
    inventory = args.model / "inventory.json"
    if inventory.exists():
        summary["model_inventory"] = json.loads(inventory.read_text())
    sampler = MemorySampler()
    sampler.start()
    failures = []

    def worker_failed(args):
        failures.append(args.exc_value)
        os.kill(os.getpid(), signal.SIGUSR1)

    def raise_worker_failure(signum, frame):
        raise RuntimeError(f"Inference worker failed: {failures[-1]}") from failures[-1]

    def request_timed_out(signum, frame):
        raise TimeoutError("Inference exceeded the request time limit")

    # CosyVoice's stock generator can poll forever after an LLM thread crashes.
    threading.excepthook = worker_failed
    signal.signal(signal.SIGUSR1, raise_worker_failure)
    signal.signal(signal.SIGALRM, request_timed_out)
    rows = []
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rate, condition, generate, model, streaming, precision = load_adapter(args)
        torch.cuda.synchronize()
        summary.update(
            model_load_seconds=time.perf_counter() - started,
            streaming=streaming,
            precision=precision,
            sample_rate=rate,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            condition()
        torch.cuda.synchronize()
        summary["conditioning_seconds"] = time.perf_counter() - started
        summary["load_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        summary["load_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        summary["packages"] = {}
        for package in ["torch", "torchaudio", "transformers", "chatterbox-tts", "qwen-tts"]:
            try:
                summary["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        import numpy as np
        import soundfile as sf

        suite = json.loads(args.suite.read_text())
        requests = [("first_inference", suite[0], 0)]
        if not args.smoke_only:
            requests += [
                ("warm", item, repeat) for item in suite for repeat in range(1, args.repeats + 1)
            ]
        for phase, item, repeat in requests:
            seed = 42 + repeat
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            signal.alarm(args.request_timeout)
            chunks, arrivals, durations = [], [], []
            with torch.inference_mode():
                for chunk in generate(item["text"]):
                    if torch.is_tensor(chunk):
                        chunk = chunk.detach().float().cpu().numpy()
                    chunk = np.asarray(chunk, dtype=np.float32).squeeze()
                    torch.cuda.synchronize()
                    if chunk.ndim != 1 or not chunk.size or not np.isfinite(chunk).all():
                        raise ValueError(
                            f"Invalid waveform: shape={chunk.shape}, "
                            f"nonfinite_samples={np.count_nonzero(~np.isfinite(chunk))}"
                        )
                    chunks.append(chunk)
                    arrivals.append(time.perf_counter() - started)
                    durations.append(len(chunk) / rate)
            finished = time.perf_counter()
            signal.alarm(0)
            if not chunks:
                raise ValueError("No audio generated")
            wave = np.concatenate(chunks)
            elapsed, duration = finished - started, len(wave) / rate
            name = f"{phase}-{item['id']}-{repeat:02d}.wav"
            sf.write(args.output / name, wave, rate, subtype="PCM_24")
            onset = np.flatnonzero(np.abs(wave) > 0.003)
            row = {
                "phase": phase,
                "id": item["id"],
                "text": item["text"],
                "spoken_text": item.get("spoken_text", item["text"]),
                "category": item.get("category", ""),
                "generation_config": args.generation,
                "seed": seed,
                "repeat": repeat,
                "audio": name,
                "generation_seconds": elapsed,
                "audio_ready_seconds": arrivals[0],
                "duration_seconds": duration,
                "rtf": elapsed / duration,
                "chunk_arrival_seconds": arrivals,
                "chunk_duration_seconds": durations,
                "leading_silence_seconds_proxy": (int(onset[0]) / rate if onset.size else None),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "peak_amplitude": float(np.max(np.abs(wave))),
                "naturalness": None,
                "similarity": None,
                "consistency": None,
            }
            rows.append(row)
            write_json(args.output / "measurements.json", rows)
            print(
                f"{phase} {item['id']} take {repeat}: ready={arrivals[0]:.3f}s "
                f"total={elapsed:.3f}s duration={duration:.2f}s RTF={elapsed / duration:.3f}",
                flush=True,
            )
        del model
    except Exception as error:
        summary["errors"].append(f"{type(error).__name__}: {error}")
        write_json(args.output / "measurements.json", rows)
        raise
    finally:
        signal.alarm(0)
        summary.update(sampler.stop())
        warm = [r for r in rows if r["phase"] == "warm"]
        if warm:
            summary.update(
                warm_median_audio_ready_seconds=statistics.median(
                    r["audio_ready_seconds"] for r in warm
                ),
                warm_median_rtf=statistics.median(r["rtf"] for r in warm),
            )
        write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
