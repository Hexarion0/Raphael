"""Bounded Qwen experiments; no training, downloads, or production TTS changes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import queue
import signal
import statistics
import sys
import threading
import time
import types
from pathlib import Path

from benchmark_cloning_voice import MemorySampler, deny_network, file_hash, write_json
from voice_pipeline_probe import PacedSink


def cached_prompt(model, args, summary: dict):
    """Persist tensors with weights-only loading and strict source/model/code fingerprints."""
    import torch
    from qwen_tts import VoiceClonePromptItem

    metadata = {
        "format": 1,
        "reference_sha256": summary["reference_sha256"],
        "text_sha256": summary["reference_text_sha256"],
        "model_inventory_sha256": file_hash(args.model / "inventory.json"),
        "inference_source_sha256": file_hash(Path(inspect.getfile(type(model)))),
        "model_source_sha256": file_hash(Path(inspect.getfile(type(model.model)))),
        "conditioning_dtype": "bfloat16",
        "torch": str(torch.__version__),
    }
    if args.prompt_cache and args.prompt_cache.exists():
        data = torch.load(args.prompt_cache, map_location="cpu", weights_only=True)
        if data["metadata"] != metadata:
            raise ValueError("Prompt cache fingerprint mismatch; use a new cache path")
        summary["prompt_cache_hit"] = True
        return [
            VoiceClonePromptItem(
                ref_code=data["ref_code"].to(model.device),
                ref_spk_embedding=data["ref_spk_embedding"].to(model.device),
                ref_text=data["ref_text"],
                x_vector_only_mode=False,
                icl_mode=True,
            )
        ]
    prompts = model.create_voice_clone_prompt(
        ref_audio=str(args.reference), ref_text=args.reference_text.read_text().strip()
    )
    summary["prompt_cache_hit"] = False
    if args.prompt_cache:
        args.prompt_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.prompt_cache.with_suffix(".tmp")
        torch.save(
            {
                "metadata": metadata,
                "ref_code": prompts[0].ref_code.cpu(),
                "ref_spk_embedding": prompts[0].ref_spk_embedding.cpu(),
                "ref_text": prompts[0].ref_text,
            },
            temporary,
        )
        temporary.replace(args.prompt_cache)
    return prompts


def install_compact_predictor(predictor) -> None:
    """Experiment: avoid 15-token HF setup and unused hidden-state histories per frame.

    This preserves temperature/top-k/top-p sampling and the native KV-cached forward.
    Equivalence must be checked against stock code sequences, not assumed.
    """
    import torch
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    def compact(
        self, inputs_embeds, max_new_tokens, do_sample, top_p, top_k, temperature, **unused
    ):
        if inputs_embeds.shape[0] != 1 or max_new_tokens != 15:
            raise ValueError("Compact predictor is only validated for 16-code, batch-one Qwen")
        processors = []
        if do_sample:
            if temperature != 1:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k:
                processors.append(TopKLogitsWarper(top_k))
            if top_p < 1:
                processors.append(TopPLogitsWarper(top_p))
        tokens, past = [], None
        for step in range(max_new_tokens):
            output = self(
                inputs_embeds=inputs_embeds if step == 0 else None,
                input_ids=None if step == 0 else tokens[-1],
                past_key_values=past,
                generation_steps=step,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            past = output.past_key_values
            scores = output.logits[:, -1, :].float().clone()
            for processor in processors:
                scores = processor(None, scores)
            token = (
                torch.multinomial(scores.softmax(dim=-1), num_samples=1)
                if do_sample
                else scores.argmax(dim=-1, keepdim=True)
            )
            tokens.append(token)
        return types.SimpleNamespace(sequences=torch.cat(tokens, dim=-1))

    predictor.generate = types.MethodType(compact, predictor)


def install_stable_fp16(talker) -> list[str]:
    """Experiment: FP16 talker with FP32 gated products/down projections in predictor."""
    talker.half()
    changed = []
    for name, module in talker.named_modules():
        if not name.startswith("code_predictor.") or not hasattr(module, "gate_proj"):
            continue
        module.down_proj.float()

        def forward(self, value):
            product = self.act_fn(self.gate_proj(value)).float() * self.up_proj(value).float()
            return self.down_proj(product).to(value.dtype)

        module.forward = types.MethodType(forward, module)
        changed.append(name)
    return changed


def install_graph_predictor(predictor, preparation: dict) -> None:
    """Experiment: capture only the fixed 15-code predictor, not the growing talker.

    One graph is allowed per process. Keep native parameters/sampling; restore RNG
    after warmup and verify generated codes against stock before claiming equivalence.
    """
    import torch

    install_compact_predictor(predictor)
    compact = predictor.generate
    state = {}

    def graphed(self, inputs_embeds, **kwargs):
        key = (inputs_embeds.shape, inputs_embeds.dtype, tuple((k, v) for k, v in kwargs.items()))
        if state and state["key"] != key:
            raise ValueError("This bounded experiment allows only one predictor graph")
        if not state:
            if torch.cuda.mem_get_info()[0] < 1.5 * 2**30:
                raise RuntimeError("Insufficient headroom for a predictor graph experiment")
            started = time.perf_counter()
            rng = torch.cuda.get_rng_state()
            static = inputs_embeds.detach().clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compact(inputs_embeds=static, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compact(inputs_embeds=static, **kwargs)
            torch.cuda.set_rng_state(rng)
            state.update(key=key, static=static, graph=graph, output=output)
            preparation["graph_preparation_seconds"] = time.perf_counter() - started
        state["static"].copy_(inputs_embeds)
        state["graph"].replay()
        return types.SimpleNamespace(sequences=state["output"].sequences.clone())

    predictor.generate = types.MethodType(graphed, predictor)


class CodecStream:
    """Experimental early decoding through a native talker forward hook.

    Generation and decoding share one worker, with CPU audio handed to the caller.
    Keep the stock 300-code / 25-left-context decoder boundaries. Recompute each
    unfinished segment prefix; this trades throughput for earlier audio.
    """

    def __init__(self, model, prompt, frames: int):
        self.model, self.prompt, self.frames = model, prompt, frames
        self.codes = []
        self.reference = prompt[0].ref_code
        self.emitted = len(self.reference)
        self.messages = queue.Queue(maxsize=8)
        self.cancelled = threading.Event()
        self.decode_seconds = 0.0

    def emit(self, final: bool = False) -> None:
        import torch

        available = len(self.reference) + len(self.codes)
        if available == self.emitted or (not final and available - self.emitted < self.frames):
            return
        all_codes = torch.cat([self.reference, torch.stack(self.codes)], dim=0)
        decoder = self.model.model.speech_tokenizer.model.decoder
        while self.emitted < available:
            segment_start = self.emitted // 300 * 300
            left = max(0, segment_start - 25)
            end = min(available, segment_start + 300)
            before = time.perf_counter()
            decoded = decoder(all_codes[left:end].T[None])[0, 0]
            audio = decoded[(self.emitted - left) * 1920 : (end - left) * 1920]
            audio = audio.float().cpu().numpy()
            self.decode_seconds += time.perf_counter() - before
            self.messages.put(audio, timeout=10)
            self.emitted = end

    def hook(self, module, inputs, output) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("Streaming consumer cancelled")
        code = output.hidden_states[1]
        if code is not None:
            if int(code[0, 0]) == self.model.model.config.talker_config.codec_eos_token_id:
                return
            self.codes.append(code[0].detach())
            self.emit()

    def run(self, text: str, kwargs: dict):
        import torch

        error = []

        def worker():
            handle = self.model.model.talker.register_forward_hook(self.hook)
            try:
                with torch.inference_mode():
                    # Use the stock wrapper, but skip its redundant final full decode.
                    original = self.model.model.speech_tokenizer.decode

                    def skip_decode(encoded):
                        import numpy as np

                        return [np.zeros(1, dtype=np.float32)], 24000

                    self.model.model.speech_tokenizer.decode = skip_decode
                    try:
                        self.model.generate_voice_clone(
                            text=text, language="English", voice_clone_prompt=self.prompt, **kwargs
                        )
                    finally:
                        self.model.model.speech_tokenizer.decode = original
                    self.emit(final=True)
            except Exception as exc:
                error.append(exc)
            finally:
                handle.remove()
                if not self.cancelled.is_set():
                    self.messages.put(None, timeout=10)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                audio = self.messages.get(timeout=120)
                if audio is None:
                    break
                yield audio
            if error:
                raise error[0]
        finally:
            self.cancelled.set()
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("Streaming worker did not stop; exit this benchmark process")


def main() -> None:
    """Use pinned cached inputs and record stages, real chunk arrivals and memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("docs/voice-evaluation.json"))
    parser.add_argument(
        "--reference", type=Path, default=Path("data/voice/references/raphael/reference-1.wav")
    )
    parser.add_argument(
        "--reference-text", type=Path, default=Path("data/voice/references/raphael/reference-1.txt")
    )
    parser.add_argument("--model", type=Path, default=Path("data/voice/models/qwen06"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument(
        "--precision", choices=["bfloat16", "mixed", "float32-talker"], default="bfloat16"
    )
    parser.add_argument("--predictor", choices=["stock", "compact", "graph"], default="stock")
    parser.add_argument("--allocator", choices=["default", "expandable"], default="expandable")
    parser.add_argument("--offload-encoders", action="store_true")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--stream-frames", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--paced-sink", action="store_true")
    parser.add_argument("--prompt-cache", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.stream_frames < 0:
        parser.error("Invalid repetition/chunk count")
    if args.output.exists():
        parser.error("Output already exists; never overwrite previous benchmark evidence")
    if not all(p.exists() for p in [args.model, args.suite, args.reference, args.reference_text]):
        parser.error("All inputs must exist locally; this benchmark never downloads models")
    if args.allocator == "expandable":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    sys.addaudithook(deny_network)
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(4)
    torch.cuda.init()
    torch.zeros(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    if free < 3.5 * 2**30:
        raise RuntimeError("Less than 3.5 GiB free; defer this experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    suite = json.loads(args.suite.read_text())
    if args.precision == "float32-talker" and any(len(i["text"].split()) > 20 for i in suite):
        raise ValueError(
            "FP32 talker is restricted to short screening; long-response headroom unsafe"
        )
    summary = {
        "candidate": "qwen06",
        "display_name": args.label,
        "errors": [],
        "training": False,
        "playback_measured": False,
        "sample_rate": 24000,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "free_vram_mib_before_load": free / 2**20,
        "total_vram_mib": total / 2**20,
        "reference_sha256": file_hash(args.reference),
        "reference_text_sha256": file_hash(args.reference_text),
        "reference_path": str(args.reference.resolve()),
        "suite_sha256": file_hash(args.suite),
        "suite_path": str(args.suite.resolve()),
        "planned_sentence_ids": [item["id"] for item in suite],
        "planned_warm_repeats": 0 if args.smoke_only else args.repeats,
        "experiment_settings": vars(args)
        | {
            "output": str(args.output),
            "suite": str(args.suite),
            "reference": str(args.reference),
            "reference_text": str(args.reference_text),
            "model": str(args.model),
            "prompt_cache": str(args.prompt_cache) if args.prompt_cache else None,
        },
        "precision": args.precision + " / " + args.attention,
        "streaming": "experimental native-code callback"
        if args.stream_frames
        else "complete waveform; stock API",
        "model_inventory": json.loads((args.model / "inventory.json").read_text()),
    }
    sampler = MemorySampler()
    sampler.start()
    rows, timings, captured = [], [], []
    paced = None

    def timeout(signum, frame):
        raise TimeoutError("Bounded request exceeded 120 seconds")

    signal.signal(signal.SIGALRM, timeout)
    try:
        started = time.perf_counter()
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            str(args.model.resolve()),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=args.attention,
            local_files_only=True,
        )
        torch.cuda.synchronize()
        summary["model_load_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            prompts = cached_prompt(model, args, summary)
        torch.cuda.synchronize()
        summary["conditioning_seconds"] = time.perf_counter() - started
        if args.precision == "mixed":
            summary["fp32_predictor_mlp_modules"] = install_stable_fp16(model.model.talker)
        elif args.precision == "float32-talker":
            model.model.talker.float()
        if args.predictor == "compact":
            install_compact_predictor(model.model.talker.code_predictor)
        elif args.predictor == "graph":
            install_graph_predictor(model.model.talker.code_predictor, summary)
        if args.offload_encoders:
            model.model.speaker_encoder.cpu()
            model.model.speech_tokenizer.model.encoder.cpu()
            torch.cuda.empty_cache()
        if args.precision == "float32-talker":
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0] < 1250 * 2**20:
                raise RuntimeError("FP32 talker has insufficient headroom even for short decoding")
        summary["resident_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        # Cache only the invariant reference token IDs; dynamic request text stays uncached.
        tokenize = model._tokenize_texts
        ref_text = model._build_ref_text(prompts[0].ref_text)
        ref_tokens = tokenize([ref_text])

        def tokenized(texts):
            started = time.perf_counter()
            result = ref_tokens if texts == [ref_text] else tokenize(texts)
            timings.append(
                {
                    "stage": "reference_tokens" if texts == [ref_text] else "text",
                    "seconds": time.perf_counter() - started,
                }
            )
            return result

        model._tokenize_texts = tokenized
        for name, obj, attribute in [
            ("speech_codes", model.model, "generate"),
            ("waveform_decode", model.model.speech_tokenizer, "decode"),
        ]:
            original = getattr(obj, attribute)

            def measured(*inputs, _name=name, _function=original, **kwargs):
                started = time.perf_counter()
                result = _function(*inputs, **kwargs)
                torch.cuda.synchronize()
                timings.append({"stage": _name, "seconds": time.perf_counter() - started})
                if _name == "speech_codes":
                    captured.append(result[0][0].detach().cpu().numpy())
                return result

            setattr(obj, attribute, measured)
        requests = [("first_inference", suite[0], 0)]
        if not args.smoke_only:
            requests += [
                ("warm", item, repeat) for item in suite for repeat in range(1, args.repeats + 1)
            ]
        for phase, item, repeat in requests:
            if args.paced_sink and phase == "warm" and paced is None:
                paced = PacedSink()
            torch.manual_seed(42 + repeat)
            torch.cuda.manual_seed_all(42 + repeat)
            np.random.seed(42 + repeat)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings.clear()
            captured.clear()
            memory = MemorySampler()
            memory.start()
            started = time.perf_counter()
            signal.alarm(120)
            kwargs = {
                "non_streaming_mode": args.non_streaming_mode,
                "temperature": args.temperature,
                "max_new_tokens": 2048,
            }
            streamer = None
            with torch.inference_mode():
                if args.stream_frames:
                    streamer = CodecStream(model, prompts, args.stream_frames)
                    iterator = streamer.run(item["text"], kwargs)
                else:

                    def complete():
                        waves, rate = model.generate_voice_clone(
                            text=item["text"],
                            language="English",
                            voice_clone_prompt=prompts,
                            **kwargs,
                        )
                        if rate != 24000:
                            raise ValueError("Unexpected sample rate")
                        yield waves[0]

                    iterator = complete()
                chunks, arrivals = [], []
                for audio in iterator:
                    audio = np.asarray(audio, dtype=np.float32).squeeze()
                    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                        raise ValueError("Invalid audio")
                    chunks.append(audio)
                    arrivals.append(time.perf_counter() - started)
            elapsed = time.perf_counter() - started
            signal.alarm(0)
            observed_memory = memory.stop()
            if not chunks or not captured:
                raise ValueError("Missing audio or codec sequence")
            audio = np.concatenate(chunks)
            duration = len(audio) / 24000
            onset = np.flatnonzero(np.abs(audio) > 0.003)
            usable = next(
                (
                    arrival
                    for chunk, arrival in zip(chunks, arrivals)
                    if np.any(np.abs(chunk) > 0.003)
                ),
                None,
            )
            if usable is None:
                raise ValueError("Audio contains no signal above the recorded 0.003 onset proxy")
            name = f"{phase}-{item['id']}-{repeat:02d}.wav"
            sf.write(args.output / name, audio, 24000, subtype="PCM_24")
            if paced is not None:
                paced.offer(name, audio, 24000)
            np.save(args.output / name.replace(".wav", "-codes.npy"), captured[0])
            row = {
                "phase": phase,
                "id": item["id"],
                "repeat": repeat,
                "seed": 42 + repeat,
                "text": item["text"],
                "spoken_text": item.get("spoken_text", item["text"]),
                "audio": name,
                "duration_seconds": duration,
                "generation_seconds": elapsed,
                "audio_ready_seconds": usable,
                "first_packet_seconds": arrivals[0],
                "leading_silence_seconds_proxy": int(onset[0]) / 24000 if onset.size else None,
                "rtf": elapsed / duration,
                "chunk_arrival_seconds": arrivals,
                "chunk_duration_seconds": [len(c) / 24000 for c in chunks],
                "stage_timings": list(timings),
                "codec_sha256": hashlib.sha256(captured[0].tobytes()).hexdigest(),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                **observed_memory,
                "naturalness": None,
                "similarity": None,
                "consistency": None,
            }
            if streamer:
                row["early_decoder_seconds"] = streamer.decode_seconds
                row["stream_codes_equal_final"] = np.array_equal(
                    torch.stack(streamer.codes).cpu().numpy(), captured[0]
                )
            rows.append(row)
            write_json(args.output / "measurements.json", rows)
            print(
                f"{phase} {item['id']} take {repeat}: ready={arrivals[0]:.3f}s "
                f"total={elapsed:.3f}s duration={duration:.2f}s RTF={elapsed / duration:.3f} "
                f"VRAM={observed_memory['sampled_process_vram_mib']:.0f}MiB",
                flush=True,
            )
        if paced is not None:
            write_json(
                args.output / "paced-playback.json",
                {
                    "physical_playback_measured": False,
                    "description": "Clock-paced CPU sink during GPU synthesis; no audio device",
                    "events": paced.finish(),
                },
            )
    except Exception as error:
        summary["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        if paced is not None:
            paced.stop()
        summary.update(sampler.stop())
        warm = [row for row in rows if row["phase"] == "warm"]
        if warm:
            summary["warm_median_audio_ready_seconds"] = statistics.median(
                row["audio_ready_seconds"] for row in warm
            )
            summary["warm_median_rtf"] = statistics.median(row["rtf"] for row in warm)
        write_json(args.output / "measurements.json", rows)
        write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
