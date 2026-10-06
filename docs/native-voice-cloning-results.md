# Native cloning on GTX 1660 SUPER

Three models completed the same eight sentences with three warm takes per sentence:
72 warm WAVs, plus three first-inference WAVs. All used the same 12.26-second reference
and cached native conditioning. No fine-tuning, voice conversion, cloud compute or
runtime voice replacement was performed.

Open `data/voice/benchmarks/native-cloning/index.html` in a browser. It works offline,
plays every take, includes all references/transcripts and shows ASR disagreements and
speaker proxies. Listening notes can be saved locally or exported as JSON. Audio files,
model weights, environments, downloads, manifests and caches are ignored by git.

**Technical recommendation: Chatterbox Turbo.** It had the lowest latency, lowest peak
VRAM and ample generation throughput in this experiment. CosyVoice deserves a careful
similarity listen: it led both embedding proxies, despite much poorer interactive speed.
Qwen is another listening option, with good recognized text but a near-capacity GPU peak.
Human judgments of similarity, warmth, naturalness and artifacts remain pending.

## Measured performance

| Model | Fresh-process model load | Reference cache | Median warm audio ready | Median RTF | Peak process VRAM | Peak process RAM |
|---|---:|---:|---:|---:|---:|---:|
| Chatterbox Turbo | 11.17 s | 0.70 s | 1.26 s | 0.400 | 3,792 MiB | 4,863 MiB |
| CosyVoice3 0.5B | 20.61 s | 0.40 s | 12.85 s | 2.339 | 4,882 MiB | 6,226 MiB |
| Qwen3-TTS 0.6B Base | 4.94 s | 0.83 s | 8.26 s | 2.113 | 5,406 MiB | 2,480 MiB |

Model loading begins after PyTorch import/CUDA initialization, includes adapter imports,
and runs in a new process with no model resident. OS file caches were not flushed.
Turbo's earlier first load took 19.01 seconds; the comparison process took 11.17 seconds.
These are model-load measurements, not entire application startup times. RAM includes
temporary loading allocations. NVML process peaks are sampled at 100 ms; per-request
exact PyTorch allocation/reservation peaks are also preserved in JSON.

RTF = generation time / generated audio duration; below 1 means faster than playback.
The same sentence distribution is used for all medians. Audio-ready time includes obtaining
finite PCM on the CPU, not audio-device playback or LLM/text-chunk arrival. First inference
is reported separately. Seeds 43, 44 and 45 cover repeated warm requests without discarding
unflattering takes. Longer generated utterances differ in duration/prosody across models.

| Long response, three takes | First usable PCM | Total synthesis | Native chunks |
|---|---:|---:|---:|
| Turbo | 7.36–9.00 s | 7.36–9.00 s | 1 |
| CosyVoice3 | 14.81–15.05 s | 68.88–71.01 s | 5 |
| Qwen | 37.70–48.55 s | 37.70–48.55 s | 1 |

Turbo's stock API returns a complete waveform. Qwen's tested Python wrapper also decodes
and returns the complete waveform; `non_streaming_mode=False` is not observable live PCM
streaming through this API. CosyVoice's `stream=True` genuinely yields PCM chunks, but
arrivals lag playback. For the measured long response, starting about 49–51 seconds after
the request would be necessary to avoid underruns in an ideal playback buffer, calculated
from chunk arrival times/durations. This is a buffer simulation, not a playback measurement.
The HTML's pre-generated WAVs concatenate chunks and therefore have no synthesis gaps.

These tests ran on the actual GTX with NVIDIA driver 610.57.04, CUDA 12.4/PyTorch 2.6.0,
compute capability 7.5, and an i5-10400F. CUDA reports 5,747.56 MiB addressable memory;
NVML reports 6,144 MiB physical memory. TTS ran without resident RAPHAEL STT/LLM models.
Desktop GPU use was included separately: whole-GPU peaks were 4,623 MiB for Turbo,
5,584 MiB for CosyVoice, and 6,110 MiB for Qwen. Qwen's long response left only about
34 MiB physical headroom at the observed peak. Shorter text chunks and STT coexistence
must be verified before using it in an always-running assistant.

## Reference material

Source: [PYmd20HsBj4](https://www.youtube.com/watch?v=PYmd20HsBj4), Clem Whispers,
14 minutes 25 seconds, English original audio. The selected download was `251-7`,
Opus 48 kHz stereo, about 9.46 MB; its metadata identifies English (US) original/default.
The original WebM and native lossless FLAC decode are preserved with SHA-256 provenance.
The lossless working format does not undo YouTube compression.

| Reference | Original timestamps | Duration | Music tag | Source-speaker cosine | Review |
|---|---|---:|---:|---:|---|
| Primary | 8:25.36–8:37.62 | 12.26 s | 0.023 | 0.931 | Continuous conversational passage |
| Alternative 2 | 5:24.07–5:37.06 | 12.99 s | 0.020 | 0.973 | Noticeable pauses |
| Alternative 3 | 4:34.84–4:46.10 | 11.26 s | 0.018 | 0.972 | Noticeable pauses |

WAV/TXT/JSON files are under `data/voice/references/raphael/`; exact transcripts appear
on both listening pages. Both cached ASR models agreed on the selected words. No clipping
was detected in selected passages. Scores are diagnostic: speaker confidence/overlap
probabilities remain null. Identity, delivery and absence of overlap need a brief human
check. The source is a soft spoken ASMR roleplay, so cloning may inherit breathiness.

Preparation ranked 49 overlapping candidates and exported 22 spaced alternatives for
speaker/sound analysis. Three references total 36.51 seconds. There is no final training
dataset: accepted count is zero pending review; the shortlist has 22 provisional clips,
and no hard clipping/signal rejects. The broader candidate count is not a dataset count
or a filtered dataset duration. Training data should wait for a model choice.

Turbo internally uses up to 15 seconds for encoder conditioning and 10 seconds for its
decoder prompt. Other candidates used the same supplied file through their native
conditioning paths. Those model-specific internal limits are retained for this comparison.

## Content, similarity and stability

Two existing CPU speaker encoders were compared on the same 15 warm recordings that
were at least three seconds long across every model. Very short clips are excluded from
this aggregate and flagged individually on the page.

| Model | Mean Chatterbox encoder cosine | Mean CAMPPlus cosine | Normalized ASR word-edit proxy |
|---|---:|---:|---:|
| Turbo | 0.925 | 0.802 | 2 / 360 |
| CosyVoice3 | 0.936 | 0.861 | 17 / 360 |
| Qwen | 0.912 | 0.744 | 0 / 360 |

Neither cosine is a calibrated measure of perceived similarity. Chatterbox uses its own
encoder during conditioning, and CosyVoice uses CAMPPlus; the proxies may favor those
systems. These values do not score naturalness, warmth or emotion. ASR can mishear,
hallucinate words on short audio, or format spoken numbers differently. Explicit normalization
handles the suite's decimal/time/acronym spellings and `work station` tokenization; raw
recognition text is retained. Qwen's zero word edits do not clear its 0.24-second short take.

All final configurations completed 24 warm requests without a synthesis crash or invalid
waveform. Problems found and diagnosed before/during those runs:

- CosyVoice's stock `fp16=True` enables autocast while keeping FP32 weights resident.
  Its first synthesis hit an OOM (260 MiB requested with about 16.6 MiB free). The worker
  was stopped; that oversized configuration was not repeated. Resident FP16 LLM/flow
  then produced non-finite PCM. The successful configuration keeps only the LLM in
  FP16, restores flow/HiFT to FP32, and disables AMP. It is slow and leaves little headroom.
- CosyVoice's older dependencies needed compatibility fixes: ONNX Runtime GPU 1.24.4
  follows the [CUDA 12/cuDNN 9 compatibility guidance](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
  newer Whisper fixes its old build failure, and setuptools <81
  restores Matcha's `pkg_resources` import. Its blank Qwen backbone is initialized from
  official local config, then fully overwritten by strict `llm.pt` loading; no duplicate
  blank weights are downloaded. This loader adjustment changes no checkpoint architecture.
- Its optional WeText/ModelScope normalizer returned HTTP 403. Inference uses the native
  inflect fallback without a network dependency. That fallback renders `4.2` as `four.two`;
  all three technical takes have decimal-recognition problems. One serious take and one
  very short take are also flagged by ASR; one long take may omit the final sentence.
  Fixing numeric text handling should precede any training experiment.
- Qwen's FP16 run produced invalid sampling probabilities and a CUDA device assertion.
  A fresh process passed tiny BF16 attention/GEMM/convolution tests. BF16 then produced
  valid PCM throughout the full benchmark. This GPU has no native BF16 support; PyTorch
  uses supported emulation here. No FlashAttention2 was installed. One warm `Of course`
  take lasts only 0.24 seconds and needs a listening check.
- Turbo has one `online`/`on mine` ASR disagreement. Listen before treating this as an
  actual pronunciation error. Its final one-sentence proof also succeeded with Python
  internet connections and DNS blocked.

Network calls are blocked by Python audit hooks during inference; all assets load locally.
GPU worker exceptions are surfaced to the main process, and each request has a timeout,
avoiding CosyVoice's otherwise indefinite polling after an LLM thread fails. Invalid PCM
stops the run; no automatic OOM retry is performed. Diagnostic attempts remain separate
from the final comparison folders.

Chatterbox Nano was deferred: Turbo already completed the workload comfortably and was
the lowest-VRAM successful comparison. Another lower-resource download is not necessary
to choose the first voice. No additional Piper optimization/benchmarking was performed.

**Training decision:** there is no demonstrated need for fine-tuning yet. First judge
these zero-shot samples. Precision compatibility, decimal handling, short-answer
generation settings and chunking are inference issues; more training time is not a
diagnosis for them. If a chosen model's voice already meets the target, retain reference
conditioning. Any later training proposal must justify an audible improvement and first
pass the user's short validation experiment and approval gates.

## Exact versions and reproduction

| Candidate | Official weights and pinned revision | Code revision |
|---|---|---|
| Turbo | `ResembleAI/chatterbox-turbo` · `749d1c1a46eb10492095d68fbcf55691ccf137cd` | Chatterbox `5de7a54aa4e5e2baadb0182dde554908b48b85c2` |
| CosyVoice | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` · `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`; Matcha `dd9105b34bf2be2230f4aa1e4769fb586a3c824e` |
| Qwen | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` · `5d83992436eae1d760afd27aff78a71d676296fc` | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |

The download utility inventories/pins one model at a time and excludes duplicate/training
files. Selected sizes were 2.99 GB Turbo, 4.44 GB CosyVoice and 2.52 GB Qwen, in decimal
bytes. Its per-model `inventory.json` records every selected file and repository revision.
Tested dependency snapshots are in `scripts/voice_requirements/{clone,cosy,qwen}.txt`.
Use Python 3.12 in isolated environments under `data/voice/envs/`, leaving `.venv` intact.
The GPU-compatible Torch 2.6/CUDA 12.4 build includes `sm_75`; newer builds must not be
substituted without checking their supported architectures.

For a new setup, clone each official code repository at the listed revision; initialize
CosyVoice's pinned Matcha submodule. Install each requirements snapshot in its matching
isolated environment using PyPI and a shared local uv cache. Use `--no-deps` when installing
these complete snapshots, preserving the recorded package set:

```bash
uv pip install --no-deps --python data/voice/envs/clone/bin/python \
  --cache-dir data/voice/cache/uv -r scripts/voice_requirements/clone.txt
uv pip install --no-deps --python data/voice/envs/cosy/bin/python \
  --cache-dir data/voice/cache/uv -r scripts/voice_requirements/cosy.txt
uv pip install --no-deps --python data/voice/envs/qwen/bin/python \
  --cache-dir data/voice/cache/uv -r scripts/voice_requirements/qwen.txt
```

The Qwen snapshot installs
its inference package without the optional Gradio UI; its UI dependency is intentionally
absent. Review one inventory with `--inventory-only` before fetching that candidate:

```bash
.venv/bin/python scripts/fetch_voice_model.py chatterbox-turbo --inventory-only
.venv/bin/python scripts/fetch_voice_model.py chatterbox-turbo
```

Prepare references using [the preparation instructions](voice-reference-preparation.md).
Use new output directories for every benchmark; existing results cannot be overwritten.
These commands use the installed assets and create the final comparison:

```bash
data/voice/envs/clone/bin/python scripts/benchmark_cloning_voice.py \
  --candidate chatterbox-turbo --model data/voice/models/chatterbox-turbo \
  --reference data/voice/references/raphael/reference-1.wav \
  --reference-text data/voice/references/raphael/reference-1.txt \
  --output data/voice/benchmarks/new-comparison/chatterbox-turbo --repeats 3

data/voice/envs/cosy/bin/python scripts/benchmark_cloning_voice.py \
  --candidate cosyvoice3 --vendor data/voice/vendor/cosyvoice \
  --model data/voice/models/cosyvoice3 \
  --reference data/voice/references/raphael/reference-1.wav \
  --reference-text data/voice/references/raphael/reference-1.txt \
  --output data/voice/benchmarks/new-comparison/cosyvoice3 \
  --repeats 3 --request-timeout 300

data/voice/envs/qwen/bin/python scripts/benchmark_cloning_voice.py \
  --candidate qwen06 --qwen-precision bfloat16 --model data/voice/models/qwen06 \
  --reference data/voice/references/raphael/reference-1.wav \
  --reference-text data/voice/references/raphael/reference-1.txt \
  --output data/voice/benchmarks/new-comparison/qwen06 \
  --repeats 3 --request-timeout 300

.venv/bin/python scripts/build_voice_comparison.py \
  data/voice/benchmarks/new-comparison/chatterbox-turbo \
  data/voice/benchmarks/new-comparison/cosyvoice3 \
  data/voice/benchmarks/new-comparison/qwen06 \
  --references data/voice/references/raphael/references.json \
  --output data/voice/benchmarks/new-comparison/index.html
```

Run GPU benchmarks sequentially. `--preflight --qwen-precision bfloat16` performs small
kernel tests; `--smoke-only` synthesizes one short response. Do those before a full new
candidate/configuration. The shared evaluation suite is `docs/voice-evaluation.json`.
Ctrl-C stops a benchmark; completed WAVs and incremental measurements remain available.
Model/conditioning load once per run and are reused for every sentence.

Optional postchecks run after timing: `scripts/assess_cloning_audio.py` uses a specified
existing faster-whisper path; `scripts/measure_voice_similarity.py` uses `--encoder
chatterbox` or `campplus` with the already downloaded weights. Each accepts the benchmark
directories; the comparison builder automatically includes any generated checks.

Validation: 583 repository tests passed; Ruff checks passed for `src`, `tests` and all
new voice scripts. Runtime configuration remains Piper / `en_US-raphael-medium` / speed
0.8, and Amy's fallback weights remain available. Integration is deferred until listening
and model selection.
