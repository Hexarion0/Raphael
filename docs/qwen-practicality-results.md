# Qwen3-TTS 0.6B Base on GTX 1660 SUPER: inference experiments

Measured on 2026-10-05. **Qwen now fits comfortably as a standalone TTS worker and is
plausible for deliberate, short conversational replies. It is not yet a consistently
smooth real-time assistant voice.** Complete short replies generally take 2.4–3.5 seconds
with the better short-text configuration; one longer annoyed reply takes 5.12 seconds.
Experimental audio streaming starts earlier, but still produces audible gaps.

The strongest optimization keeps the existing BF16 parameters, sampling and reference.
It reproduces the stock output sample for sample on the matched evaluation takes. It
therefore preserves the existing emotional delivery and speaker reproduction on those
takes. Other settings change the output and require listening before adoption.

No training, paid compute, quantization, new model downloads, voice conversion, or
production TTS/configuration changes occurred. Piper remains RAPHAEL's runtime voice.
Chatterbox Turbo remains the performance and user-rated identity baseline.

## Listen and inspect the evidence

Open `data/voice/benchmarks/qwen-optimized/index.html` locally in a browser. It links to:

- `short.html`: eight assistant-sized replies, default versus full-text conditioning.
- `words.html`: original text versus normalization and temperature 0.8.
- `short-controls.html`: very short replies and the minimum-length experiment.
- `screens.html`: attention and precision experiments on three fixed sentences.

The main page compares Turbo, archived stock Qwen, optimized Qwen, full-text conditioning,
and two experimental PCM streaming configurations. Every recorded take is available;
failed or truncated content is retained. The source reference is playable on every page.
Three timing replays at the bottom insert the measured pauses into the long response.
They omit the initial wait, which is displayed beside the player. Ordinary comparison
WAVs contain the complete generated audio without generation waits.

Raw measurements, stage timings, code arrays, ASR checks, speaker proxies, memory peaks,
equivalence checks and replay schedules are in the same ignored experiment directory.
`stock-view` and the short-control views link archived audio; they are not new GPU runs.

## Same speaker, pinned implementation, bounded methodology

- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, weights revision
  `5d83992436eae1d760afd27aff78a71d676296fc`.
- Installed and current official code revision:
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`. The inference wrapper, core generation code
  and 12 Hz codec source were compared byte for byte with that official revision;
  all three match. See `official-revision.json`.
- Python 3.12.14, PyTorch 2.6.0+cu124, Transformers 4.57.3, qwen-tts 0.1.1;
  dependency snapshot: `scripts/voice_requirements/qwen.txt`.
- GPU: GTX 1660 SUPER, compute capability 7.5, driver 610.57.04; i5-10400F, 16 GB RAM.
  NVML reports 6,144 MiB physical VRAM; CUDA reports 5,747.56 MiB addressable VRAM.
- Primary reference: `data/voice/references/raphael/reference-1.wav`, 12.26 seconds,
  [PYmd20HsBj4, 505.36–517.62 seconds](https://www.youtube.com/watch?v=PYmd20HsBj4&t=505).
  SHA-256: `c23d11bc85d32115fb216e953d384d66cf72801096360b6c68c54e2c9557acdc`.
  The exact transcript and native in-context cloning prompt are unchanged.
- The eight original sentences plus seven previous emotion sentences are unchanged.
  Main warm takes use seeds 43 and 44; streaming/full-text comparisons use seed 43.
  Additional short replies include caring, concerned, serious, happy, playful, sleepy
  and annoyed wording. These labels describe requested content, not verified emotions.
- Batch one; temperature 0.9, top-k 50, top-p 1, repetition penalty 1.05;
  subtalker temperature 0.9/top-k 50/top-p 1. The temperature experiment changes only
  the main temperature to 0.8. Output is bounded at 2,048 generated frames.
- All timed TTS runs were sequential, without resident RAPHAEL STT or a local LLM.
  GPU ASR ran afterward. Python network access was blocked during TTS generation.
  No OOM retry loop was used. The harness checks free memory and limits each request
  to 120 seconds; the FP32 talker experiment is restricted to short text.

Model load is a fresh-process load after Torch/CUDA initialization, including Qwen
imports and weights. OS file caches were not flushed. It is not complete application
startup time. Audio-ready means finite PCM has reached CPU memory; it excludes LLM
text arrival, sound-device buffering and the audio's own leading silence. The newer
harness records signal-onset/first-packet information as well. Naturalness and emotion
ratings remain unfilled for the user to judge.

Process and whole-GPU memory are sampled with NVML every 100 ms. Exact Torch allocation
and reservation peaks are also saved per request. Whole-GPU totals include desktop use,
which varies between runs. RAM peaks include temporary loading allocations.

## What changed the eight-second wait

The previous original-eight-sentence median was 8.26 seconds across three warm takes.
It already loaded the model and processed the reference once. Repeating CUDA setup or
re-encoding the reference was not the cause of that warm wait.

The [official wrapper](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py)
waits for every speech code, then decodes the waveform and returns it. In a representative
stock server-status take, speech-code generation took about 6.84 seconds, decoding 0.72
seconds, and text tokenization less than a millisecond. Full-waveform buffering hides
early generated speech, but most of the delay is genuine autoregressive computation.

The talker generates a first code, then runs a five-layer predictor for the remaining
15 code groups for each speech frame, at about 12.5 frames per second. The stock path
re-enters Hugging Face generation for that small predictor repeatedly. The benchmark's
compact path uses the same native KV-cached forward and sampling processors directly;
the graph path captures only its fixed 15-step operation in a reusable CUDA graph.
The growing 28-layer talker remains native generation. This is an experimental local
inference adapter, not an official Qwen optimization or an architecture change.

The graph costs about 0.55 seconds to prepare once, then stays warm. It is restricted to
batch one and one fixed sampling signature per process. Model, prompt, decoder and graph
state are mutable: inference requests must be serialized through one GPU worker.

Matched **30 warm takes**: the same 15 sentences, seeds 43 and 44:

| Configuration | Median audio ready | Median RTF | Sum of synthesis time | Peak process VRAM | Whole-GPU peak | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| Archived stock BF16 | 9.55 s | 2.093 | 332.30 s | 5,406 MiB | 6,110 MiB | 2,550 MiB |
| Compact predictor + memory changes | 8.90 s | 1.946 | 308.35 s | 3,512 MiB | 4,300 MiB | 2,473 MiB |
| Predictor graph + memory changes | **5.43 s** | **1.193** | **187.81 s** | **3,616 MiB** | **4,404 MiB** | **2,463 MiB** |

This is about 43% less median wait and 33% less peak process VRAM. On just the original
eight sentences, the matched two-take median fell from 8.55 to 4.93 seconds. This matched
8.55-second baseline differs from the earlier 8.26-second three-take median.

The compact and graph paths reproduce all matched warm stock PCM exactly. All 31 graph
requests, including first inference, match compact code arrays and PCM exactly. Both
corrected streaming variants also match all 16 corresponding graph requests exactly.
See `equivalence.json`, `graph-equivalence.json`, `stream12-equivalence.json`, and
`stream24-equivalence.json`. These comparisons establish preservation on the tested set,
not a guarantee for arbitrary text or another model/dependency revision.

Safe memory changes were an expandable Torch allocator, caching native conditioning,
and moving the reference-only speaker/tokenizer encoders to CPU after conditioning.
Offloaded parameters total about 92 MiB. The large reduction is mainly reduced allocator
reservation/fragmentation, not smaller weights: an earlier long take reserved about
5,296 MiB while allocating about 3,346 MiB. No quantization or second audio model is used.

Fresh optimized model load takes about 4.97–5.12 seconds. Native conditioning takes about
0.82 seconds; a validated tensor cache takes about 0.0015 seconds to load. The cache
stores the speaker embedding, reference codes and transcript, with reference/model/code
fingerprints and `weights_only=True` loading. Invariant reference text tokens are cached;
dynamic request tokenization remains negligible. Cached conditioning improves startup,
not the old already-cached eight-second warm synthesis time.

The archived Turbo run loaded in 12.13 seconds and conditioned in 0.77 seconds; archived
stock Qwen loaded in 4.94 seconds and conditioned in 0.83 seconds. These are the same
fresh-process measurement boundaries, subject to the file-cache caveat above.

## Streaming and sentence pipelining

`non_streaming_mode=False` in the installed API simulates streaming **text conditioning**.
Its own documentation explicitly says that it does not enable real input/output streaming.
The stock `generate_voice_clone` API returns complete waveforms. `True` changes the
conditioning layout; it is not a switch that enables PCM streaming.

The benchmark adds a genuine early-code callback and a CPU PCM queue around the native
talker and causal decoder. It keeps the stock 300-code/25-code-left-context boundaries,
recomputes incomplete decoder prefixes, and emits only newly available samples. It uses
the existing Qwen decoder, with no voice conversion. Earlier audio costs extra decoding
work, explaining the worse total throughput.

Matched **15 warm takes**, seed 43, including the original and emotion sentences:

| Configuration | First usable PCM, median | Total synthesis, median | Median RTF | Process VRAM peak | Whole-GPU peak |
|---|---:|---:|---:|---:|---:|
| Turbo, archived baseline | 1.85 s | 1.85 s | 0.447 | 3,416 MiB | 4,253 MiB |
| Stock Qwen, archived | 9.07 s | 9.07 s | 2.104 | 5,406 MiB | 6,110 MiB |
| Optimized Qwen, complete waveform | 5.17 s | 5.17 s | 1.205 | 3,616 MiB | 4,404 MiB |
| Optimized Qwen, full-text conditioning | 5.29 s | 5.29 s | 1.212 | 3,600 MiB | 4,403 MiB |
| Optimized Qwen, 12-frame blocks (~0.96 s) | **1.73 s** | 8.09 s | 1.873 | 3,656 MiB | 4,444 MiB |
| Optimized Qwen, 24-frame blocks (~1.92 s) | **2.69 s** | 6.67 s | 1.467 | 3,554 MiB | 4,357 MiB |

The memory peaks cover each complete run; warm latency medians use the matched take
subset. Very short outputs remain included and flagged. RTF = synthesis time/audio
duration; above 1 cannot sustain immediate uninterrupted playback without buffering.

For the same long-response take:

| Path | Initial wait | Synthesis duration | Audio duration | Playback gaps after starting | Request to playback finish |
|---|---:|---:|---:|---:|---:|
| Complete optimized waveform | 20.15 s | 20.15 s | 18.64 s | 0 | 38.79 s |
| 12-frame streaming | 1.73 s | 33.31 s | 18.64 s | 13.45 s | 33.82 s |
| 24-frame streaming | 2.66 s | 26.35 s | 18.64 s | 6.55 s | 27.85 s |
| Seven sentence chunks, overlapped CPU sink | 3.11 s | ~26.4 s, sum | 20.56 s | 4.85 s | 28.53 s |

Playback schedules are measured-arrival calculations, not physical speaker recordings.
The clock-paced CPU sink ran during GPU synthesis: all 14 adjacent requests overlapped
the previous audio's duration, totaling 24.64 seconds of overlap. Audio remains on CPU;
there is never concurrent Qwen GPU inference. A bounded queue and cancellation checks
were exercised. This proves scheduling overlap without a second GPU model, but does
not validate PortAudio latency, interruption handling or a warmed whole assistant.

The paragraph's seven sentence boundaries come from RAPHAEL's existing `SentenceBuffer`;
decimal numbers and 9:30 AM stay intact. All text was preavailable in this experiment.
In RAPHAEL, the existing LLM reader can continue collecting sentence N+1 while a persistent
worker synthesizes N and a separate CPU player consumes N-1. The current blocking
`speak` path would need a queue to overlap synthesis and playback. No such production
change was made. A local LLM sharing this GPU could change memory/throughput substantially.

Pipelining reduces repeated waits but does not eliminate them: the long second sentence
leaves a 4.85-second gap. Splitting sentences changes prosody and generated duration.
PCM blocks can introduce pauses within words when synthesis falls behind playback.
Listen to the timing replays before favoring their low first-audio figure.

The current [official vLLM-Omni example](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/text_to_speech/)
provides genuine Qwen PCM streaming, including the 0.6B family. It is a separate serving
implementation. Its [Qwen stage configurations](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/)
default to BF16; inspected fused-predictor code requires BF16, and some prompt/encoder
paths explicitly use BF16. The [generic vLLM CUDA requirement](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
includes compute capability 7.5, so this is not proof that all vLLM inference is unsupported
on Turing. That large dependency stack was not installed or benchmarked here. No claimed
latency or fit advantage is attributed to it. A stateful codec cache could avoid prefix
redecoding, but this is an unmeasured engineering possibility, not an achieved optimization.

## FP16 diagnosis and attention/precision results

The FP16 failure is numerical overflow, not proof that the GPU cannot execute FP16.
Instrumentation located the first nonfinite output at
`talker.code_predictor.model.layers.2.mlp.down_proj`. Its input was already infinite;
the gated multiplication of otherwise finite projections overflowed FP16's range.
Native conditioning was finite. The old CUDA probability assertion was a downstream
symptom. The guarded diagnostic stopped before another device assertion.

An experimental repair performs the gated products and down projections in FP32 in all
five predictor MLPs, with other talker parameters FP16 and native FP32 rotary buffers
preserved. It produces finite audio, but did not beat the BF16 graph path. It is a local
experiment, not a supported official precision preset. Full FP32 talker works on short
text but uses 5,038 MiB process VRAM and leaves insufficient long-response headroom;
long FP32 runs were deliberately skipped.

The GTX has no native BF16 execution. In this Torch build, BF16 SDPA selects the math
backend. Forced Flash and efficient BF16 backends fail with no available kernel.
FP16 efficient attention works in an isolated probe, but does not repair the MLP overflow.
The official [FlashAttention-2 implementation](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)
targets Ampere/Ada/Hopper; its separate Turing implementation was not installed.

Three-sentence screens use the same reference, sentences and seed. These are bounded
comparisons, not full-suite quality evidence:

| Screen | Median first PCM | Median RTF | Process VRAM peak | Result |
|---|---:|---:|---:|---|
| Stock BF16 SDPA, expandable allocator | 7.56 s | 2.198 | 3,168 MiB | Reference screen |
| Compact BF16 SDPA | 6.73 s | 1.955 | 3,166 MiB | Same tested codes |
| Graph BF16 SDPA | **4.23 s** | **1.230** | 3,194 MiB | Same tested codes and PCM |
| Compact BF16 eager attention | 8.04 s | 2.095 | 3,230 MiB | Slower; changed output |
| Repaired FP16 graph, FP32 buffers | 8.02 s | 1.699 | 3,498 MiB | Finite; slower; changed output |
| FP32 talker, BF16 decoder, compact | 5.79 s | 1.767 | 5,038 MiB | Short text only; little headroom |
| BF16 graph, FP32 decoder, 12-frame blocks | 1.56 s | 1.653 | 3,280 MiB | Finite; changed PCM; listening pending |
| BF16 graph, FP16 decoder, 12-frame blocks | 2.99 s | 3.315 | 2,830 MiB | Finite; substantially slower |

The earlier `screen-mixed` prototype also rounded positional buffers; the corrected
`screen-mixed-graph` is the controlled repaired-FP16 comparison above. Short screen peaks
are not predictions of long-response memory. No quality-preserving alternative to the
tested BF16 graph path was demonstrated. No quantization was attempted.

One benchmark bug was caught by PCM comparisons: a redundant `.decoder.to(bfloat16)`
rounded the decoder's native FP32 positional buffers. Three intermediate directories
were archived with an exclusion reason (`*-buffer-cast`) and omitted from the listening
pages and results. The corrected 24-frame streaming and short-reply runs were regenerated;
both streaming paths now match stock PCM exactly. Nothing was silently overwritten.

## Words, speaker identity and emotion

Automatic transcription is an error proxy, not a correctness certificate. Cached
distil-whisper-large-v3 checked generated audio; cached medium.en independently checked
11 ambiguous/high-value takes. Both raw results appear on the page. Numeric/acronym
spellings and `alright`/`all right` are normalized, while contractions and actual omitted
words remain disagreements. Recognizers disagree on some ambiguous audio.

- Stock/optimized main suite: one substitution in 30 warm takes, sleepy `we` → `you`,
  confirmed by both ASR models. Exact PCM optimization preserves this error too.
- Eight additional short replies, default conditioning: `You did it!` became a 0.32-second
  single word and `All right, you win.` a 0.24-second fragment. Both recognizers flag
  the omissions. Their subsecond timings are not evidence of useful fast replies.
- Official full-text conditioning fixed those two replies at 2.42 and 2.75 seconds.
  All eight short warm replies had zero primary-ASR disagreements, median 2.67 seconds,
  peak process VRAM 3,134 MiB. This is one take per sentence, not proof of robustness.
- Full-text conditioning on the original/emotion suite still truncates `Of course.`
  to `of` and has a two-edit long-response disagreement (`I've finished` / `I finish`).
  Both recognizers agree on these flags. The default 0.24-second `Of course` take also
  remains unusually short despite both recognizers returning the expected words.
- Sentence chunking had one `server is` / `servers` disagreement with the primary ASR;
  medium.en recognized the expected phrase. Do not count this as a confirmed model error.
- Explicit numeric spelling showed no accuracy benefit on these already-correct numbers.
  Changing sleepy punctuation to a comma corrected the known `we`/`you` flag in two takes.
  Normalized long text still had an `I've` / `I` disagreement in both takes and both ASRs.
- Temperature 0.8 on the original GPU and sleepy sentences: all six warm takes had zero
  primary-ASR disagreements; the matched sleepy substitution disappeared. Emotion and
  speaker identity must be listened to; this small result is not a general solution.
- Experimental minimum eight code frames prevented several tiny replies, but one happy
  take lasted 11.76 seconds and the primary ASR heard repetition. Medium.en transcribed
  only one occurrence, demonstrating ASR uncertainty. The long duration itself is real.
  Forcing minimum length is not recommended as a blanket fix.

Default short-text truncation is premature termination under that conditioning/sampling
configuration. It is not established that unseen later text caused it: this reference's
155 codes allow the short target texts into the stock prefill. Full-text conditioning
changes that layout and improves some tested phrases, but does not fix every short reply.
Increasing training duration would not diagnose these inference failures.

Two CPU speaker-embedding proxies were measured separately from timed synthesis.
On 12 common seed-43 sentences with at least three seconds of audio in every compared
profile, median cosine scores are:

| Profile | Chatterbox encoder | CAMPPlus |
|---|---:|---:|
| Turbo | 0.925 | 0.817 |
| Stock Qwen | 0.922 | 0.778 |
| Optimized Qwen | 0.922 | 0.778 |
| Qwen full-text conditioning | 0.923 | 0.772 |

These are uncalibrated, encoder-dependent identity proxies, not perceptual ratings.
Very short audio is flagged rather than used to claim similarity. The unchanged graph
and streaming audio reuse archived ASR/embedding measurements only after exact PCM
verification; records identify the original analysis source. Changed settings received
fresh analysis. Neither ASR nor embedding similarity measures emotional quality.

The installed **0.6B Base cloning API has no explicit `instruct` emotion selector**.
Do not transfer controls from the 1.7B CustomVoice/VoiceDesign models to this checkpoint.
Its emotionally alive delivery in the earlier listening test is real user feedback,
but controlled caring/happy/annoyed states remain a separate unresolved requirement.
The graph optimization preserves the liked delivery in the tested samples; it does not
add an emotion knob. Full-text, temperature and precision changes need the user's A/B
judgment before claiming improvement or equal emotional range.

## Practical verdict and stopping point

The 6 GB card is no longer the immediate obstacle: optimized standalone Qwen peaks at
about 3.53 GiB process VRAM, 4.30 GiB including the desktop, and 2.41 GiB process RAM.
The 198 completed requests across 19 bounded configurations produced finite audio
without an OOM or synthesis crash. Numerical failures, superseded buggy runs and
content defects are documented separately; stable generation does not imply correct words.

Qwen is a realistic option if a 2–5-second first sentence and occasional longer wait
are comfortable, with full-text conditioning promising for short replies. It is not
ready to promise uninterrupted low-latency speech: default short-answer reliability,
remaining word errors, streaming gaps and whole-assistant STT coexistence need resolution.
Turbo remains substantially faster and has the user's strongest identity rating.

The audit found a persistent GPU STT model and no shared GPU memory reservation; an
optional retry STT model can also load. Standalone headroom does not prove they fit
together. A future Qwen integration would use one persistent Python 3.12 worker because
RAPHAEL's main environment is Python 3.14, load native prompts once, and serialize GPU
synthesis while a CPU audio queue plays. That work is outside this stopped experiment.

No evidence here justifies training to solve speed. Listen to the main, short, word and
timing comparisons before choosing Qwen versus Turbo or changing RAPHAEL's runtime.

## Reproduce with existing local assets

Run from the repository root with the pinned isolated environment described in
[the original benchmark report](native-voice-cloning-results.md#exact-versions-and-reproduction).
Use fresh output directories; the harness refuses to overwrite evidence. To recreate
the combined evaluation suite without replacing a different manifest:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('data/voice/benchmarks/qwen-optimized')
root.mkdir(parents=True, exist_ok=True)
suite = json.loads(Path('docs/voice-evaluation.json').read_text())
suite += json.loads(Path('docs/voice-emotion-evaluation.json').read_text())
path = root / 'suite.json'
if path.exists():
    assert json.loads(path.read_text()) == suite
else:
    path.write_text(json.dumps(suite, indent=2) + '\n')
PY

data/voice/envs/qwen/bin/python scripts/benchmark_qwen_practicality.py \
  --output data/voice/benchmarks/qwen-reproduction/graph \
  --suite data/voice/benchmarks/qwen-optimized/suite.json \
  --label 'Qwen BF16 graph reproduction' --predictor graph --offload-encoders \
  --prompt-cache data/voice/cache/qwen-primary-prompt.pt --repeats 2

data/voice/envs/qwen/bin/python scripts/benchmark_qwen_practicality.py \
  --output data/voice/benchmarks/qwen-reproduction/stream24 \
  --suite data/voice/benchmarks/qwen-optimized/suite.json \
  --label 'Qwen native codec streaming reproduction' \
  --predictor graph --offload-encoders --stream-frames 24 \
  --prompt-cache data/voice/cache/qwen-primary-prompt.pt --repeats 1

data/voice/envs/qwen/bin/python scripts/benchmark_qwen_practicality.py \
  --output data/voice/benchmarks/qwen-reproduction/shorts \
  --suite docs/voice-qwen-short-evaluation.json \
  --label 'Qwen short full-text reproduction' \
  --predictor graph --offload-encoders --non-streaming-mode --paced-sink \
  --prompt-cache data/voice/cache/qwen-primary-prompt.pt --repeats 1
```

The last command also runs the paragraph chunks; the original short-full-text experiment
used only the first eight entries of that suite. Defaults are BF16/SDPA, expandable
allocator, batch one, the same reference and sampling described above. The original
graph run measured reference encoding; reproductions with an existing prompt cache
measure a cache hit instead. `--smoke-only` limits a fresh configuration to one request.
Ctrl-C stops work; completed WAVs and incremental measurements remain. Run GPU timing
jobs sequentially, and assess audio after synthesis ends.

For word experiments, use `docs/voice-qwen-word-evaluation.json --temperature 0.8` or
`docs/voice-qwen-normalization-evaluation.json` as the suite in a new output directory.
Keep these changes in their own columns. Rebuild the recorded experiment's listening
pages with `.venv/bin/python scripts/build_qwen_practicality_page.py`.

For newly generated directories, `scripts/build_voice_comparison.py` accepts any list
of benchmark directories plus `--references data/voice/references/raphael/references.json`
and `--output <new-directory>/index.html`. Optional `--only-ids` limits a comparison to
shared probe IDs. Postchecks use `scripts/assess_cloning_audio.py` with an existing local
`--asr-model`, and `scripts/measure_voice_similarity.py` with the existing encoder weights.
The former supports a separate `--checks-name word_checks-medium.json` for independent ASR.

`scripts/render_voice_timing.py <run-directory> --output <new-replay.wav>` makes a gap
replay from recorded streaming arrivals; add `--paragraph` for the measured sentence
pipeline. Replays are derived artifacts, never physical playback measurements. Downloads,
environments, caches, datasets, WAVs, code arrays and benchmark manifests remain ignored
by git; only tooling, evaluation text and this report are committed.

Validation: `.venv/bin/ruff check src tests scripts` passed. Focused benchmark/listening
tests passed (11); the standard `.venv/bin/pytest` suite passed (591 tests, three existing
live-audio integration tests deselected by repository configuration). All local media
and navigation targets on the five generated pages exist; all 90 main sentence/profile
cells have warm audio, with 126 selectable takes including first inference. Generation
equivalence and timings were checked on the actual GTX separately from these CPU tests.
