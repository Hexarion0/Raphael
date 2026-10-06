# Chatterbox Turbo workload experiments — 2026-10-05–06

These experiments do not modify production source, `.env`, voice selection, model
weights on disk, or system GPU settings. No training, downloads, cloud compute, or
voice conversion was used. Work is on `perf/turbo-efficiency-benchmarks`.

Local listening page and raw data: `data/voice/benchmarks/turbo-efficiency/index.html`.
All generated audio, traces, and telemetry are ignored by Git.

## Exact inference audit

- Chatterbox code: `5de7a54aa4e5e2baadb0182dde554908b48b85c2`;
  checkpoint: `749d1c1a46eb10492095d68fbcf55691ccf137cd`;
  Torch `2.6.0+cu124`, installed Transformers `5.2.0`.
- Profile: `voice_profiles/raphael/voice.json`; primary reference SHA-256:
  `c23d11bc85d32115fb216e953d384d66cf72801096360b6c68c54e2c9557acdc`.
  The same exact transcript is retained. Turbo does not take a reference transcript
  in its inference API. Sampling, reference normalization, watermarking, and native
  event vocabulary were unchanged.
- Production loads a persistent worker once. `prepare_conditionals` runs once:
  reference loading/resampling/loudness adjustment, reference speech tokens,
  mel features, and two speaker encoders. The cached reference speech-token embedding
  is populated on first generation. It is reused thereafter.
- Each request does text normalization/tokenization on CPU, GPT2 speech-token
  generation with a KV cache on CUDA, the **two-step** mean-flow decoder on CUDA,
  the HiFT vocoder on CUDA, one waveform transfer to CPU, then CPU Perth watermarking.
  The waveform must finish before this API returns. RAPHAEL overlaps the next
  sentence's synthesis with playback using its existing bounded queue.
- All six parameter groups are initially FP32 on CUDA. Raw parameter sizes:

  | Component | MiB | Needed after reference is cached? |
  |---|---:|---|
  | T3 speech-token generator | 1630.3 | Yes |
  | Reference speech tokenizer | 471.9 | No |
  | Flow decoder | 437.3 | Yes |
  | HiFT vocoder | 79.4 | Yes |
  | Reference speaker encoder | 26.1 | No |
  | Voice encoder | 5.4 | No |

- Worker generation already uses `torch.inference_mode()`. T3/S3Gen inference methods
  also have inference-mode decorators. Modules are in evaluation mode. Production
  has no autocast; TF32 matmul is false; cuDNN benchmark and deterministic flags are
  false. This GPU does not offer native BF16 or TF32 acceleration; neither was enabled.
- Actual transformer attention is **SDPA**, with the SM75 memory-efficient FP32
  attention kernel observed in the profiler. No FlashAttention or generic compiler
  flags were added. Profiling the normal sentence attributed approximately 55% of
  self CUDA operator time to `addmm`, 21% to efficient attention, 4.6% to concatenation,
  and 4.4% to cuDNN convolution. Kernel and operator rows overlap: do not sum both.
  Reference encoding is not a per-sentence cost to eliminate.
- Stage wall times in a representative screened normal sentence: T3 0.87 s,
  flow 0.16 s, vocoder 0.04 s, watermark 0.01 s. The dominant work is T3.
- The worker blocks on stdin between requests. Loaded-model telemetry reached P8,
  300 MHz, and approximately 16–19 W. Keeping weights resident does not itself force
  full GPU utilization. Desktop activity sometimes kept P2/P0 and approximately
  40–44 W even outside TTS. Unloading every sentence would destroy warm latency.
- Sampling contains CPU-visible EOS/invalid-logit tests and dynamic KV concatenations.
  Production has an explicit end-of-generation synchronization, but the preceding
  waveform `.cpu()` already waits for its CUDA work. Removing this alone cannot remove
  the model's compute. No unmeasured synchronization/allocator changes were integrated.
- There is no optional second voice model or CFG pass to remove. The distilled
  mean-flow decoder already uses two steps without runtime CFG. Reducing it to one
  step changes acoustic synthesis; it was not used as a presumed quality-free saving.
  Watermarking was preserved.

Implementation checked against the [pinned official source](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/tts_turbo.py).

## Method and limits

`scripts/benchmark_turbo_efficiency.py` loads local assets in the existing isolated
environment. The same seeds, text, reference, and sampling parameters are used across
variants. WAVs use float PCM so numerical differences can be measured before PCM16
rounding. Stage timing instrumentation is identical across variants and adds explicit
synchronization; it is not enabled in RAPHAEL. Profiling is a separate, slower pass.

NVML is sampled at approximately 100 ms for whole-board power, temperature, fan,
graphics/memory clocks, utilization, VRAM, and process RSS. Board values include the
desktop. Short requests have too few samples for precise utilization rankings; NVML
utilization and power also have their own internal sampling windows. Integrated energy
is approximate and excludes the fraction of a sampling interval at each boundary.

Early screening allowed a start at or below 60°C. Subsequent controlled runs waited
for five seconds at or below 57°C and, where `--fan-ceiling 0` is recorded, fan 0%.
Each sample's actual start temperature is retained. The first repeated comparison had
different starting fan speeds (0% versus 27%) and is **not** a controlled temperature
ranking. The desktop was also active during portions of the early screening.

The guard stops new work at an observed 85°C. Sensor polling and already-running CUDA
kernels allow overshoot; an early burst reached 88°C. Later hooks also check
before decoder estimators. The limit was not raised to force completion. Interrupted
runs retain completed clips, errors, and full telemetry. Draining already-generated
audio is included in the simulated playback window after a thermal stop.

Repeated tests use a two-WAV queue and a duration-matched playback thread. Their gap
measurements are **simulated**, excluding the audio device. The separate runtime script
uses real `stream_reply`, IPC, and sounddevice playback. Both use a fixture LLM stream,
not a live provider request. STT is not loaded in these efficiency runs; previous STT
coexistence results do not establish thermal safety with this cooling condition.

## Screened results

Normal sentence: “Your server is online and everything looks normal.” (2.80 s PCM).
Technical sentence: “Your GPU is at 63 degrees and memory usage is 4.2 gigabytes.”
(5.48 s FP32 PCM). The native FP16 variant generated a slightly different technical
sequence; compare its own reported duration/ASR rather than assuming identical PCM.

| Experiment | Normal ready / RTF | Technical ready / RTF | Interpretation |
|---|---|---|---|
| FP32 baseline, controlled | 1.22 s / 0.435 | 2.17 s / 0.396 | Existing normal mode |
| 12 ms pause per transformer forward | 2.13–2.19 s / 0.762–0.782 | 3.73–3.76 s / 0.680–0.686 | Same generated speech tokens; slower bursts |
| 16 ms pause per transformer forward | 2.37 s / 0.847 | 4.46 s / 0.813 | Meets the throughput target, but still hit the thermal guard |
| Reference-only encoders on CPU | 1.08 s / 0.386 | 2.01 s / 0.367 | About 0.5 GiB less allocated VRAM; no removed per-sentence compute |
| T3 autocast, FP32 weights retained | 1.54 s / 0.548 | 2.60 s / 0.475 | Slower, roughly 0.5 GiB *more* peak allocated VRAM |
| FP16 T3 weights with autocast | 1.75 s / 0.626 | 2.98 s / 0.544 | Slower; about 0.8 GiB less allocated VRAM |
| FP16 T3 and conditioning, no autocast | 1.58 s / 0.563 | 2.67 s / 0.477 | Slower; finite samples; not a thermal solution |
| CPU HiFT vocoder | 1.88 s / 0.672 | 3.64 s / 0.664 | Small GPU stage moved to CPU; much slower vocoder |

These are measurements, not claims of statistical significance. Offload's slight speed
difference is not proven causal. Its ~503.5 MiB parameter saving is directly attributable
to relocation. Because S3Gen's original `device` property follows its reference
tokenizer, the isolated offload experiment overrides that property to follow the flow
decoder. Naively calling `tokenizer.cpu()` alone would break inference.

FP16 allocated peaks were roughly 2.2–2.3 GiB versus 3.0–3.1 GiB FP32. The load-then-cast
experiment retained allocator reservations, so **process VRAM did not fall by the full
allocated-memory saving**. This is not a demonstrated production memory reduction.
Offload empties the allocator once after moving reference modules; no experiment clears
the allocator on every sentence.

For matched short/normal/technical samples, reference offload, token pacing, and the
autocast variants produced the same speech-token sequences as FP32. Float PCM differed
at roughly 10^-6 amplitude or below, with difference SNR around 111–121 dB. These are
numerical equivalence checks, not subjective ratings. CPU vocoder output differed more
(roughly 38 dB difference SNR) because CPU execution and random excitation differ.
Native FP16 is reported separately wherever its token sequence differs.

During controlled single-sentence synthesis, FP32 averaged approximately 75 W (normal)
and 85 W (technical). FP16 variants averaged roughly 68–72 W but took longer. Paced
results varied with starting clocks/desktop activity: approximately 52–69 W normal and
54–64 W technical. Peaks still approached 120–128 W; the 16 ms repeated run sampled
one burst at 139 W despite the nominal 130 W limit. **A lower active-window wattage is
not evidence of lower energy per complete conversation.**

Both unmodified FP32 and 12 ms pacing stopped early in planned 120-second repeated
workloads. The paced run completed 34.24 s of audio at RTF 0.691 before a thermal stop;
FP32 completed 9.36 s at RTF 0.413. The native FP16 run also stopped at
85°C after five completed clips, at aggregate RTF 0.537. Unequal durations, abort tail
idle, and starting fan states prohibit ranking their average temperatures or energy.

The 16 ms run completed six clips / 20.08 s audio at aggregate RTF **0.826** before
the guard stopped it at an observed 87°C. It started at 51°C and fan 0%. Sensor
sampling cannot conclusively attribute each instantaneous peak to one component.
Transformer prefill and decoder stages still contain unpaced kernels; sleeping between
token forwards does not cap their peak power. No planned 120-second workload completed
without a thermal stop. There is no demonstrated steady-state temperature improvement
or reduction in joules per complete, matched conversation.

## Chunk size and quality checks

With 16 ms pacing, the identical long-response text was tested as a paragraph, four
sentences, and seven punctuation-delimited clauses:

| Segmentation | First waveform ready | Aggregate RTF | Largest simulated gap | Peak temperature |
|---|---:|---:|---:|---:|
| One paragraph | 13.06 s | 0.793 | Not applicable | 85°C |
| Sentences | 2.79 s | 0.817 | 1.565 s | 81°C |
| Clauses | 1.47 s | 0.846 | 0.313 s | 84°C |

These are single attempts, not thermal equilibrium measurements. Paragraph and sentence
runs started at 51°C; clauses started at 55°C; fan was 0%. Text grouping changes prosody
and output duration (16.44, 15.40, and 16.48 seconds respectively). Their energy per
audio-second is not a fixed-output efficiency comparison. The long second sentence
exceeded the first sentence's playback budget even with aggregate RTF below 0.9.
A future pacing policy would need to consider queued audio duration, first-response
latency, and upcoming sentence length. No chunking change was integrated.

CPU `faster-whisper-medium.en` checks found **zero normalized word disagreements** on
all nine checked clips: FP32, native FP16 T3, and CPU vocoder, each with short, normal,
and technical sentences. No model was downloaded. This is an ASR proxy, not a human
pronunciation assessment. The existing Chatterbox speaker encoder (CPU) gave normal
sentence cosine similarity 0.8536 / 0.8536 / 0.8534 and technical 0.9157 / 0.9178 /
0.9156 respectively. It is also the model's own conditioning encoder, so these scores
are uncalibrated and not independent evidence of perceptual voice quality. The two
FP16 technical sequences differ despite correct words. All comparison WAVs are retained
for listening; no subjective listening result is asserted by this report.

Paced/offloaded clips were checked directly against FP32 tokens and PCM, avoiding
redundant ASR runs on essentially identical waveforms. The 16 ms samples retained
identical tokens with PCM difference SNR 117.6–120.8 dB.

## Real playback validation

The first `runtime-pace16` run returned first-playback callbacks at 1.63 s (short)
and 2.49 s (normal), and approximately 29 ms between successive completed playback
calls. However, it took 30.65 s for a 12.36-second multi-sentence waveform sequence.
This output-stream timing anomaly makes its apparent continuity and duty-cycle results
unreliable. It is retained, not used as a successful conversation benchmark.

Subsequent GPU-free checks played one-second silent buffers in 1.05 seconds at both
24 kHz and 48 kHz. Replaying saved WAVs through `TextToSpeech.speak` took the expected
duration plus initialization/drain overhead. The cause of the anomaly is unknown.

The `runtime-pace16-verified` repeat logged actual stream sample rate, latency, and
active duration. First playback handoff was **1.78 s for “Of course”** and **2.36 s for
the normal sentence**. The first three Turbo sentences had RTF 0.886 / 0.883 / 0.857;
the first gap was **241 ms**, followed by **60 ms**. The benchmark guard then reported
**88°C during the fourth synthesis**, and RAPHAEL's existing error handling used Piper
for the last sentence. This is a failed sustained Turbo test, not a successful quiet
conversation. The last `multi-3.wav` is Piper at 22050 Hz and is labeled as fallback
on the page. The third playback interval again outlasted its PCM duration (5.28 versus
3.92 seconds), so small inter-call gaps alone are insufficient evidence of continuity.

Forced worker shutdown after fallback prevented that repeat's final telemetry dump.
Its thermal error and per-sentence/device metrics are preserved in the log and
`runtime.json`; aggregate power for that run is unavailable. The benchmark worker now
saves telemetry before raising a thermal error, without rerunning the hot workload.
All thermal guards are **benchmark-only**, not new production behavior. Playback
callbacks measure software handoff, not acoustic onset; no hardware loopback was used.
The shim waits for cooldown inside reference preparation, so its reported cold
conditioning/ready time includes that artificial wait. Use the isolated load and
conditioning figures (approximately 8.4 s and 0.95 s in `screen-1`), not the shim's
36.9-second ready time, when discussing normal startup.

## Recommendation

- **Normal mode:** retain selected FP32 Turbo, the primary reference, cached
  conditioning, and the sentence pipeline. There is no proven superior replacement in
  these tests. CPU reference-encoder offload is a viable future memory optimization,
  not a demonstrated cooling fix.
- **Quiet mode:** none is validated for sustained use with the current cooling
  behavior. 12–16 ms token pacing preserves the waveform and reaches the requested
  throughput range, but still permits high-power bursts and thermal stops. FP16 and
  CPU vocoder variants added latency without establishing an energy/thermal benefit.
- Slowing generation can lower active-window power while consuming similar or greater
  total energy. The production playback queue already creates idle time; replacing
  that idle time with longer, paced generation is not automatically an efficiency gain.
  Investigate a temporary hardware cap and the cooling condition before adopting a
  quiet mode. No experimental setting should be enabled on the basis of these runs.

## System clock and power experiments

Read-only inspection found a 130 W configured/default limit, minimum 70 W, maximum
140 W. Actual sampled bursts above 110 W mean a temporary 70 W limit could be useful,
even though the user's earlier instantaneous reading was ~55 W. This remains an
**untested hypothesis**, as does a conservative 210–1200 MHz graphics-clock range.

The system requires a sudo password (`sudo -n true` failed). No power or clock command
was executed; no system setting needs restoration. NVIDIA documents that
[power limits and clock locks require root](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

For a later administrator-assisted experiment, first record the current power limit
and any existing clock restriction. Test **one** setting at a time. Use a root-owned
supervisor with `try/finally`/signal restoration and a fixed timeout; run the Python
benchmark as the ordinary user. A power-only test would set 70 W and restore the recorded
limit (130 W in this audit). A clock test must restore the prior clock restriction;
`--reset-gpu-clocks` is appropriate only if clocks were originally unrestricted. Do not
add a boot service, persistent undervolt, fan override, or run inference as root.

## Reproduction

Examples (all artifacts remain under ignored `data/voice`):

```bash
data/voice/envs/clone/bin/python scripts/benchmark_turbo_efficiency.py \
  --output data/voice/benchmarks/turbo-efficiency/new-screen \
  --modes baseline pace12 offload --cases short normal technical \
  --cool-each-case --start-ceiling 57 --fan-ceiling 0

data/voice/envs/clone/bin/python scripts/benchmark_turbo_efficiency.py \
  --output data/voice/benchmarks/turbo-efficiency/new-repeat \
  --modes pace16 --cases normal --repeat-seconds 120 --chunk-study \
  --start-ceiling 57 --fan-ceiling 0

.venv/bin/python scripts/benchmark_turbo_efficiency_runtime.py \
  --mode pace16 --output data/voice/benchmarks/turbo-efficiency/new-runtime

.venv/bin/python scripts/build_turbo_efficiency_report.py \
  data/voice/benchmarks/turbo-efficiency

.venv/bin/pytest -q tests/test_turbo_efficiency_benchmark.py
```

Do not run GPU benchmark processes concurrently or while RAPHAEL already has Turbo
resident. No benchmark downloads models. All control patches affect only the isolated
model instance and are restored on exit. The runtime shim is never selected by production.

Validation: Ruff passed across `src`, `tests`, and `scripts`; 14 focused benchmark/
report tests passed; changed scripts compiled; all 64 listening controls and local
links resolved. Thermal-stop persistence is regression-tested without another hot run.
The existing production files and `.env` were not edited.
