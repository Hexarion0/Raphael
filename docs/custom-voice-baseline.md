# Installed Piper baseline — 2026-10-05

These are measured local **CPU synthesis** results for existing voices, not a
new-model cloning benchmark or subjective listening verdict. No reference audio,
model downloads, training, playback, or `.env` changes were involved.
The future voice-cloning candidates have not been installed or tested.

Ran one fresh process per configuration, native Piper alignment enabled, one
first-inference utterance, then three warm takes of each of the eight fixed
evaluation texts. There are **75 generated WAV files**, retained privately under
`data/voice/benchmarks/2026-10-05/`. Each run includes `summary.json`, incremental
`measurements.json`, blank `listening.csv`, and an offline HTML listening page.

Machine/environment: i5-10400F target desktop, Linux/Arch, Python 3.14.7,
Piper 1.8.0, ONNX Runtime 1.30.0, NumPy 2.5.3, soundfile 0.14.0.
The harness uses the existing RAPHAEL `TextToSpeech.synthesize()` path and verifies
nonempty, finite, mono output. No audio device is opened.

| Configuration | Cold model load (s) | First inference, Of course (s) | Warm median across all takes (s) | Warm median RTF | Process peak RSS (MiB) | Synthesis errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAPHAEL, speed 1.0 | 1.471 | 0.088 | 0.160 | 0.047 | 479.0 | 0 |
| Amy, speed 1.0 | 1.232 | 0.065 | 0.210 | 0.054 | 476.5 | 0 |
| RAPHAEL, configured speed 0.8 | 1.232 | 0.063 | 0.175 | 0.042 | 495.8 | 0 |

Cold model load excludes dependency imports and Python interpreter startup;
OS file cache was retained. Only one cold observation per configuration was run
in this initial phase. This is not a statistical cold-start comparison.
RSS is the Linux process high-water mark including dependencies and synthesis
buffers. Primary STT and the full listening loop were not loaded by this harness.
Piper explicitly uses CPU and allocates no TTS CUDA context; peak whole-GPU memory
was **not measured**. Naturalness, similarity, and consistency are **unrated**.

Median warm time until RAPHAEL receives the whole input waveform:

| Evaluation category | RAPHAEL 1.0 (s) | Amy 1.0 (s) | RAPHAEL 0.8 (s) |
| --- | ---: | ---: | ---: |
| Very short | 0.043 | 0.055 | 0.040 |
| Short conversational | 0.177 | 0.194 | 0.175 |
| Normal | 0.186 | 0.216 | 0.224 |
| Caring/soft | 0.137 | 0.210 | 0.128 |
| Serious | 0.156 | 0.193 | 0.175 |
| Question | 0.136 | 0.146 | 0.149 |
| Numbers/technical | 0.210 | 0.313 | 0.246 |
| Long paragraph | 1.006 | 1.129 | 1.072 |

The long-paragraph test demonstrates the buffering distinction: RAPHAEL 1.0
produced its first native Piper chunk around **0.110 s**, but the wrapper did not
return playable audio until approximately **1.006 s**. In normal sentence-streamed
operation, this paragraph is divided earlier by `SentenceBuffer`; the paragraph
measurement is not the full assistant's current end-to-end TTFA.

These are generation/availability timings, not first audible speaker output.
They exclude LLM token arrival, sentence buffering, leading waveform silence,
audio-device startup, and physical speaker latency. RTF divides each generation
time by that take's actual waveform length. Stochastic durations vary between
takes; lower RTF at speed 0.8 does not prove faster inference because slowed
output enlarges the denominator. Speed adjustment uses duration scaling, not
a deliberate pitch transformation.

Listen locally:

- [RAPHAEL at speed 1.0](../data/voice/benchmarks/2026-10-05/piper-raphael-speed1/index.html)
- [Amy at speed 1.0](../data/voice/benchmarks/2026-10-05/piper-amy-speed1/index.html)
- [RAPHAEL at current speed 0.8](../data/voice/benchmarks/2026-10-05/piper-raphael-speed08/index.html)

These pages/audio files are intentionally ignored by git. The app may show local
HTML as a file; opening it in a browser exposes built-in audio controls. They
establish a listening baseline; the target speaker has not yet been enrolled.

Reproduce from the repository root using a **new output directory** each time:

```bash
.venv/bin/python scripts/benchmark_piper_voice.py \
  --voice en_US-raphael-medium --alignments \
  --output data/voice/benchmarks/local-check/piper-raphael-speed1

.venv/bin/python scripts/benchmark_piper_voice.py \
  --voice en_US-amy-medium --alignments \
  --output data/voice/benchmarks/local-check/piper-amy-speed1

.venv/bin/python scripts/benchmark_piper_voice.py \
  --voice en_US-raphael-medium --speed 0.8 --alignments \
  --output data/voice/benchmarks/local-check/piper-raphael-speed08
```

The harness refuses missing model/config pairs and existing output directories.
It passes a verified existing ONNX path to Piper, avoiding automatic voice downloads.
Model/config and sentence-suite SHA256s are recorded per run. Every result is
persisted after generation so an interrupted run retains completed observations.
No new candidate adapters are implemented here: their shared measurement protocol
and rollout order are in [custom-voice-plan.md](custom-voice-plan.md).

Validation: focused TTS/streaming/dataset/benchmark checks passed (59 tests,
one integration excluded); full suite passed (569 tests, three integrations
excluded). Ruff and whitespace checks passed. Active RAPHAEL and Amy model
checksums and effective runtime configuration remained unchanged. The two added
regressions verify that missing models stop before output creation and earlier
benchmark results cannot be overwritten.
