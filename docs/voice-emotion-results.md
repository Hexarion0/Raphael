# Chatterbox emotion listening experiment — 2026-10-05

The requested samples and measurements are complete. **No training, paid compute, voice
conversion, or RAPHAEL runtime changes were made.** Piper and the existing voices remain intact.

Open the local listening page:

`data/voice/benchmarks/emotion-controls/index.html`

It contains 180 new listening takes and links to all 25 previous Qwen baseline takes, plus
the four source references. One additional original-model smoke-test WAV is retained outside
the comparison. Nothing was selected after listening to make a model look better.

Start with the main comparison, then use the tabs for original-model levels, Turbo references,
Turbo tags, punctuation, and temperature. Each player shows its actual input, reference,
generation settings, timing, and available word/speaker checks. Playback stops other players.
Listening notes persist locally and can be exported. Pages and WAVs work offline.

## What looks most promising

**Turbo with the primary reference and documented event tags is the strongest technical
starting point.** The user already rated its voice identity/quality approximately 10/10;
its latency and GPU headroom remain best here. Native sigh/laugh/chuckle/gasp comparisons
retained strong speaker proxies with no ASR word disagreement. This does **not** establish
Qwen-level emotion: adding an event is different from controlling the tone of every word.
Listen to the direct emotion rows before deciding whether the variation is enough.

**Original 500M default/subtle are the important alternatives to hear.** They have real
exaggeration and CFG controls and stayed close to the target in speaker proxies. Moderate
and strong samples are available, but stronger settings brought lower identity proxies and
more word/repetition flags. Increasing exaggeration did not reliably improve the desired
combination of voice identity and correctness.

Emotion quality, warmth, naturalness, and final voice similarity have not been assigned
subjective scores by this experiment. The user's listening decision remains the selection gate.
There is no evidence yet that fine-tuning is necessary for speaker similarity, given the
reported Turbo result. Whether more emotion requires another approach depends on these samples;
this benchmark does not justify training or promise that training would fix it.

## Verified implementation and settings

See [verified controls](voice-emotion-controls.md) for source links and the full distinction.
Installed Chatterbox 0.1.7 / official source commit
`5de7a54aa4e5e2baadb0182dde554908b48b85c2` matches current official control code.

- Turbo ignores `exaggeration`, `cfg_weight`, and `min_p`. Its emotion-conditioning flag is off.
- Original English 500M actively uses exaggeration and CFG; these are not named emotion selectors.
- Turbo documented events tested: `[sigh]`, `[laugh]`, `[chuckle]`, `[gasp]`.
- `[whisper]` and `[breath]` are absent and were not used.
- Native-token probes `[whispering]`, `[happy]`, `[sarcastic]` are separated and marked
  exploratory. They generated valid audio without literal-label ASR output, but their presence
  and this word check do not establish reliable style control.
- Turbo temperatures 0.6 and 1.0 were compared with the 0.8 default on three fixed texts.
  The other sampling parameters retained their defaults. Punctuation/internal capitals were
  compared separately. Neither mechanism guarantees a particular emotion.
- Qwen 0.6B **Base** uses cached reference cloning and emotional text content. Its tested
  cloning wrapper has no explicit emotion-instruction parameter; CustomVoice/VoiceDesign
  instruction features were not attributed to this Base model.

| Original profile | Exaggeration | CFG | Temperature | Warm requests |
| --- | ---: | ---: | ---: | ---: |
| Default | 0.50 | 0.50 | 0.80 | 30 |
| Subtle | 0.60 | 0.40 | 0.80 | 15 |
| Moderate | 0.75 | 0.30 | 0.80 | 15 |
| Strong | 1.00 | 0.30 | 0.80 | 15 |

These jointly change two parameters following official original-model guidance. They are
practical settings comparisons, not an isolated experiment on each parameter.

All original settings generated the same original eight evaluation sentences plus seven
new caring, concerned, serious, happy, excited, playful, and calm/sleepy scenarios. Turbo
primary/pleased-reference runs used the same 15 texts; playful/caring references used the
seven emotion texts. The main tag column inserts only documented events into those seven
texts; spoken words remain identical. The serious row intentionally has no inserted event.
Default original and new Qwen emotion scenarios have two warm takes. Matching take 1 uses
seed 43 throughout; additional take 2 uses 44. First-request samples use 42 and remain visible.

## GTX 1660 SUPER measurements

Measured on the actual GTX 1660 SUPER, i5-10400F, Arch Linux, Torch 2.6.0+cu124. Models ran
sequentially, with no other TTS/ASR model resident during timed generation. Chatterbox uses
stock FP32; Qwen uses BF16 with SDPA, emulated on this Turing GPU. Analysis ran afterward.

The following **warm medians use the same original eight sentences**, not the different
mixtures of new emotion/tag tests. There are 8 Turbo, 16 original-default, and 24 previous
Qwen warm measurements. Unequal repeats and generated durations limit precision.

| Model / profile | New-process load | Median audio ready | Median RTF | Peak process VRAM | Peak process RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turbo, primary reference, new run | 12.13 s | 1.44 s | 0.454 | 3416 MiB | 4860 MiB |
| Original default | 13.06 s | 4.11 s | 1.161 | 4726 MiB | 5033 MiB |
| Original subtle | 12.89 s | 4.16 s | 1.163 | 4064 MiB | 5033 MiB |
| Original moderate | 12.98 s | 4.18 s | 1.184 | 4058 MiB | 5044 MiB |
| Original strong | 12.93 s | 4.55 s | 1.169 | 4040 MiB | 4991 MiB |
| Qwen Base, previous baseline | 4.94 s | 8.26 s | 2.113 | 5406 MiB | 2480 MiB |

Process peaks cover the entire corresponding run, including loading and longer utterances.
The old Turbo baseline was 1.26 s / RTF 0.400 / 3792 MiB; the new result confirms the same
general performance class, rather than establishing an optimization. Qwen's new seven-text
emotion run loaded in 9.04 s, peaked at 4738 MiB VRAM / 2550 MiB RAM, and remained error-free.
Load timing starts at adapter imports/loading after CUDA initialization; filesystem cache
was not flushed. It is not complete RAPHAEL startup latency or a controlled disk-cold test.

For the **same seven new emotion scenarios**, warm medians were:

| Run | Audio ready | RTF |
| --- | ---: | ---: |
| Turbo primary | 1.95 s | 0.437 |
| Turbo pleased reference | 1.86 s | 0.399 |
| Turbo documented events | 2.36 s | 0.439 |
| Original default | 5.74 s | 1.084 |
| Original subtle | 5.88 s | 1.081 |
| Original moderate | 5.65 s | 1.080 |
| Original strong | 5.46 s | 1.084 |
| Qwen Base | 10.58 s | 2.081 |

Turbo is fastest and uses the least VRAM. Original expression settings are faster than
Qwen here, but considerably slower than Turbo. Raw output levels can vary; adjust playback
volume when assessing tone rather than treating louder audio as better expression.

### Long responses and streaming

| Run / take | Audio ready / total generation | Audio duration | PyTorch peak allocated / reserved |
| --- | ---: | ---: | ---: |
| Turbo primary | 7.65 s | 18.56 s | Recorded in measurements.json |
| Original default 1 | 25.25 s | 25.92 s | 3841 / 4608 MiB |
| Original default 2 | 20.55 s | 21.24 s | 3667 / 4608 MiB |
| Original subtle | 16.71 s | 17.60 s | 3549 / 3948 MiB |
| Original moderate | 16.70 s | 17.48 s | 3548 / 3942 MiB |
| Original strong | 15.77 s | 16.24 s | 3510 / 3924 MiB |
| Qwen previous baseline, three takes | 37.7–48.6 s | Approximately 18–24 s | Recorded in previous benchmark |

Every tested stock Chatterbox/Qwen API returned a complete waveform. First usable CPU audio
therefore arrives near total synthesis completion; none of these runs streamed playable
chunks. Playback/device latency was not measured. Longer responses need future text-chunking
work if the chosen model is integrated; that optimization was not performed in this phase.

There were **no synthesis exceptions, non-finite output failures, or OOMs** in the completed
180 new listening requests and smoke test. This does not mean error-free speech. Original
default's longest output contains an ASR repetition flag. Whole-card peaks were 4253 MiB
for the new Turbo primary run, 5497 MiB for original default, and 6110 MiB for the previous
Qwen long-response benchmark, out of 6144 MiB physical VRAM. Actual GPU-sharing behavior
with RAPHAEL's STT/other workloads remains untested.

## Quality flags and identity diagnostics

ASR is an automated disagreement check, not proof of a pronunciation error. It can misread
soft speech, contractions, numbers, and nonverbal events. Every flagged take remains accessible.

- Original default long take 1: ASR repeats “For now, take your time” and misses the final
  “Everything is under control.” Both long takes have uncertain time pronunciation.
- Original strong: repeated “Of course” in both short takes; repeated “I'm here with you”
  and changed wording in comforting; unstable time pronunciation. This setting is a useful
  boundary test, not an automatic choice for normal assistant speech.
- Original moderate: flags on the very short response, time, and the serious sentence.
  Subtle's flags were limited to the long response.
- Turbo alternate pleased reference: the opening of “I've checked everything” is flagged,
  and speaker proxies declined. Its playful “All right” / “Alright” disagreement is partly
  a tokenization artifact, so the total edit count should not be treated as a human error rate.
- Turbo primary: the GPU sentence has two small ASR word disagreements. Documented tag
  comparisons, separate tag probes, punctuation comparisons, and playful-reference samples
  had no word disagreements. That does not certify natural emotion.
- Temperature 0.6: missing “Hey” flagged in comforting. Temperature 1.0: “All right” flagged
  in playful. Raising temperature is not a reliably better-emotion strategy.
- New Qwen emotion set: one disagreement, “There you go” versus “There we go,” in sleepy take 1.
  The user's previously heard pronunciation issues still matter despite low automated counts.

Mean speaker cosines below use **matching first warm takes of the seven emotion texts**, each
at least three seconds. Both encoders are uncalibrated and domain-sensitive; Chatterbox's
encoder also conditions the compared Chatterbox models. These are not similarity ratings.

| Run | Chatterbox encoder | Independent CAMPPlus proxy |
| --- | ---: | ---: |
| Turbo primary | 0.926 | 0.827 |
| Turbo documented events | 0.942 | 0.836 |
| Turbo pleased reference | 0.861 | 0.663 |
| Turbo playful reference | 0.881 | 0.713 |
| Turbo caring reference | 0.873 | 0.727 |
| Original default | 0.931 | 0.826 |
| Original subtle | 0.923 | 0.777 |
| Original moderate | 0.909 | 0.767 |
| Original strong | 0.853 | 0.713 |
| Qwen Base | 0.921 | 0.786 |

The drop across stronger original settings and alternate Turbo references is a reason to
listen carefully for identity changes, not proof that the speaker changed. Nonverbal events,
delivery, and short reference content can affect embeddings. Primary reference plus documented
tags is the most encouraging identity-preserving technical result; perceived emotional depth
still needs listening.

## References, provenance, and exact versions

The primary reference is unchanged: `reference-1.wav`, 505.36–517.62 s / 12.26 s, SHA256
`c23d11bc85d32115fb216e953d384d66cf72801096360b6c68c54e2c9557acdc`.
All references come from the permitted source `https://www.youtube.com/watch?v=PYmd20HsBj4`.

| Alternate reference | Source timestamps | Duration | Speaker cosine / minimum window | Music tag |
| --- | --- | ---: | ---: | ---: |
| Playful | 637.91–645.42 s | 7.51 s | 0.880 / 0.772 | 0.021 |
| Pleased | 780.42–786.35 s | 5.93 s | 0.925 / 0.891 | 0.026 |
| Caring | 359.20–370.62 s | 11.42 s | 0.855 / 0.731 | 0.023 |

WAVs, exact transcripts, source hashes, timestamps, reasons, sound tags, clipping/noise
diagnostics, and speaker windows are in `data/voice/references/raphael/emotion/`. All three
transcripts agree with cached medium.en ASR in addition to the source distil-large-v3 ASR.
No clipping was detected; all three have quiet/silence-ratio warnings. They remain provisional
delivery experiments, not automatically accepted training clips. Reference names describe
content; this ASMR source does not provide a verified wide range of happy/excited recordings.
No denoising, pitch manipulation, or external voice conversion was applied.

Pinned weight revisions:

- Original English: `ResembleAI/chatterbox` at
  `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`; 3,191,859,618 selected bytes.
- Turbo: `ResembleAI/chatterbox-turbo` at `749d1c1a46eb10492095d68fbcf55691ccf137cd`.
- Qwen: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` at `5d83992436eae1d760afd27aff78a71d676296fc`.

Original weights have streamed SHA256 verification in their ignored model directory.
Each run records model inventory, installed package versions, reference/suite hashes, seeds,
conditioning time, per-request allocator peaks, sampled process/whole-GPU peaks, and RAM.
Machine-readable aggregate files are `performance.json`, `quality-checks.json`,
`control-audit.json`, and `diagnostics.json` alongside the listening page. Per-run raw
measurements and WAVs remain available. Benchmark assets/model weights stay out of git.

## Reproduction and stop point

The pinned isolated environments and earlier source-download workflow are documented in
[the previous benchmark report](native-voice-cloning-results.md). With these assets already
present, the bounded experiment is:

```bash
data/voice/envs/clone/bin/python scripts/prepare_voice_emotion_references.py
.venv/bin/python scripts/verify_voice_transcripts.py \
  --references data/voice/references/raphael/emotion/references.json \
  --model /path/to/existing/local/faster-whisper-medium.en/snapshot
.venv/bin/python scripts/benchmark_voice_emotions.py chatterbox-turbo
.venv/bin/python scripts/benchmark_voice_emotions.py chatterbox500
.venv/bin/python scripts/benchmark_voice_emotions.py qwen06
.venv/bin/python scripts/build_voice_emotion_comparison.py
```

Run one synthesis backend at a time. `--plan-only` prints commands without inference.
Completed runs resume by skipping; an incomplete attempt is retained and deliberately requires
inspection rather than overwriting results. Ctrl+C stops the controller/current subprocess.
Every completed take is persisted incrementally. Each model/ref loads once per experiment run,
and conditioning is cached across its sentences. No per-sentence model reload or reference
re-encoding was introduced.

Optional word checks use `scripts/assess_cloning_audio.py` with an existing local ASR snapshot.
Speaker checks use `scripts/measure_voice_similarity.py`: the clone environment for the
Chatterbox encoder, the existing cosy environment for the CPU CAMPPlus/ONNX encoder. These
analysis models are not part of the final TTS runtime and run after timed synthesis.
Rebuild listening pages afterward to include the checks.

Validation: 584 standard repository tests passed (three hardware integration tests excluded),
Ruff passed for src/tests/scripts, and all local listening audio paths were checked.
An extra run accidentally removed the default integration exclusion and stalled on an audio
device test; it was interrupted, and the standard suite was rerun. This does not affect the
separately completed GTX synthesis measurements. The experiment stops here for the user's listening and
model/delivery choice. No further training or integration is authorized by these results.
