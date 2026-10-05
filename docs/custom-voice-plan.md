# Custom voice: candidate research and experiment plan

Researched 2026-10-05 against official repositories, code, and model cards.
Target: English adult female around 20–30, warm, calm, caring, confident,
slightly lower/softer register, faithful to the actual reference speaker.
First source: <https://www.youtube.com/watch?v=PYmd20HsBj4>, recorded in the ignored
`data/voice/source_urls.txt`. Browser extraction of this video's contents failed;
its duration, cleanliness, and speaker turns have not yet been verified.

## Recommendation and evidence limits

Benchmark **Chatterbox Turbo first**, **Fun-CosyVoice3-0.5B-2512 second**, then
**Qwen3-TTS-12Hz-0.6B-Base** if they do not settle the quality/latency tradeoff.
Use **Chatterbox Nano** as a smaller companion comparison if Turbo is too slow
or consumes too much memory. **F5TTS_v1_Base** is a quality challenger if those
results disappoint. This is an experiment order, not a declaration of a winner.
Do not download every candidate in advance.

Published demo quality and benchmark numbers do not predict this speaker on a
GTX 1660 SUPER. No target-speaker similarity or subjective naturalness scores
are available yet. Actual peak VRAM/RAM, cold loads, warm TTFA, speed, stability,
and consistency on this machine are **unmeasured for all new candidates**.
The baseline report covers only already-installed Piper voices.

| Candidate / model | Native custom voice and quality controls | Audio delivery in inspected official implementation | Assistant integration / warm operation |
| --- | --- | --- | --- |
| Chatterbox Turbo (350M backbone) | Zero-shot reference cloning, native paralinguistic tags; listen for calm everyday prosody. Turbo ignores original Chatterbox CFG/exaggeration controls. | `generate()` returns a complete waveform; no streaming generator in this class. Sentence delivery initially. | Moderate complexity. Prepare conditionals once and call without `audio_prompt_path` thereafter. |
| Chatterbox Nano (110M backbone) | Same reference mechanism; smaller capacity may lose similarity/naturalness. | Same complete-waveform API. | Same adapter as Turbo; useful CPU/resource comparison. |
| Fun-CosyVoice3-0.5B-2512 | Zero-shot cloning, instruction and fine-grained controls. | `stream=True` yields speech chunks; test actual chunk cadence on this GPU. | More dependencies/components. Cache `zero_shot_spk_id`; keep one LLM/flow/vocoder stack warm. |
| Qwen3-TTS-12Hz-0.6B-Base | Audio + exact text cloning; cache the full voice-clone prompt. Embedding-only mode can reduce quality. | Architecture advertises streaming; inspected Python `generate_voice_clone()` returns full audio. Its `non_streaming_mode=False` flag alone does not enable live PCM output. | Moderate for batch/sentence output, higher for actual streaming. Use Base for cloning, not predefined CustomVoice speakers. |
| F5TTS_v1_Base (~0.3B class) | Native reference audio/text conditioning; multi-style references. | Chunk inference/socket service; diffusion completes a synthesis chunk before delivery. | Moderate; keep model/vocoder loaded, supply verified reference text to avoid loading reference ASR. |
| Fish OpenAudio S1 Mini / current `fishaudio/s1-mini` (0.5B) | Reference cloning and emotion/tone markers. Existing RAPHAEL launcher points at this older family. | Family supports server delivery; old RAPHAEL client explicitly disables streaming. | Higher recovery/version risk: absent external checkout and no pinned historical revision. |
| Piper Amy / installed RAPHAEL | Amy is a fixed speaker; custom identity requires a trained ONNX voice. Limited expressive control. | Native chunks, currently buffered by RAPHAEL. | Already stable CPU integration; retain as fallback and latency baseline. |

Turbo/Nano sizes, API, and lack of Turbo exaggeration/CFG are confirmed in
[Resemble's model overview](https://github.com/resemble-ai/chatterbox) and
[Turbo implementation](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/tts_turbo.py).
The Nano CPU speed claim is a vendor result, not an i5-10400F measurement.
Reference prompts must be longer than five seconds in the inspected code;
roughly 10 clean seconds is a sensible initial reference.

CosyVoice3 is a current 0.5B release with English support and prosody controls:
[official repository](https://github.com/FunAudioLLM/CosyVoice).
Caching and streamed audio are exposed in its
[Python implementation](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py).
The code warns that generation text much shorter than reference text may perform
poorly; the two-word test is therefore essential. Do not pad user-facing answers
with unwanted words to hide a model's short-response weakness.

Qwen's Base model and reusable prompts are documented in the
[official repository](https://github.com/QwenLM/Qwen3-TTS).
Check the actual [Python wrapper](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
before interpreting advertised latency: it generates codes and then decodes audio.
Its streaming-input simulation flag is not a streaming-output API.
The advertised 97 ms result is not a GTX 1660 SUPER estimate.

F5's [inference guide](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md)
covers reference conditioning, chunk delivery, socket serving, and early-checkpoint
EMA behavior. Its [training guide](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/README.md)
provides a fine-tuning path. A vocoder is part of native waveform synthesis,
not a voice-conversion model after TTS.

Fish's current [S1 Mini card](https://huggingface.co/fishaudio/s1-mini) names the
0.5B distilled family and expressive markers; the historical OpenAudio model URL
redirects there. Access is gated and weights have CC-BY-NC-SA-4.0 terms. Do not
silently substitute the latest Fish repository for the old launcher's version.
[Fish Speech 1.5.1 inference docs](https://github.com/fishaudio/fish-speech/blob/v1.5.1/docs/en/inference.md)
describe an older, different checkpoint/codec family and streaming API.

## Hardware, memory, training feasibility, and exclusions

Backbone size is **not** total download size or peak memory: speaker encoders,
audio codecs/vocoders, contexts, CUDA libraries, and activations also count.
The FP16 numbers below are arithmetic backbone-weight floors, not fit guarantees.
Pin exact source commits and model revisions and inventory actual files during
candidate setup before downloading weights.

| Candidate | FP16 backbone floor | VRAM / RAM evidence | Fine-tuning support and 6 GB assessment |
| --- | ---: | --- | --- |
| Turbo | ~0.70 GB | Whole-stack peak unknown; default loader does not imply FP16. Measure its actual dtype and CPU alternative. | No supported end-to-end training recipe identified in inspected official overview; initially zero-shot only. |
| Nano | ~0.22 GB | Whole-stack peak unknown; shared non-backbone components remain. Best memory-first candidate. | Same training limitation as Turbo; zero-shot first. |
| CosyVoice3 | ~1.0 GB | Peak unknown; additional flow/vocoder and ONNX frontend. FP16 available, optional accelerators disabled initially. | Official training example exists, notes LLM training support. Full AdamW training is not a safe 6 GB assumption. |
| Qwen 0.6B Base | ~1.2 GB | Peak unknown; includes a separate tokenizer/codec. FP16 with compatible attention must be measured. | Official single-speaker SFT exists, but current recipe requires adaptation for this GPU. Full SFT unlikely to fit; no stock 6 GB guarantee. |
| F5 v1 Base | ~0.6 GB | Peak unknown; vocoder and diffusion activations matter. Plausible inference candidate, not proven. | Official fine-tuning exists; batch-one cropped training may be viable, requires memory dry run. |
| S1 Mini | ~1.0 GB | Peak unknown here; codec and semantic caches matter. Absent installation complicates verification. | Family adaptation exists, but compatible S1-specific recipe/revision must be verified before proposing a run. |
| Piper | Local file ~64 MB | CPU path; measured process RSS in baseline report. No TTS CUDA allocation. | Existing Amy warm-start training script; 6 GB suitability requires a short measured experiment, never a default 500-epoch run. |

CosyVoice's [official training example](https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/libritts/cosyvoice3/run.sh)
is not a turnkey 6 GB speaker-adaptation prescription.
Qwen's [single-speaker trainer](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/sft_12hz.py)
hard-codes BF16 and FlashAttention 2 and uses full-model AdamW. It checkpoints by
epoch and lacks the short-run/resume controls required here. We must adapt and
verify it rather than blindly execute it. A typical full mixed-precision AdamW
state budget near 16 bytes per trained parameter would already be ~9.6 GB for
0.6B parameters before activations; LoRA saves optimizer state, not base weights.

The GTX 1660 SUPER is Turing. Use a tested CUDA build retaining its architecture,
FP16/FP32 as appropriate, and supported SDPA or eager attention. Check actual
kernel availability with a tiny inference before optimization. Upstream
[FlashAttention 2](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)
supports Ampere/Ada/Hopper, with a separate limited Turing implementation; its
BF16 GPU path requires Ampere or newer. Avoid using modern-GPU benchmark results
as expected performance on this GTX.

**VoxCPM considered, deferred:** the current official comparison reports
VoxCPM2 (2B) ~8 GB, VoxCPM1.5 (0.6B) ~6 GB, and VoxCPM-0.5B ~5 GB inference
VRAM. The two newer stock paths leave insufficient room for RAPHAEL's STT and
desktop; legacy 0.5B is borderline. The project supports streaming and native
SFT/LoRA, making it interesting if later memory profiling changes this conclusion.
These figures and RTX 4090 RTFs are publisher measurements, not ours.
[Official model comparison](https://github.com/OpenBMB/VoxCPM#-models--versions).
Its [VoxCPM1.5 model card](https://huggingface.co/openbmb/VoxCPM1.5) warns that
external denoising can distort references; keep denoising disabled.

**Latest Fish S2 Pro excluded from initial downloads:** official inference docs
recommend at least **24 GB GPU memory**. It is not an appropriate stock runtime
for a 6 GB always-on assistant. Cloud training would not make those inference
requirements disappear. [Official Fish inference requirements](https://github.com/fishaudio/fish-speech/blob/main/docs/en/inference.md).
XTTS v2 is not prioritized over the actively maintained candidates above; adding
another legacy baseline would increase installation work before listening value
is demonstrated. No real-time voice conversion is proposed.

All selected systems can use local model files for normal inference after setup.
Offline operation must be tested with networking disabled, including tokenizers,
text frontends, reference caches, and fallback behavior. Stability and RAM fit
will be assessed in repeated warm runs plus a small full-assistant session,
not inferred from successful isolated synthesis.

## Benchmark protocol before training

1. Extract only a handful of promising passages from the first source. VAD,
   transcript diagnostics, and signal checks shortlist candidates. Listen to
   approximately 20–50 short candidates at most, then verify **2–3 references
   around 6–12 seconds each** and their exact transcripts. Use a natural neutral
   passage first, with little music/reverb and one clear speaker. Similarity must
   follow the source's real voice; the desired style is not a promise to transform
   an unrelated timbre into a different person's voice.
2. Save source ID, URL, timestamp, reference SHA256, and processing recipe.
   Keep enrollment and held-out listening references separate from future training.
3. Install one candidate in an isolated compatible environment; record code/model
   revisions, dependencies, dtype, attention implementation, seed, and settings.
   Download only necessary inference artifacts. Run a tiny GPU compatibility and
   memory test before attempting the whole evaluation suite.
4. Reuse `docs/voice-evaluation.json` unchanged for every model. It includes all
   seven requested categories and a longer paragraph with time/numbers. Benchmark
   neutral voice first; compare caring/serious conditioning separately without
   changing spoken text. Keep native pace for quality comparison; run any speed
   adjustment as a separate experiment.
5. Use fresh processes for three cold-load trials (retain OS file cache and label
   that fact; don't flush system caches). Separate imports, model load, CUDA init,
   reference encoding, first inference, and optional compilation. Warm once, then
   measure three takes of every sentence. Retain slow/error takes, including retries.
6. Record request-to-first usable PCM, request-to-first audible non-silent region,
   playback startup separately, total generation, output seconds, and **RTF =
   generation time / output duration**. Batch APIs have TTFA equal to full output
   availability; do not count an internal token as playable audio.
7. For streaming record every chunk's arrival and duration, startup buffering,
   worst inter-chunk gap, underruns, sentence gaps, and cancellation delay. Include
   a timed LLM-token replay to compare chunk policies independently of provider
   network variation. Initial policy: complete sentences or substantial clauses,
   roughly 40–240 characters, preserve short complete replies, never split words.
8. Reset PyTorch peaks per trial; record peak allocated **and reserved** VRAM,
   plus sampled GPU process memory (includes non-PyTorch allocations). Sample RSS
   across the model worker/process tree. Record idle GPU usage and sampling interval;
   short peaks can escape sampling. CPU Piper's GPU peak remains unmeasured/N/A.
9. Repeat the finalists with resident `small.en` STT and CPU wake spotting, then
   test the optional STT retry under the intended GPU budget. Target initial
   headroom of at least 0.5–1 GB after resident workloads; tune from evidence.
   Never unload/reload TTS per sentence to manufacture a fit.
10. Export WAVs, machine JSON/CSV, and an offline HTML listening page. Randomize
    anonymous labels for the final comparison; rate naturalness, similarity,
    warmth, prosody, intelligibility, artifacts, and consistency (1–5 + comments).
    Speaker distance/ASR disagreement are diagnostics, not proof of audible quality.

Proposed benchmark targets, adjustable after listening: warm first usable audio
under ~0.5 s for short replies, sustained RTF below 1, no underruns or speaker drift.
Quality has priority; report any latency tradeoff explicitly. No candidate wins
until the user listens. Existing Piper baseline samples are already reproducible
without new dependencies; see [custom-voice-baseline.md](custom-voice-baseline.md).

## Proposed YouTube → clean dataset pipeline

The following is the next implementation design, **not an already working CLI**:

```text
python -m raphael voice-data prepare --urls data/voice/source_urls.txt --profile raphael
```

Keep this separate from the legacy `prepare-voice` command, which exports directly
to Piper format. The new canonical dataset is independent of the selected backend.

```text
data/voice/
  source_urls.txt
  sources/<video_id>/original.<ext>, source.info.json
  working/<video_id>/audio.flac
  state.sqlite3
  clips/<stable_clip_id>.wav
  manifests/accepted.jsonl, questionable.jsonl, rejected.jsonl
  review/index.html, decisions.jsonl
  references/raphael/normal/reference.wav, reference.txt, provenance.json
  exports/<model>/<dataset_version>/
  benchmarks/<run>/<candidate>/
  checkpoints/<experiment>/
```

**Download and resume.** Read one URL per line, support blanks/comments, reject
playlists by default, canonicalize YouTube video IDs, and deduplicate aliases.
Use yt-dlp `-f bestaudio/best --no-playlist --continue --write-info-json`, an
ID-based output template, and `--download-archive`. Download compressed originals
first without `-x`; convert separately with ffmpeg. Preserve submitted URL and
canonical webpage URL, ID, selected format/codec, source rate/channels/duration,
download tool version, and original hash. Archive entries are not proof that files
still exist: reconcile missing/corrupt originals against state before skipping.
Never archive a failed download. Preserve `.part` files for yt-dlp continuation.
[yt-dlp official options](https://github.com/yt-dlp/yt-dlp#usage-and-options).

Use explicit stage states (downloaded, decoded, segmented, transcribed, scored,
reviewed, exported), per-stage input hashes and tool/config fingerprints. A failed
stage cannot claim completion. Use one writer lock, temporary files plus atomic
rename, and a transaction for each finished artifact. After a crash, verify files
and hashes and redo incomplete work. Changed transcripts/settings invalidate only
dependent stages. Keep all originals; deletion is a separate deliberate operation.

**Decode and segment.** Verify decode with ffmpeg/ffprobe and preserve native-rate
lossless FLAC. Lossless conversion does not restore YouTube's lossy information.
Create analysis copies at 16 kHz for VAD/ASR/embeddings, never upsample everything
to pretend it has more detail. Inspect stereo channels before averaging because
phase cancellation or separate speakers can matter.

Use Silero speech regions, retain approximately 100–200 ms context padding, and
split at pauses or ASR word boundaries. Detect boundary uncertainty. Start with
an analysis window around 2–15 s, not a mandatory universal training limit;
exports enforce the chosen model's verified token/duration limits. Reference
prompts need their own length rules. Keep useful shorter phrases separately;
reject fragments and forced cuts, not every short sentence. Oversize continuous
speech gets aligned splits or review rather than arbitrary sample cuts.

**Speaker identity.** Enroll the target from manually confirmed references, not
from the largest cluster alone. Use ECAPA-TDNN embeddings on suitable voiced
windows, compare to multiple enrollment passages, and cluster by source/session.
Calibrate scores using a small labeled target/other-speaker sample: cosine scores
are not probabilities. Low match rejects; intermediate match queues review.
Missing speaker evidence queues review instead of silently accepting everything.
[SpeechBrain's official speaker model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
Embeddings cannot reliably detect overlap. On recordings with guests/interruptions,
use a dedicated overlap/diarization stage where compatible and affordable, or
require review of affected windows. Model access/dependencies must be assessed
before claiming automatic overlap detection exists. Cut out secondary speakers;
do not use source separation to disguise uncertain identity.

**Transcribe.** Use offline faster-whisper, isolated from normal listening,
with word timestamps, beam decoding, fixed language, no forced contextual
continuations, and retained segment log probability, no-speech probability,
compression ratio, and language diagnostics. Try the installed model first; for
uncertain clips consider a stronger model in a separate pass after unloading
other preparation GPU stages. New larger ASR weights are not automatically
downloaded. Punctuation can be repaired, words cannot be invented. Flag truncation,
repetitions, mismatch, and non-speech events. Verify transcripts of the tiny
training subset and enrollment references manually before any training.

**Signal/background scoring.** Detect decode failure, non-finite samples,
clipping/flat tops, extreme gain, VAD speech occupancy, long silence, and boundary
cuts. Estimate background level from non-speech regions, label its uncertainty
(breathing/music may defeat this proxy). Add a speech/music classifier only if
its actual implementation is validated; otherwise `music_score=null` and uncertain
sources require review. Do not treat RMS, VAD, or ASR log probability as a music
detector, clean-speech guarantee, or calibrated transcript accuracy.

**Grades and review.** Store component metrics and reason codes alongside a
versioned heuristic score. A: strongest identity, clear boundaries, clean signal,
good transcript evidence. B: usable with mild imperfections. C: uncertain identity,
background/overlap/boundary/transcript issue or missing evidence. D: definite
other speaker, corrupt audio, mostly silence, dominant music, severe distortion,
or unusable content. Hard failures override averages; a high ASR score cannot
cancel a wrong-speaker failure. Thresholds start as hypotheses calibrated on the
first reviewed batch, not universal truth.

Canonical clips remain in one place; manifests and accepted/questionable/rejected
folder views reference them. The offline HTML review page includes play controls,
transcript, reasons, confidence diagnostics, and original timestamp/link. Save
accept/reject/transcript corrections as durable decisions (an imported decisions
file is sufficient initially). Prioritize 20–50 borderline/high-value clips,
representative source clusters, and a small accepted audit sample. If uncertainty
is widespread, quarantine the source rather than pretend 50 reviews certify it.
The user can stop reviewing and retain a smaller confidently clean dataset.

**Export.** Preserve natural clean audio; no blanket denoising. Apply only required
resampling, mono selection, and model-specific gain handling, with recorded gain
and clipping checks. Near-duplicate/overlapping segments must share the same split.
Hold out entire videos or sessions for validation; reserve fixed listening texts
and genuine speaker references. Export selected A/B clips through a model adapter
(Piper metadata, Qwen JSONL/codec preparation, F5 format, etc.), not by feeding
one format to every trainer.

Each canonical manifest row includes stable ID/path/hash, exact transcript,
submitted/canonical URL, video ID, original start/end, duration/rate/channels,
speaker model/revision and similarity diagnostics, ASR diagnostics, noise/music/
overlap metrics (nullable), grade/score/reasons, preprocessing version, split,
and manual decision. Report raw unique duration, selected duration, grade counts,
and average duration; never count duplicated clips twice.

## Training gates, cloud, and final integration

First decide if verified reference conditioning already meets quality/similarity.
If it does, stop at a cached native voice profile and skip training.

If a clear defect remains and the winning model has a verified adaptation path,
prepare approximately 5–10 minutes of excellent audio plus session-held-out data.
Run a memory/dataset smoke check, then a deliberately short validation experiment
with early/frequent checkpoints and fixed sentences. A tentative 100–300 optimizer
steps is a validation budget, not a model-independent command or convergence
claim. Choose rate, precision, crop length, batch/accumulation, and adapters from
that model's verified recipe; record finite losses, gradient behavior, elapsed
time, and actual memory. Reload checkpoints in a fresh inference process and
compare base, early, and later checkpoints using identical conditioning/seeds.

No audible improvement means **stop and diagnose** transcript/identity quality,
conditioning, learning rate, changed parameters, model/codec compatibility,
checkpoint load, EMA selection, and inference settings. Training loss alone is
not success. Look for improved held-out similarity/consistency without artifacts,
memorized phrases, pronunciation regression, or exaggerated prosody. Choose the
best listening checkpoint, not the newest checkpoint.

Before any 7–8 hour run, present the selected pinned model, actual benchmark and
filtering results, all hyperparameters, measured/projected memory and duration,
save/evaluation cadence, exact start/resume/stop commands, and overfitting/early
stop criteria. **Wait for explicit approval.** Existing auto-install training
scripts are not part of this gated workflow.

Cloud is deferred until a validation run establishes a reason to train. A 16–24 GB
GPU may solve adaptation-memory limits while the same smaller base/checkpoint
still runs locally. Obtain a current provider quote including storage/billing
increments, and calculate total cost from measured throughput plus setup time;
the $3–4 budget is a ceiling, not an assumed sufficient amount. No paid instance
will be created without approval. A larger model that cannot infer locally does
not qualify just because training is affordable.

Final runtime: one persistent TTS model, one cached normal profile initially,
safe text chunking, bounded PCM queues, sounddevice output streaming when supported,
generation-aware cancellation, and local Amy fallback. If a separate Python worker
is required for dependency compatibility, start it once and retain it. No process,
CUDA initialization, checkpoint load, or reference encode per sentence. Later
profiles (`caring`, `serious`, `sleepy`) can reuse the model with separate native
conditioning. No second real-time voice-conversion stage.

## Current scope boundary

Completed: repository inspection, candidate research, dataset design, fixed
evaluation suite, and an offline installed-Piper baseline harness.
Pending: source download/clean-reference review, new-model environments/benchmarks,
human listening decisions, training decision, and winning-voice integration.
No long or short training, paid compute, model downloads, or runtime replacement
has occurred in this phase.
