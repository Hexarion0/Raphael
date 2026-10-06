# Source audio and reference preparation

This phase extracts references for native zero-shot cloning. It creates no training
dataset and changes no installed RAPHAEL voice. Everything generated lives under ignored
`data/voice/`. Originals, metadata, timestamps and hashes stay available.

Put individual HTTPS YouTube URLs in `data/voice/source_urls.txt`, one per line. Blank lines
and `#` comments are allowed. Video-ID aliases are deduplicated; unrelated hosts and
playlist-only URLs are rejected. English original audio is preferred over auto-dubbed tracks.

On the prepared machine, run from the repository root:

```bash
.venv/bin/python scripts/prepare_voice_reference_bundle.py
```

The bundle uses existing local ASR snapshots and the isolated analysis environment.
`--asr-model`, `--verification-model`, `--analysis-python`, `--urls`, `--root` and `--yt-dlp`
override locations. It validates installed assets first; it does not install packages,
fetch ASR weights, download TTS checkpoints, fine-tune, or start paid jobs.

The lower-level downloader/preparer remains available:

```bash
.venv/bin/python scripts/prepare_voice_references.py \
  --yt-dlp data/voice/vendor/yt-dlp \
  --asr-model /absolute/path/to/existing/faster-whisper/snapshot
```

Use `--download-only` to preserve source audio without ASR. An interrupted yt-dlp download
resumes using its partial file/archive. Completed sources are verified by SHA-256 before
reuse. Changed originals are preserved and rejected for investigation. FLAC decodes are
written to staging files, validated and renamed; native source rate/channels are retained.
The working decode is lossless relative to the downloaded compressed audio, not an upgrade
to YouTube's original recording quality. No denoising, pitch shifting, voice conversion or
automatic gain processing is applied.

Cached local Whisper ASR provides speech/word boundaries and transcription diagnostics.
Sentence-aware grouping finds 10–30 second references with at most four seconds between
utterances. Original timestamps include small boundary padding. Signal checks penalize
silence, clipping and weak/hallucination-like recognition. A bounded shortlist avoids
listening through the full video. PANNs Cnn6 sound tags flag music and atypical delivery;
Chatterbox speaker embeddings screen changes relative to the dominant source speaker.
Two cached ASR models cross-check selected words. Their agreement is not human certification.

Inspect `data/voice/references/raphael/index.html`. Each reference has a WAV, transcript TXT,
JSON with original URL/video ID/start/end, source and waveform hashes, ASR diagnostics,
sound tags, speaker consistency and a selection reason. `ranked_candidates.json` retains
the ranked shortlist. `accepted/manifest.json` stays empty until human acceptance;
`questionable/manifest.json` contains provisional clips; `rejected/manifest.json` records
hard signal rejects. `sources/<id>/all_candidates.json` preserves the wider candidate pool.
These are overlapping reference alternatives, so summed durations are not dataset duration.

Automatic scores are diagnostic, not calibrated probabilities. Silence-floor measurements
are noise proxies, not reliable SNR. Speaker consistency does not establish identity or
exclude every short interruption. Overlap confidence remains null because a reliable
overlap detector has not been run. All selected references therefore require a short
identity/delivery/overlap check before a final voice or training dataset is approved.

The current source is a soft spoken ASMR roleplay. The primary reference was chosen for
continuous conversational delivery, strong transcription, low detected music and consistent
speaker embeddings. Two alternatives contain more pauses. Further naturally spoken sources
may improve everyday-assistant delivery more effectively than training on additional ASMR.

Preparation dependencies stay separate from RAPHAEL's Python 3.14 environment. Analysis
uses Python 3.12, CPU PyTorch/Chatterbox's encoder, librosa and the official
[PANNs Cnn6 checkpoint](https://zenodo.org/records/3987831). ASR uses the already installed
faster-whisper/CTranslate2 with `int8_float16` on CUDA. Audio is decoded through ffmpeg;
feeding NumPy arrays avoids the installed PyAV/faster-whisper decode API incompatibility.
GPU access requires running outside Codex's filesystem/network sandbox on this machine.
