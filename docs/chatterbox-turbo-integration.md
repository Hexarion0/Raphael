# Chatterbox Turbo RAPHAEL voice

RAPHAEL can use the pinned local Chatterbox Turbo checkpoint with the selected primary
reference. The production request path is:

```text
LLM stream -> visible text filter -> sentence buffer -> bounded synthesis queue
           -> persistent Chatterbox Turbo worker -> ordered audio playback
```

The model runs in a persistent Python 3.12 child process so it does not conflict with
RAPHAEL's Python 3.14 dependencies. It loads once in the background after STT is ready,
keeps the primary reference conditioning warm, and runs offline. A free-VRAM guard skips
Turbo startup below `TTS_MIN_FREE_VRAM_MB`. A synthesis or initialization failure is logged
and latched to the local Piper RAPHAEL voice; Amy is tried if that fallback model is absent.
The existing Piper backend remains selectable with `TTS_ENGINE=piper` and its selected
voice remains `TTS_VOICE`.

The voice manifest is [voice_profiles/raphael/voice.json](../voice_profiles/raphael/voice.json).
It names the exact reference WAV/transcript and records the pinned model revision
`749d1c1a46eb10492095d68fbcf55691ccf137cd`. Model files, audio, generated samples, and
benchmarks are kept under ignored `data/` paths. The inference environment is pinned in
`scripts/voice_requirements/clone.txt`.

Turbo's nine verified event markers are optional and only valid at the beginning of a
spoken sentence: `[clear throat]`, `[sigh]`, `[shush]`, `[cough]`, `[groan]`, `[sniff]`,
`[gasp]`, `[chuckle]`, and `[laugh]`. With Turbo selected, RAPHAEL's dialogue prompt says
to use one sparingly when a brief audible event adds meaning; normal replies have no marker.
The reusable `speech_event_instruction` and `add_speech_event` helpers keep dialogue policy
separate from backend parsing. These markers control vocal events, not sustained moods.

## Sentence queue and interruption

Turbo produces sentence waveforms rather than a live audio stream. RAPHAEL starts synthesis
as soon as a complete sentence is available, while a separate player emits earlier audio in
order. One sentence may wait in the text queue, and the audio queue defaults to two waveforms.
Backpressure limits memory use on long replies. A barge-in invalidates the active generation,
stops playback, and discards queued audio; it cannot be replayed by a later turn.

The benchmark uses a local fixed text stream through RAPHAEL's `stream_reply` implementation,
so it measures actual text filtering, sentence preparation, worker IPC, and local playback
without making an LLM/API request. Live LLM time before the first sentence is available will
add to the reported first-audio times. Device output must be available for playback timing.

## Reproduction

Run the full on-device TTS/STT and playback test with:

```bash
.venv/bin/python scripts/benchmark_raphael_turbo_runtime.py
```

It plays and saves the short response, a multi-sentence response, a longer reply, `[chuckle]`,
`[sigh]`, two rapid requests, an interrupted reply, the Piper fallback, a CUDA STT overlap,
and a post-restart sample. It writes a private `index.html` listening page, WAVs, and
`report.json` below `data/voice/benchmarks/raphael-turbo-runtime/`. Use `--no-playback` to
save samples without sending them to the audio device.

## GTX 1660 SUPER validation (2026-10-05)

Run on the RAPHAEL machine with its configured CUDA `small.en` STT model, `int8_float16`,
the pinned Turbo worker, the primary reference, and the default audio output. The benchmark
used a local provider fixture (no paid or remote LLM request):

- Turbo cold ready after STT was resident: **13.13 s** (model load **7.69 s**, reference
  conditioning **0.68 s**). A new process after shutdown took **13.03 s** to become ready.
- Short “Of course.” warm text-to-first-playback: **1.00 s**. Other warm first-audio times
  were **0.73–1.56 s**, depending on sentence length and GPU load.
- The two-sentence reply had one measured inter-sentence gap of **70 ms**. The four-sentence
  longer reply had gaps of **71–74 ms**. The synthesis queue ran ahead while playback was
  active; maximum observed audio queue depth was **2**. Sentence real-time factors ranged
  from **0.39–0.61** in this run.
- Total GPU memory peaked at **4,347 MiB of 6,144 MiB** with STT and Turbo active. The
  Chatterbox worker's peak resident RAM was **2,357 MiB**. No OOM or inference failure occurred.
- STT decoded the 31-word primary reference with **0 word errors**. STT latency was **0.56 s**
  alone and **1.32 s** during simultaneous Turbo synthesis/playback; overlap worked, with the
  expected GPU contention increasing STT latency.
- Two rapid requests completed in order. A barge-in stopped the active response in under
  **0.93 s** from request start and no second queued sentence started playing. A forced Turbo
  failure produced audio through local Piper fallback. Cold worker reload and synthesis both
  succeeded.

These are one on-device run with a local text-stream fixture; real LLM first-token delay and
environmental variation are not included. The listening page from this run is stored in the
ignored benchmark directory alongside the measurements.
