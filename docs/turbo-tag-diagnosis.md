# Turbo tag display diagnosis and corrected A/B — 2026-10-05

The tag generations did not fail. The listening page used a misleading fallback message for
comparisons that had not been scheduled. The fixed main page is:

`data/voice/benchmarks/emotion-controls/index.html`

The focused, documented-event A/B page is `turbo-tags.html` in the same directory.
No training, paid compute, reference replacement, model upgrade, or runtime change was made.

## 1. Exact cause of the previous message

`turbo-emotion-tags/measurements.json` contained seven completed warm outputs for IDs
09–15 plus a first-request sample. Every WAV existed and the run's `summary.json` had
`errors: []`. All eight outputs had completed ASR checks. The seven-text suite was intentional.

The main page joined the union of all candidates' sentence IDs: 01–15. Consequently, IDs
01–08 had no rows in that tag run. The renderer handled every empty cell with the same
message: “No completed warm sample. See recorded errors above.” It did not distinguish
an unscheduled comparison from a failed request. That message incorrectly implied failures.

The fix records planned sentence IDs/repeats in new benchmark summaries. The renderer now
distinguishes unscheduled, unknown, incomplete, and failed requests. The main tag column combines
the previously successful seven-text run with the newly generated eight-text run. It checks
that all 15 warm IDs are present and that their reference hashes agree. Existing audio and
measurements were preserved; successful previous requests were not rerun.

## 2. Supported tags and exact implementation

Installed Chatterbox is 0.1.7, with source matching pinned official commit
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`. The installed `tts_turbo.py` was inspected directly
under the isolated clone environment and matches the official source byte for byte, SHA256
`2f27e8ff2fa35181fdd2849367fb96ecc2cb87a1088ca7520c8bd51946599f08`.
Turbo weights remain `ResembleAI/chatterbox-turbo` revision
`749d1c1a46eb10492095d68fbcf55691ccf137cd`.

The [official Turbo demo's event list](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/gradio_tts_turbo_app.py)
contains these exact forms. Each was verified to encode as one native token in the installed
tokenizer, rather than a sequence spelling a bracketed word:

| Documented event | Native token ID | Generated in this experiment |
| --- | ---: | --- |
| `[clear throat]` | 50267 | Not tested |
| `[sigh]` | 50268 | Yes |
| `[shush]` | 50269 | Not tested |
| `[cough]` | 50270 | Not tested |
| `[groan]` | 50271 | Not tested |
| `[sniff]` | 50272 | Not tested |
| `[gasp]` | 50273 | Yes |
| `[chuckle]` | 50274 | Yes |
| `[laugh]` | 50275 | Yes |

Use lowercase brackets and the space in `[clear throat]`. `[whisper]` and `[breath]` are not
documented native events and were not generated. `[whispering]`, `[happy]`, and `[sarcastic]`
exist as reserved native tokens, but their presence does not establish reliable style control.
Previous exploratory samples remain in `turbo-tag-probes-archive.html`; they were removed
from the active documented-event comparison. No unsupported probes were regenerated.

The [Turbo implementation](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/tts_turbo.py)
still ignores exaggeration/CFG/min-p; this fix does not attribute original-model controls to it.

## 3. Successfully tested comparisons

`[sigh]`, `[laugh]`, `[chuckle]`, and `[gasp]` generated valid native audio in the previous
paired event and seven-emotion runs. Their previous outputs had no ASR word disagreements.

The eight original sentences now have new valid documented-event comparisons under
`turbo-tags-baseline/`: one first-request output and eight warm outputs. Every original spoken
word is unchanged; only a verified event prefix was added. These baseline additions use
sigh/chuckle/gasp; laugh remains tested in the happy emotion row and focused paired comparison.
The existing seven emotion comparisons retain their original insertion positions. The serious
emotion row remains an intentionally untagged control.

The primary WAV is identical across all tag/control comparisons, SHA256
`c23d11bc85d32115fb216e953d384d66cf72801096360b6c68c54e2c9557acdc`.
Reference-conditioning, native FP32, GPU, offline enforcement, default sampling, and timing
methodology are unchanged. Each run loads the model/reference once. Seeds remain 42 for the
first request and 43 for matching warm takes. Alternate-reference experiments are separate.

New eight-text results on GTX 1660 SUPER:

- Model load: 11.93 s; median warm complete-audio-ready: 1.57 s; median RTF: 0.450.
- Peak process VRAM: 3572 MiB. Long response: 8.59 s generation for 20.60 s of audio.
- Nine finite, valid outputs; no inference exceptions or OOMs.
- ASR checked all nine. Three word edits were flagged across two warm samples: “a system”
  versus “the system,” and the GPU line including an “Ah” plus an and/in disagreement.
  The “Ah” may represent the intended sigh. These are review flags, not proven spoken errors.
- Main tag view now contains 15 warm outputs and two first-request outputs, with no empty cells.
  Four documented-event pairs are shown separately. New word and two speaker checks are available.

The combined view links to the actual source runs; it does not invent another timed run.
Per-take timing is retained. View load timing is from its latest source run, and its memory
figures are the maximum across source runs. The initial experiment's aggregate JSONs remain
historical; follow-up measurements/diagnosis are in the new run and `tag-display-diagnosis.json`.

## 4. Events versus the surrounding emotional delivery

The documented controls are **paralinguistic events**, not persistent caring/happy/sleepy modes.
The implementation passes the tagged text through native tokenization and T3 speech-token
generation, then its native decoder. It does not splice prerecorded sound effects into a fixed
neutral utterance. A tag changes the generation context, so surrounding timing, prosody, or
word realization can also vary. There is no API guarantee of a particular surrounding emotion.

Valid audio, preserved words, and speaker scores do not demonstrate Qwen-like emotional depth.
Matched seeds also do not guarantee bit-identical CUDA generation; the untagged serious control
was not byte-identical across independent processes. Differences alone are not proof of an
intentional emotional style. Listening to the same-reference pairs is still required to judge
whether the speech around the event is convincingly more expressive. Do not treat the event
tags or the archived reserved-token probes as verified global emotion controls.

Validation: 586 standard tests passed, three integration tests excluded, Ruff passed, and all
active local media paths checked. Regression tests cover unplanned cells versus failures and
preservation of spoken words while rejecting unverified tag assignments.
