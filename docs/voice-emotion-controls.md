# Native emotion controls verified on 2026-10-05

This experiment changes neither RAPHAEL's runtime nor its voice configuration. No training,
voice conversion, paid compute, or remote inference is involved.

The installed `chatterbox-tts` package is 0.1.7, built from official source commit
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`. The current official GitHub commit was checked;
both installed `tts.py` and `tts_turbo.py` match that commit byte for byte. Detailed source
hashes, signatures, and token lists are in the ignored experiment's `control-audit.json`.

| Mechanism | Turbo | Original English 500M |
| --- | --- | --- |
| `exaggeration` | Ignored; emotion conditioning is disabled in T3 configuration | Used as the emotion conditioning value; default 0.5 |
| `cfg_weight` | Ignored | Used to combine conditional and unconditional text logits; default 0.5 |
| `min_p` | Ignored | Active sampling filter; default 0.05 |
| Temperature | Active; default 0.8 | Active; default 0.8 |
| `top_p` | Active; default 0.95 | Active; default 1.0 |
| `top_k` | Active; default 1000 | No public generate parameter |
| Reference audio | Native speaker embedding, speech prompt tokens, decoder conditioning | Native speaker embedding, speech prompt tokens, decoder conditioning |
| Named emotion instruction | No documented general instruction interface | No documented general instruction interface |
| Output streaming | Tested stock API returns a complete waveform | Stock API returns a complete waveform |

The distinction is explicit in the [installed-equivalent Turbo implementation](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/tts_turbo.py).
Passing its ignored parameters is not an emotion experiment. The benchmark harness rejects
these overrides for Turbo rather than silently producing a misleading comparison.

Turbo's [official demo](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/gradio_tts_turbo_app.py)
lists nine event tags: `[clear throat]`, `[sigh]`, `[shush]`, `[cough]`, `[groan]`, `[sniff]`,
`[gasp]`, `[chuckle]`, and `[laugh]`. This test uses the four relevant requested events.
`[whisper]` and `[breath]` are absent from the installed native token map and are not used.
`[whispering]`, `[happy]`, and `[sarcastic]` exist as individual native tokenizer tokens;
their samples are explicitly exploratory. Token presence does not establish reliable
delivery control. These probes must not become production controls without listening.

Reference delivery may influence the conditioned speech, but neither implementation
promises disentangled emotion transfer. Turbo consumes up to 15 seconds of reference speech
tokens and 10 seconds of decoder reference audio; original consumes 6 and 10 seconds.
Both speaker encoders use the provided reference. Turbo applies its own default -27 LUFS
reference normalization. The saved source WAVs remain unprocessed, apart from mono extraction.

Both models capitalize the first character, collapse whitespace, normalize selected
punctuation, and add a sentence terminator when absent. Original replaces `...` with a comma;
Turbo keeps it. Internal capitals and `!` / `?` reach tokenization, but are indirect cues,
not guaranteed emotion selectors. Temperature changes sampling variability, not a named
emotion or expression-strength setting. Repetition and pronunciation need checking.

Original profiles are bounded to four sensible combinations:

| Label | Exaggeration | CFG | Temperature |
| --- | ---: | ---: | ---: |
| Default | 0.50 | 0.50 | 0.80 |
| Subtle | 0.60 | 0.40 | 0.80 |
| Moderate | 0.75 | 0.30 | 0.80 |
| Strong | 1.00 | 0.30 | 0.80 |

The [official original-model guidance](https://github.com/resemble-ai/chatterbox#original-chatterbox-tips)
suggests increasing exaggeration to about 0.7 or higher and lowering CFG to about 0.3 for
expression. Higher exaggeration can change pacing. The profiles jointly change both values;
they compare practical settings rather than isolating each parameter's causal effect.
Default means the actual model defaults, not a guarantee of emotionally neutral speech.
The seven emotion names describe the requested conversational scenarios, not measured success.

Every matching warm take uses seed 43 across settings/references. Default original and new
Qwen emotion scenarios also include seed 44. All earlier baseline takes remain accessible;
there is no selection of favorable generated takes. The original eight evaluation texts and
primary reference SHA remain unchanged. All models run separately with offline enforcement.

Use the experiment's results document for measurements, listening paths, and reproduction.
