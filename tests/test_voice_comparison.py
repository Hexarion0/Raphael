"""Protect listening artifacts from omitted takes and unescaped source text."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_spelling_normalization_does_not_hide_different_numbers():
    assess = load_script("assess_cloning_audio")
    assert assess.tokens("GPU at 63; 4.2 gigabytes; 9:30 AM") == assess.tokens(
        "G P U at sixty three; four point two gigabytes; nine thirty A M"
    )
    assert assess.word_distance(assess.tokens("63 degrees"), assess.tokens("62 degrees")) > 0
    assert assess.tokens("9:30 AM workstation") == assess.tokens("9.30am work station")
    assert assess.tokens("9 30 a.m.") == assess.tokens("nine thirty a m")


def test_edit_distance_preserves_omitted_and_extra_words():
    assess = load_script("assess_cloning_audio")
    assert assess.word_distance(["your", "server", "is", "online"], ["server", "online"]) == 2
    assert assess.word_distance(["of", "course"], ["of", "course", "yes"]) == 1


def test_comparison_keeps_all_takes_and_escapes_transcript(tmp_path, monkeypatch):
    script = load_script("build_voice_comparison")
    model = tmp_path / "model"
    model.mkdir()
    (model / "summary.json").write_text(
        json.dumps(
            {
                "candidate": "test-model",
                "errors": [],
                "streaming": "complete waveform",
            }
        )
    )
    (model / "measurements.json").write_text(
        json.dumps(
            [
                {
                    "phase": "warm",
                    "id": "short",
                    "text": "<script>untrusted</script>",
                    "repeat": take,
                    "audio": f"take-{take}.wav",
                    "audio_ready_seconds": take,
                    "generation_seconds": take,
                    "rtf": 0.5,
                }
                for take in [1, 2, 3]
            ]
        )
    )
    refs = tmp_path / "references.json"
    refs.write_text(
        json.dumps(
            [
                {
                    "start": 20,
                    "end": 32,
                    "source_url": "https://youtu.be/PYmd20HsBj4?x=1",
                    "video_id": "PYmd20HsBj4",
                    "reference_wav": str(tmp_path / "reference.wav"),
                    "text": "<script>source</script>",
                    "reasons": [],
                }
            ]
        )
    )
    output = tmp_path / "index.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_voice_comparison.py",
            str(model),
            "--references",
            str(refs),
            "--output",
            str(output),
        ],
    )
    script.main()
    page = output.read_text()
    for take in [1, 2, 3]:
        assert f"take-{take}.wav" in page
    assert "&lt;script&gt;untrusted&lt;/script&gt;" in page
    assert "<script>untrusted</script>" not in page
    assert "&lt;script&gt;source&lt;/script&gt;" in page


def test_profiles_keep_unique_notes_and_show_tagged_input(tmp_path, monkeypatch):
    script = load_script("build_voice_comparison")
    directories = []
    for profile, text in [("neutral", "Hello."), ("tagged", "[chuckle] Hello.")]:
        directory = tmp_path / profile
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({
            "candidate": "chatterbox-turbo", "display_name": "Turbo " + profile,
            "errors": [], "reference_path": str(tmp_path / (profile + ".wav")),
        }))
        (directory / "measurements.json").write_text(json.dumps([{
            "id": "hello", "phase": "warm", "repeat": 1, "text": text,
            "spoken_text": "Hello.", "audio": "hello.wav", "audio_ready_seconds": 1,
            "generation_seconds": 1, "rtf": 0.5,
        }]))
        directories.append(str(directory))
    references = tmp_path / "references.json"
    references.write_text("[]")
    output = tmp_path / "index.html"
    monkeypatch.setattr(sys, "argv", [
        "build_voice_comparison.py", *directories,
        "--references", str(references), "--output", str(output),
    ])
    script.main()
    page = output.read_text()
    assert "Turbo neutral" in page and "Turbo tagged" in page
    assert 'data-key="neutral-hello"' in page
    assert 'data-key="tagged-hello"' in page
    assert "Actual input: [chuckle] Hello." in page
    assert "Reference: neutral.wav" in page
    assert "Reference: tagged.wav" in page


def test_unplanned_and_missing_samples_are_not_reported_as_inference_errors():
    script = load_script("build_voice_comparison")
    summary = {"planned_sentence_ids": ["emotion"], "errors": []}
    assert script.missing_sample_message(summary, "baseline") == "Not scheduled for this run."
    assert "incomplete" in script.missing_sample_message(summary, "emotion")
    assert "errors" not in script.missing_sample_message({"errors": []}, "baseline")
    failed = {"planned_sentence_ids": ["emotion"], "errors": ["CUDA out of memory"]}
    assert "errors" in script.missing_sample_message(failed, "emotion")
    assert "errors" not in script.missing_sample_message(failed, "baseline")


def test_supported_tag_baseline_preserves_spoken_words_and_rejects_unverified_tags():
    import re

    script = load_script("benchmark_turbo_supported_tags")
    baseline = json.loads((SCRIPTS.parent / "docs/voice-evaluation.json").read_text())
    allowed = {"[sigh]", "[chuckle]", "[gasp]"}
    native = {tag: index for index, tag in enumerate(sorted(allowed))}
    rows = script.baseline_tag_suite(baseline, allowed, native)
    assert [r["id"] for r in rows] == [r["id"] for r in baseline]
    for row, original in zip(rows, baseline):
        assert row["spoken_text"] == original["text"]
        assert re.sub(r"\[[^]]+\]\s*", "", row["text"]) == original["text"]
        assert set(re.findall(r"\[[^]]+\]", row["text"])) <= allowed
    import pytest

    with pytest.raises(ValueError, match="Not a documented native Turbo event"):
        script.baseline_tag_suite(baseline, {"[sigh]"}, native)
