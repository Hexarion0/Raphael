"""Explicit, optional Chatterbox Turbo nonverbal events for speech text."""

from __future__ import annotations

import re
from enum import Enum


class SpeechEvent(str, Enum):
    """The complete native event set from the pinned Chatterbox Turbo tokenizer."""

    CLEAR_THROAT = "[clear throat]"
    SIGH = "[sigh]"
    SHUSH = "[shush]"
    COUGH = "[cough]"
    GROAN = "[groan]"
    SNIFF = "[sniff]"
    GASP = "[gasp]"
    CHUCKLE = "[chuckle]"
    LAUGH = "[laugh]"


def speech_event_instruction(*, enabled: bool) -> str:
    """Return optional guidance for the dialogue layer, separate from TTS token handling."""
    if not enabled:
        return ""
    supported = ", ".join(event.value for event in SpeechEvent)
    return (
        "\nOptional vocal events for spoken replies: when a brief audible event would feel "
        "natural and add meaning, you may put exactly one of these supported markers at the "
        "start of that sentence: " + supported + ". Use them sparingly; ordinary replies "
        "should contain no marker. Do not use markers for sustained emotion, do not invent "
        "other bracket tags, and do not describe an event instead of simply using its marker."
    )


_EVENT_PREFIX = re.compile(r"^\s*(?:\[[^\]\r\n]{1,24}\]\s*)+")
_EVENT_MARKERS = re.compile(
    "|".join(re.escape(event.value) for event in SpeechEvent), re.IGNORECASE
)


def add_speech_event(text: str, event: SpeechEvent | None) -> str:
    """Attach a deliberate event request to one sentence; never insert events implicitly."""
    content = text.strip()
    if event is None or not content:
        return content
    if not isinstance(event, SpeechEvent):
        raise TypeError("event must be a documented SpeechEvent")
    marker = event.value
    return content if content.lower().startswith(marker) else f"{marker} {content}"


def split_speech_event(text: str) -> tuple[str, str]:
    """Return canonical supported leading events separately from caption text."""
    match = _EVENT_PREFIX.match(text)
    if match is None:
        return "", text.strip()
    raw = match.group(0)
    tokens = re.findall(r"\[([^\]]+)\]", raw)
    events = {event.value[1:-1]: event.value for event in SpeechEvent}
    if any(token.strip().casefold() not in events for token in tokens):
        # Unknown leading tags are markup requests, never fake native tokens or spoken text.
        return "", text[match.end() :].strip()
    prefix = " ".join(events[token.strip().casefold()] for token in tokens)
    return prefix, text[match.end() :].strip()


def strip_speech_events(text: str) -> str:
    """Remove supported markers anywhere in visible captions/transcripts."""
    return _EVENT_MARKERS.sub("", text).strip()
