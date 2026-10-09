"""Explicit, optional Chatterbox Turbo nonverbal events for speech text."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

SpeechExpressiveness = Literal["expressive", "natural", "off"]


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


def voice_delivery_instruction(*, enabled: bool) -> str:
    """Guide spoken phrasing without promising engine-level mood controls."""
    if not enabled:
        return ""
    return (
        "\nVoice delivery: write for the ear, with contractions, varied short clauses, and "
        "natural punctuation. Let wording carry emotion: warm enthusiasm for good news, "
        "gentle patience for difficult moments, dry amusement when joking, and steady focus "
        "during tasks. Match the user's moment rather than sounding uniformly cheerful. "
        "Occasional brief pauses can help; avoid repeated ellipses, filler, exclamation marks, "
        "or theatrical stage directions. Never announce the emotion you are performing."
    )


def speech_event_instruction(
    *, enabled: bool, expressiveness: SpeechExpressiveness = "expressive",
) -> str:
    """Return optional guidance for the dialogue layer, separate from TTS token handling."""
    if not enabled or expressiveness == "off":
        return ""
    supported = ", ".join(event.value for event in SpeechEvent)
    frequency = (
        "Use at most two events in a reply, one per sentence; short replies usually need "
        "zero or one. Let lively casual exchanges have an audible reaction when it fits."
        if expressiveness == "expressive" else
        "Use at most one event per reply, only when it adds clear meaning. Most replies need none."
    )
    return (
        "\nOptional vocal events for spoken replies: " + supported + ". " + frequency +
        " Place a marker at the start of a sentence or at its natural reaction point. "
        "Use them sparingly and deliberately: [chuckle] for amusement or a playful aside, "
        "[laugh] for something truly funny, [sigh] for relief or shared weariness, [gasp] for "
        "real surprise, and [groan] for light frustration. A sigh must not sound like annoyance "
        "at the user. Keep serious explanations and sensitive disclosures calm; never laugh "
        "at distress. Don't routinely cough, sniff, clear your throat, or shush; use those "
        "only when the conversation calls for that sound. Do not stack events, decorate "
        "every sentence, invent bracket tags or sustained mood controls, or describe the "
        "event instead of using its marker. These markers are speech cues, not dialogue."
    )


_EVENT_PREFIX = re.compile(r"^\s*(?:\[[^\]\r\n]{1,24}\]\s*)+")
_CANONICAL_EVENTS = {event.value.casefold(): event.value for event in SpeechEvent}
# Application aliases; neither is an additional native Turbo token.
_CANONICAL_EVENTS.update({"[moan]": SpeechEvent.GROAN.value,
                          "[giggle]": SpeechEvent.CHUCKLE.value})
_EVENT_MARKERS = re.compile("|".join(map(re.escape, _CANONICAL_EVENTS)), re.IGNORECASE)


class SpeechEventLimiter:
    """Canonicalize cues and bound their frequency for one reply, including streaming."""

    def __init__(self, expressiveness: SpeechExpressiveness = "expressive") -> None:
        self.remaining = {"expressive": 2, "natural": 1, "off": 0}[expressiveness]

    def apply(self, text: str) -> str:
        """Keep supported inline cues in place; discard excess or stacked cues."""
        last_event_end: int | None = None

        def replace(match: re.Match[str]) -> str:
            nonlocal last_event_end
            # Several adjacent tags are one reaction, not a sequence of noises.
            stacked = last_event_end is not None and not text[
                last_event_end:match.start()
            ].strip()
            last_event_end = match.end()
            if self.remaining <= 0 or stacked:
                return ""
            self.remaining -= 1
            return _CANONICAL_EVENTS[match.group().casefold()]

        return _EVENT_MARKERS.sub(replace, text).strip()


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
    events = {key[1:-1]: value for key, value in _CANONICAL_EVENTS.items()}
    # Drop unknown leading markup without losing a valid neighboring cue.
    prefix = " ".join(
        events[token.strip().casefold()] for token in tokens
        if token.strip().casefold() in events
    )
    return prefix, text[match.end() :].strip()


def strip_speech_events(text: str) -> str:
    """Remove native cues and application aliases from captions, transcripts, and history."""
    return _EVENT_MARKERS.sub("", text).strip()
