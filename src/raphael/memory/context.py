"""Build dated, bounded memory context without replaying archived conversation summaries."""

import re
from datetime import date, datetime

from raphael.memory.models import MemoryItem, MemoryType
from raphael.memory.recall import rank_memories, terms
from raphael.memory.store import MemoryStore
from raphael.providers.base import ChatMessage

PROFILE_KEYS = ("user:preferred_name", "user:current_project", "user:goal", "user:occupation")
FOLLOWUP = re.compile(
    r"^(?:continue|go on|tell me more|more details|what about (?:that|it|this)|"
    r"how (?:about|long)|and\b|also\b|but\b|so\b)|\b(?:that|this|it|those|them)\b", re.I,
)
REFERENCE_TERMS = set(
    "continue go more detail details long take takes last same still again it this that those them "
    "one ones thing things else next also but so explain work works mean means much cost give "
    "another".split()
)


def contextual_recall_query(query: str, recent_messages: list[ChatMessage]) -> str:
    """Resolve short references from user dialogue, without borrowing assistant claims."""
    if (
        len(query.split()) > 12 or not FOLLOWUP.search(query)
        or terms(query) - REFERENCE_TERMS
    ):
        return query
    for message in reversed(recent_messages[-8:]):
        if message.role != "user" or message.content.strip() == query.strip():
            continue
        previous = message.content[:500].strip()
        if previous and terms(previous) - REFERENCE_TERMS:
            return f"{query}\n{previous}"
    return query


def recall_context_memories(
    store: MemoryStore, query: str = "", now: datetime | None = None,
    *, recent_messages: list[ChatMessage] | None = None,
) -> list[str]:
    """Recall durable facts with timestamps and authoritative project date arithmetic."""
    current_date = (now or datetime.now()).date()
    recalled: list[MemoryItem] = []
    if query:
        contextual_query = contextual_recall_query(query, recent_messages or [])
        recalled.extend(item for _, item in rank_memories(store, contextual_query))
    # A small confirmed profile supplies continuity even for informal conversation.
    for key in PROFILE_KEYS:
        item = store.get_fact(key)
        if item is not None:
            recalled.append(item)
    # Project anchors must survive informal questions that don't match their exact wording.
    recalled.extend(store.list_memories(memory_type=MemoryType.PROJECT, limit=3))
    start = store.get_fact("raphael:project_start_date")
    if start is not None:
        recalled.append(start)
    recalled.extend(store.list_memories(memory_type=MemoryType.PREFERENCE, limit=3))

    context: list[str] = []
    seen: set[int | str] = set()
    for memory in recalled:
        identity = memory.id if memory.id is not None else memory.content
        if identity in seen:
            continue
        seen.add(identity)
        recorded_at = memory.metadata.get("observed_at") or (
            memory.created_at.astimezone().isoformat(timespec="seconds")
        )
        content = store.redact_forgotten(memory.content)[:600]
        context.append(
            f"Recorded at {recorded_at}: {content} "
            f"[type={memory.memory_type.value}; source={memory.source}]"
        )
        if (
            memory.memory_type == MemoryType.PROJECT
            and memory.metadata.get("project") == "raphael"
            and memory.metadata.get("fact_key") == "project_start_date"
        ):
            try:
                start_date = date.fromisoformat(str(memory.metadata["value"]))
            except (KeyError, ValueError):
                continue
            elapsed = (current_date - start_date).days
            if elapsed >= 0:
                context.append(
                    f"RAPHAEL timeline as of {current_date.isoformat()}: confirmed project "
                    f"start date {start_date.isoformat()}; {elapsed} elapsed calendar days; "
                    f"development day {elapsed + 1} counting the start date as day one. "
                    "Use this dated calculation instead of old relative day-count statements."
                )
    bounded: list[str] = []
    size = 0
    for entry in context:
        if size + len(entry) > 6000:
            break
        bounded.append(entry)
        size += len(entry)
    return bounded
