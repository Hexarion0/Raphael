"""Tests for structured logging and secret masking."""

import logging

from raphael.logging import ConciseLogFilter, SecretMaskingFilter, get_logger, setup_logging


def test_secret_masking_filter():
    """Verify that known sensitive API key patterns are masked."""
    redactor = SecretMaskingFilter()

    # Test raw message masking
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Using key nvapi-abcdef1234567890 for NIM API",
        args=(),
        exc_info=None,
    )
    assert redactor.filter(record) is True
    assert "nvapi-abcdef1234567890" not in record.msg
    assert "[REDACTED_SECRET]" in record.msg

    # Test args masking (tuple/list)
    record_args = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Calling Groq with key: %s",
        args=("gsk_1234567890abcdefgh",),
        exc_info=None,
    )
    assert redactor.filter(record_args) is True
    assert "gsk_1234567890abcdefgh" not in record_args.args[0]
    assert "[REDACTED_SECRET]" in record_args.args[0]


def test_setup_logging():
    """Verify logger creation and namespacing."""
    logger = setup_logging("DEBUG")
    assert logger.name == "raphael"

    sub_logger = get_logger("providers.nim")
    assert sub_logger.name == "raphael.providers.nim"


def test_concise_log_filter_keeps_conversation_and_problems_only():
    concise = ConciseLogFilter()

    def record(level, message):
        return logging.LogRecord(
            name="raphael.audio.listener", level=level, pathname="test.py", lineno=1,
            msg=message, args=(), exc_info=None,
        )

    assert concise.filter(record(logging.INFO, '🗣️ You: "hello"'))
    assert concise.filter(record(logging.INFO, '🤖 RAPHAEL: "hi"'))
    assert concise.filter(record(logging.INFO, "Microphone active — ambient listening"))
    assert concise.filter(record(logging.INFO, "Chatterbox Turbo ready (pid=1)"))
    assert not concise.filter(record(logging.INFO, "Chatterbox Turbo ready (pid=1)"))
    assert not concise.filter(record(logging.INFO, "Transcribing 4.8s of audio"))
    assert not concise.filter(record(logging.DEBUG, "provider payload details"))
    assert concise.filter(record(logging.WARNING, "provider request failed"))
