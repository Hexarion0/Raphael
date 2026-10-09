"""Desktop chat/status service uses bounded state and validated local requests."""

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from raphael.web import DesktopWebUI, WebErrorHandler


@pytest.fixture
def web_server():
    ui = DesktopWebUI("hexarion", "hey raphael")
    submit = MagicMock(return_value=True)
    url = ui.start(submit, port=0)
    with httpx.Client(base_url=url, trust_env=False, timeout=2) as client:
        yield ui, submit, client
    ui.close()


def test_web_chat_progress_keeps_sentences_in_one_reply_and_freezes_interruption():
    ui = DesktopWebUI("owner", "hey raphael")
    ui.user_message("Hello")
    ui.start_sentence()
    ui.progress("Hi there.")
    ui.start_sentence()
    ui.progress("What")
    assert ui.snapshot()["messages"][-1]["text"] == "Hi there. What"
    ui.progress("What's up", interrupted=True)
    ui.progress("What's up with your project?")
    reply = ui.snapshot()["messages"][-1]
    assert reply["text"] == "Hi there. What's up"
    assert reply["interrupted"] and not reply["live"]


def test_empty_unplayed_sentence_does_not_create_a_fake_reply():
    ui = DesktopWebUI("owner", "hey raphael")
    ui.start_sentence()
    ui.finish_reply()
    assert ui.snapshot()["messages"] == []
    ui.text_reply("Voice unavailable, but here is the answer.")
    assert ui.snapshot()["messages"][-1]["text"].startswith("Voice unavailable")


def test_web_state_is_bounded_and_secrets_are_masked():
    ui = DesktopWebUI("owner", "hey raphael")
    for number in range(205):
        ui.user_message(str(number))
    ui.user_message("key nvapi-0123456789secret")
    ui.set_muted(True)
    ui.set_state("processing")
    for number in range(35):
        ui.feedback(str(number))
    snapshot = ui.snapshot()
    assert len(snapshot["messages"]) == 200
    assert len(snapshot["activity"]) == 30
    assert snapshot["muted"] and snapshot["state"] == "processing"
    assert snapshot["messages"][-1]["text"] == "key [REDACTED_SECRET]"
    snapshot["messages"][-1]["text"] = "changed"
    assert ui.snapshot()["messages"][-1]["text"] == "key [REDACTED_SECRET]"


def test_http_serves_page_and_submits_to_the_existing_worker(web_server):
    ui, submit, client = web_server
    response = client.get("/")
    assert response.status_code == 200
    assert "Message Raphael" in response.text
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert client.get("/api/state").json()["owner"] == "hexarion"
    for text in ["hello", "/mute", "/stop", "/exit"]:
        assert client.post("/api/message", json={"text": text}).json() == {"accepted": True}
    assert [call.args[0] for call in submit.call_args_list] == [
        "hello", "/mute", "/stop", "/exit",
    ]
    assert ui.snapshot()["messages"] == []  # Queuing isn't proof of a completed reply.


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"}, {"Origin": "https://evil.example"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_http_rejects_requests_from_other_websites(web_server, headers):
    _ui, submit, client = web_server
    assert client.post("/api/message", json={"text": "/exit"}, headers=headers).status_code == 403
    submit.assert_not_called()


@pytest.mark.parametrize("data", [{}, [], {"text": ""}, {"text": 5}, {"text": "x" * 4001}])
def test_http_rejects_invalid_messages(web_server, data):
    _ui, submit, client = web_server
    assert client.post("/api/message", json=data).status_code == 400
    submit.assert_not_called()


def test_http_reports_unavailable_worker_and_does_not_accept_form_posts(web_server):
    _ui, submit, client = web_server
    assert client.post("/api/message", data={"text": "/exit"}).status_code == 415
    submit.return_value = False
    response = client.post("/api/message", json={"text": "hello"})
    assert response.status_code == 503
    assert response.json() == {"accepted": False}


def test_runtime_warnings_are_masked_for_browser():
    ui = DesktopWebUI("owner", "hey raphael")
    handler = WebErrorHandler(ui)
    handler.handle(logging.LogRecord(
        "raphael", logging.ERROR, "file.py", 1, "key=%s", ("sk-0123456789secret",), None,
    ))
    assert ui.snapshot()["activity"][-1]["text"] == "key=[REDACTED_SECRET]"


def test_history_uses_current_conversation():
    ui = DesktopWebUI("owner", "hey raphael")
    ui.load_history([SimpleNamespace(role="user", content="hello"),
                     SimpleNamespace(role="assistant", content="Hi!")])
    assert [entry["text"] for entry in ui.snapshot()["messages"]] == ["hello", "Hi!"]


def test_history_preserves_dates_and_omits_internal_playback_notes():
    ui = DesktopWebUI("owner", "hey raphael")
    timestamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
    ui.load_history([SimpleNamespace(
        role="assistant", content="Hi!\n[Playback was interrupted.]", timestamp=timestamp,
    )])
    message = ui.snapshot()["messages"][0]
    assert message["text"] == "Hi!"
    assert message["time"] == timestamp.timestamp()


def test_http_get_rejects_untrusted_host_and_port_conflicts_are_reported(web_server):
    _ui, _submit, client = web_server
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 403
    second_ui = DesktopWebUI("owner", "hey raphael")
    with pytest.raises(OSError):
        second_ui.start(lambda _text: True, port=client.base_url.port)
    second_ui.close()
