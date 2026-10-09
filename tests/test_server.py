import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from helpers import build, speak_cycle
from starlette.websockets import WebSocketDisconnect

from stt.audio import SAMPLE_RATE
from stt.server import create_app, host_ok

SECRET = "s" * 64
AUTH = {"Authorization": "Bearer " + SECRET}
BASE = "http://127.0.0.1:8890"
WS = "ws://127.0.0.1:8890"  # TestClient.websocket_connect ignores base_url and would send Host: testserver
INFO = {"version": "t", "hotkey": "ctrl+alt+v", "wake": False, "engines": {"command": "fast", "dictation": "slow"}}


def wav_bytes(seconds=1.0, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    buf = io.BytesIO()
    sf.write(buf, (amp * np.sin(2 * np.pi * 440 * t)).astype("float32"), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def api():
    pipe, bus, calls, engines = build(command_replies=["status"] * 5, dictation_replies=["status please"] * 5)
    with TestClient(create_app(pipe, bus, SECRET, INFO), base_url=BASE) as client:
        yield client, pipe, bus, engines


def test_host_ok_accepts_loopback_names_only():
    for good in ("127.0.0.1", "127.0.0.1:8890", "localhost:9", "LOCALHOST", "[::1]:8890", "::1"):
        assert host_ok(good), good
    for bad in ("", None, "evil.example", "evil.example:8890", "127.0.0.1.evil.example", "0.0.0.0", "10.0.0.5:8890"):
        assert not host_ok(bad), bad


def test_health_is_open_but_not_to_a_foreign_host(api):
    client, pipe, bus, engines = api
    body = client.get("/health").json()
    assert body == {"ok": True, "version": "t", "state": "idle"}
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 403


PROTECTED = [("/listen", {"json": {}}), ("/finish", {}), ("/cancel", {}), ("/transcribe", {"content": b"x"}),
             ("/retranscribe", {"json": {"id": "a"}})]


@pytest.mark.parametrize("path,kw", PROTECTED)
def test_protected_endpoints_refuse_missing_and_wrong_tokens_and_foreign_hosts(api, path, kw):
    client, pipe, bus, engines = api
    assert client.post(path, **kw).status_code == 401
    assert client.post(path, headers={"Authorization": "Bearer nope"}, **kw).status_code == 401
    assert client.post(path, headers={"Authorization": "Basic " + SECRET}, **kw).status_code == 401
    assert client.post(path, headers={**AUTH, "Host": "evil.example"}, **kw).status_code == 403
    assert pipe.state == "idle" and engines["command"].hints == []


def test_listen_starts_once_validates_the_window_and_cancel_and_finish_report_state(api):
    client, pipe, bus, engines = api
    assert client.post("/finish", headers=AUTH).json() == {"finished": False}
    assert client.post("/cancel", headers=AUTH).json() == {"cancelled": False}
    assert client.post("/listen", headers=AUTH).json() == {"started": True}
    assert client.post("/listen", headers=AUTH).json() == {"started": False}
    assert pipe.state == "listening"
    assert client.post("/cancel", headers=AUTH).json() == {"cancelled": True}
    for bad in (0, -1, 61):
        assert client.post("/listen", json={"window_s": bad}, headers=AUTH).status_code == 422
    assert pipe.state == "idle"
    assert client.post("/listen", json={"window_s": 60, "context": {"target": "s1"}}, headers=AUTH).json() == {"started": True}


def test_the_websocket_says_hello_then_streams_events_with_the_listen_context(api):
    client, pipe, bus, engines = api
    with client.websocket_connect(WS + "/events?token=" + SECRET) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["version"] == "t" and hello["state"] == "idle"
        client.post("/listen", json={"context": {"target": "s1"}}, headers=AUTH)
        assert ws.receive_json() == {"type": "listening", "source": "api", "context": {"target": "s1"}}
        client.post("/cancel", headers=AUTH)
        assert ws.receive_json()["reason"] == "cancelled"


def test_the_websocket_refuses_a_wrong_token_a_missing_token_and_a_foreign_host(api):
    client, pipe, bus, engines = api
    for url, headers in (("/events?token=nope", {}), ("/events", {}), ("/events?token=" + SECRET, {"Host": "evil.example"})):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(WS + url, headers=headers):
                pass
    assert not bus._subs


def test_a_closed_websocket_is_unsubscribed(api):
    client, pipe, bus, engines = api
    with client.websocket_connect(WS + "/events?token=" + SECRET) as ws:
        ws.receive_json()
        assert len(bus._subs) == 1
    deadline = time.time() + 2
    while bus._subs and time.time() < deadline:
        time.sleep(0.02)
    assert not bus._subs


def test_transcribe_uses_the_requested_engine_and_never_asks_an_engine_about_silence(api):
    client, pipe, bus, engines = api
    fast = client.post("/transcribe", content=wav_bytes(), headers=AUTH).json()
    assert (fast["text"], fast["engine"]) == ("status", "fast")
    slow = client.post("/transcribe?kind=dictation&language=en", content=wav_bytes(), headers=AUTH).json()
    assert (slow["text"], slow["engine"]) == ("status please", "slow")
    assert engines["dictation"].hints[0].language == "en"
    assert client.post("/transcribe", content=wav_bytes(amp=0.0), headers=AUTH).json()["text"] == ""
    assert len(engines["command"].hints) == 1
    assert client.post("/transcribe?kind=nope", content=wav_bytes(), headers=AUTH).status_code == 422


def test_transcribe_rejects_unreadable_and_oversized_bodies(api):
    client, pipe, bus, engines = api
    assert client.post("/transcribe", content=b"this is not a wav file", headers=AUTH).status_code == 400
    assert client.post("/transcribe", content=b"0" * (10 * 1024 * 1024 + 1), headers=AUTH).status_code == 413
    assert engines["command"].hints == []


def test_retranscribe_redoes_a_recent_utterance_with_the_dictation_engine(api):
    client, pipe, bus, engines = api

    async def speak():
        speak_cycle(pipe)

    with client.websocket_connect(WS + "/events?token=" + SECRET) as ws:
        assert ws.receive_json()["type"] == "hello"
        client.portal.call(speak)
        events = [ws.receive_json() for _ in range(4)]
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        uid = events[2]["id"]
        body = client.post("/retranscribe", json={"id": uid}, headers=AUTH).json()
        assert (body["text"], body["engine"], body["revision"], body["id"]) == ("status please", "slow", 2, uid)
        assert ws.receive_json()["revision"] == 2
    assert client.post("/retranscribe", json={"id": "nope"}, headers=AUTH).status_code == 404


def test_an_engine_failure_is_a_502_and_the_daemon_keeps_answering():
    pipe, bus, calls, engines = build(command_replies=())
    with TestClient(create_app(pipe, bus, SECRET, INFO), base_url=BASE) as client:
        reply = client.post("/transcribe", content=wav_bytes(), headers=AUTH)
        assert reply.status_code == 502 and "engine failed" in reply.json()["detail"]
        assert client.get("/health").status_code == 200
