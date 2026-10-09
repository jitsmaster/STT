import asyncio

import numpy as np
import pytest
from helpers import LOUD, QUIET, build, collect, speak_cycle


def run(coro):
    return asyncio.run(coro)


def test_a_hotkey_utterance_is_transcribed_and_announced():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        assert pipe.trigger("hotkey", {"target": "s1"}) is True
        for frame in (LOUD, LOUD, QUIET, QUIET):
            pipe.feed(frame)
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        t = events[2]
        assert (t["text"], t["engine"], t["revision"], t["context"]) == ("status", "fast", 1, {"target": "s1"})
        assert events[3]["reason"] == "done"
        assert calls == ["tts-stop", "start", "end"]
        assert engines["command"].hints[0].glossary == ("worktree",)

    run(scenario())


def test_silence_is_announced_without_calling_an_engine():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        pipe.trigger("api")
        for _ in range(3):
            pipe.feed(QUIET)
        events = await collect(q, 2)
        assert [e["type"] for e in events] == ["listening", "idle"] and events[1]["reason"] == "no_speech"
        assert engines["command"].hints == [] and calls[-1] == "cancel"

    run(scenario())


def test_cancel_and_finish_report_whether_anything_was_listening():
    async def scenario():
        pipe, bus, calls, engines = build()
        assert pipe.cancel() is False and pipe.finish() is False
        pipe.trigger("api")
        pipe.feed(LOUD)
        assert pipe.cancel() is True and pipe.state == "idle"

    run(scenario())


def test_retranscribe_uses_the_dictation_engine_and_unknown_ids_raise():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        speak_cycle(pipe)
        uid = (await collect(q, 4))[2]["id"]
        again = await pipe.retranscribe(uid)
        assert (again["text"], again["engine"], again["revision"], again["id"]) == ("status please", "slow", 2, uid)
        assert (await collect(q, 1))[0]["revision"] == 2
        with pytest.raises(LookupError):
            await pipe.retranscribe("nope")

    run(scenario())


def test_near_silent_audio_never_reaches_an_engine():
    async def scenario():
        pipe, bus, calls, engines = build()
        out = await pipe.transcribe_audio(np.zeros(16000, dtype="float32"))
        assert out["text"] == "" and engines["command"].hints == []

    run(scenario())


def test_an_engine_failure_is_reported_and_the_daemon_keeps_working():
    async def scenario():
        pipe, bus, calls, engines = build(command_replies=())
        q = bus.subscribe()
        speak_cycle(pipe)
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "error", "idle"]
        assert events[3]["reason"] == "error" and pipe.state == "idle"
        assert pipe.trigger("hotkey") is True

    run(scenario())


def test_only_the_last_five_utterances_are_kept_for_retranscription():
    async def scenario():
        pipe, bus, calls, engines = build(command_replies=["x"] * 6, dictation_replies=["y"] * 6)
        q = bus.subscribe()
        ids = []
        for _ in range(6):
            speak_cycle(pipe)
            ids.append((await collect(q, 4))[2]["id"])
        with pytest.raises(LookupError):
            await pipe.retranscribe(ids[0])
        assert (await pipe.retranscribe(ids[5]))["text"] == "y"

    run(scenario())


def test_a_subscriber_that_never_reads_cannot_block_publishing():
    async def scenario():
        pipe, bus, calls, engines = build()
        bus.subscribe()
        for _ in range(500):
            bus.publish({"type": "idle", "reason": "x", "context": None})

    run(scenario())
