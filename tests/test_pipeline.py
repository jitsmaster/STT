import asyncio
import threading

import numpy as np
import pytest
from fakes import FakeEngine, ScriptEndpointer
from helpers import LOUD, QUIET, build, collect, speak_cycle

from stt.core import ListenerCore
from stt.pipeline import Bus, Pipeline


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


# --- spoken end word --------------------------------------------------------------------------------------------------

END_KW = {"silence_end_s": 15.0, "max_s": 60.0, "no_speech_s": 1.0, "end_check": True, "min_speech_s": 0.15}


def build_end(replies, **kw):
    return build(command_replies=replies, core_kw=END_KW, end_words=("over",), **kw)


def say(pipe, speech, gap):
    for _ in range(speech):
        pipe.feed(LOUD)
    for _ in range(gap):
        pipe.feed(QUIET)


async def settle(pipe):
    while pipe._tasks:
        await asyncio.gather(*list(pipe._tasks))
        await asyncio.sleep(0)


def test_saying_over_ends_the_window_and_is_not_part_of_the_transcript():
    async def scenario():
        pipe, bus, calls, engines = build_end(["Over.", "fix the build"])
        q = bus.subscribe()
        pipe.trigger("hotkey", {"target": "s1"})
        say(pipe, 4, 5)  # "fix the build", then a 0.5 s pause
        say(pipe, 2, 6)  # "over", then 0.6 s of silence
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        assert events[2]["text"] == "fix the build" and events[3]["reason"] == "done"
        check, final = engines["command"].audios
        assert len(check) == (3 + 2) * 1600 and len(final) == (4 + 5) * 1600  # the final decode never saw the burst
        assert engines["command"].hints[0].language == "en" and pipe.state == "idle"

    run(scenario())


def test_saying_only_over_closes_the_window_as_no_speech():
    async def scenario():
        pipe, bus, calls, engines = build_end(["over"])
        q = bus.subscribe()
        pipe.trigger("api")
        say(pipe, 2, 6)
        events = await collect(q, 2)
        assert [e["type"] for e in events] == ["listening", "idle"] and events[1]["reason"] == "no_speech"
        assert len(engines["command"].audios) == 1

    run(scenario())


@pytest.mark.parametrize("heard", ["all over", "Hover.", "hand it over", "Over, over.", "Okay."])
def test_anything_but_the_end_word_alone_leaves_the_window_open(heard):
    async def scenario():
        pipe, bus, calls, engines = build_end([heard])
        q = bus.subscribe()
        pipe.trigger("hotkey")
        say(pipe, 2, 6)
        await settle(pipe)
        assert pipe.state == "listening" and q.qsize() == 1  # only "listening"

    run(scenario())


def test_a_failing_check_is_ignored_and_a_later_end_word_still_works():
    async def scenario():
        pipe, bus, calls, engines = build_end([RuntimeError("boom"), "over", "fix it"])
        q = bus.subscribe()
        pipe.trigger("hotkey")
        say(pipe, 3, 6)
        await settle(pipe)
        assert pipe.state == "listening" and q.qsize() == 1
        say(pipe, 2, 6)
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        assert events[2]["text"] == "fix it"

    run(scenario())


def test_two_quick_checks_keep_only_the_newest_pending_one():
    async def scenario():
        gate = threading.Event()
        pipe, bus, calls, engines = build_end(["hover", "hover"], gate=gate)
        pipe.trigger("hotkey")
        say(pipe, 2, 6)  # check A starts and blocks inside the engine
        await asyncio.sleep(0.05)
        say(pipe, 3, 6)  # check B waits
        say(pipe, 4, 6)  # check C replaces B
        gate.set()
        await settle(pipe)
        assert [len(a) for a in engines["command"].audios] == [2 * 1600, (3 + 4) * 1600]
        assert pipe.state == "listening"

    run(scenario())


def test_a_check_that_comes_back_after_the_user_spoke_again_is_ignored():
    async def scenario():
        gate = threading.Event()
        pipe, bus, calls, engines = build_end(["over"], gate=gate)
        q = bus.subscribe()
        pipe.trigger("hotkey")
        say(pipe, 2, 6)
        await asyncio.sleep(0.05)
        say(pipe, 2, 0)  # the user carried on talking while the engine was busy
        gate.set()
        await settle(pipe)
        assert pipe.state == "listening" and q.qsize() == 1

    run(scenario())


def test_without_end_words_no_engine_is_ever_asked():
    async def scenario():
        pipe, bus, calls, engines = build(command_replies=[], core_kw=END_KW, end_words=())
        pipe.trigger("hotkey")
        say(pipe, 2, 6)
        await settle(pipe)
        assert engines["command"].audios == [] and pipe.state == "listening"

    run(scenario())


def test_silent_audio_is_not_sent_to_the_engine_for_an_end_check():
    async def scenario():
        core = ListenerCore(ScriptEndpointer("SS" + "." * 8), **{**END_KW, "end_gap_s": 0.6})
        engine = FakeEngine("fast", [])
        pipe = Pipeline(core, {"command": engine}, Bus(), end_words=("over",))
        pipe.trigger("hotkey")
        for _ in range(10):
            pipe.feed(np.zeros(1600, dtype="float32"))  # the script says speech, the samples are silent
        await settle(pipe)
        assert engine.audios == [] and pipe.state == "listening"

    run(scenario())
