import asyncio

from fakes import ScriptSpotter
from helpers import LOUD, QUIET, WAKE, build, collect, speak_cycle

from stt.wake import WakeWatcher


def run(coro):
    return asyncio.run(coro)


def test_a_hit_is_reported_for_the_frame_that_holds_the_keyword_only():
    spot = ScriptSpotter()
    watcher = WakeWatcher(spot)
    assert watcher.feed(QUIET) is False
    assert watcher.feed(WAKE) is True
    assert spot.resets == 1
    assert watcher.feed(QUIET) is False


def test_reset_starts_a_fresh_stream():
    spot = ScriptSpotter()
    watcher = WakeWatcher(spot)
    assert spot.created == 1
    watcher.reset()
    assert spot.created == 2


def test_a_wake_hit_starts_listening_with_source_wake():
    async def scenario():
        spot = ScriptSpotter()
        pipe, bus, calls, engines = build(wake=WakeWatcher(spot))
        q = bus.subscribe()
        pipe.feed(QUIET)
        assert pipe.state == "idle"
        pipe.feed(WAKE)
        (ev,) = await collect(q, 1)
        assert ev == {"type": "listening", "source": "wake", "context": None}
        assert pipe.state == "listening" and calls[:2] == ["tts-stop", "start"]

    run(scenario())


def test_the_watcher_is_not_fed_while_listening_and_restarts_after_the_utterance():
    async def scenario():
        spot = ScriptSpotter()
        pipe, bus, calls, engines = build(wake=WakeWatcher(spot))
        q = bus.subscribe()
        pipe.feed(WAKE)
        for frame in (WAKE, LOUD, QUIET, QUIET):  # the second WAKE is speech inside the utterance, not a new trigger
            pipe.feed(frame)
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        assert spot.accepted == 1  # only the first frame reached the spotter
        assert spot.created == 2  # the stream was replaced when the utterance ended

    run(scenario())


def test_a_hotkey_utterance_also_resets_the_watcher():
    async def scenario():
        spot = ScriptSpotter()
        pipe, bus, calls, engines = build(wake=WakeWatcher(spot))
        q = bus.subscribe()
        speak_cycle(pipe)
        await collect(q, 4)
        assert spot.created == 2

    run(scenario())
