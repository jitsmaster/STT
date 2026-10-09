import asyncio

import numpy as np
import pytest
from fakes import FakeSource, ScriptSpotter
from helpers import LOUD, QUIET, WAKE, build, collect, stop_task

from stt.mic import MicSource, capture_loop, resolve_device
from stt.wake import WakeWatcher

DEVICES = [
    {"name": "Microsoft Sound Mapper - Output", "max_input_channels": 0},
    {"name": "Microphone Array (Realtek)", "max_input_channels": 2},
    {"name": "Headset (soundcore Life Q30) Hands-Free", "max_input_channels": 1},
    {"name": "Speakers", "max_input_channels": 0},
]


def run(coro):
    return asyncio.run(coro)


def test_resolve_device_by_index_digit_string_and_name_fragment():
    assert resolve_device(DEVICES, None) is None
    assert resolve_device(DEVICES, 2) == 2 and resolve_device(DEVICES, "2") == 2
    assert resolve_device(DEVICES, "SoundCore") == 2
    assert resolve_device(DEVICES, "microphone") == 1


def test_resolve_device_refuses_outputs_and_unknown_names_and_lists_the_inputs():
    with pytest.raises(ValueError, match="not an input device"):
        resolve_device(DEVICES, 0)
    with pytest.raises(ValueError, match="not an input device"):
        resolve_device(DEVICES, 99)
    with pytest.raises(ValueError, match="Realtek") as err:
        resolve_device(DEVICES, "nothing")
    assert "no input device matches" in str(err.value)


class FakeInputStream:
    last = None

    def __init__(self, **kwargs):
        self.kwargs, self.active, self.closed = kwargs, False, False
        FakeInputStream.last = self

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


def push(stream, value):
    stream.kwargs["callback"](np.full((1600, 1), value, dtype="float32"), 1600, None, None)


def test_mic_source_opens_a_16k_mono_stream_and_hands_out_flat_frames():
    mic = MicSource(device=3, stream_factory=FakeInputStream)
    assert mic.started is False
    mic.start()
    kw = FakeInputStream.last.kwargs
    assert (kw["samplerate"], kw["channels"], kw["blocksize"], kw["device"], kw["dtype"]) == (16000, 1, 1600, 3, "float32")
    assert mic.next_frame(0.01) is None
    push(FakeInputStream.last, 0.2)
    frame = mic.next_frame(0.01)
    assert frame.shape == (1600,) and abs(float(frame[0]) - 0.2) < 1e-6
    mic.start()  # already open: no second stream
    assert mic.started and FakeInputStream.last.active


def test_stopping_closes_the_stream_and_drops_stale_frames():
    mic = MicSource(stream_factory=FakeInputStream)
    mic.start()
    stream = FakeInputStream.last
    push(stream, 0.2)
    mic.stop()
    assert stream.closed and not stream.active and mic.started is False
    mic.start()
    assert mic.next_frame(0.01) is None


def test_a_full_queue_drops_the_oldest_frames():
    mic = MicSource(stream_factory=FakeInputStream, max_frames=100)
    mic.start()
    for i in range(150):
        push(FakeInputStream.last, i / 1000)
    frames = []
    while (f := mic.next_frame(0.001)) is not None:
        frames.append(f)
    assert len(frames) == 100 and abs(float(frames[0][0]) - 0.05) < 1e-6


def test_the_mic_stays_closed_while_idle_and_opens_only_for_a_listening_window():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        src = FakeSource()
        task = asyncio.create_task(capture_loop(pipe, src, False, poll_s=0.005))
        await asyncio.sleep(0.05)
        assert src.starts == 0 and src.started is False
        pipe.trigger("hotkey")
        await asyncio.sleep(0.05)
        assert src.started
        src.frames += [LOUD, LOUD, QUIET, QUIET]
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        await asyncio.sleep(0.05)
        assert src.started is False and src.starts == 1
        await stop_task(task)

    run(scenario())


def test_with_the_wake_word_on_the_mic_stays_open_and_a_hit_starts_listening():
    async def scenario():
        spot = ScriptSpotter()
        pipe, bus, calls, engines = build(wake=WakeWatcher(spot))
        q = bus.subscribe()
        src = FakeSource()
        task = asyncio.create_task(capture_loop(pipe, src, True, poll_s=0.005))
        await asyncio.sleep(0.05)
        assert src.started
        src.frames += [QUIET, WAKE]
        (ev,) = await collect(q, 1)
        assert ev["type"] == "listening" and ev["source"] == "wake"
        pipe.cancel()
        await asyncio.sleep(0.05)
        assert src.started  # wake word on: never closed while idle
        await stop_task(task)
        assert src.started is False  # but closed when the daemon shuts down

    run(scenario())


def test_a_mic_that_cannot_open_reports_once_cancels_and_keeps_retrying():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        src = FakeSource(fail=True)
        pipe.trigger("hotkey")
        task = asyncio.create_task(capture_loop(pipe, src, False, poll_s=0.005, retry_s=0.01))
        events = await collect(q, 3)
        assert [e["type"] for e in events] == ["listening", "error", "idle"]
        assert "device busy" in events[1]["message"] and events[2]["reason"] == "cancelled"
        assert pipe.state == "idle" and not task.done()
        src.fail = False
        pipe.trigger("hotkey")
        await asyncio.sleep(0.05)
        assert src.started
        await stop_task(task)

    run(scenario())
