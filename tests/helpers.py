import contextlib
import asyncio

import numpy as np
from fakes import FakeEngine

from stt.core import ListenerCore
from stt.endpointers import EnergyEndpointer
from stt.pipeline import Bus, Pipeline

LOUD = np.full(1600, 0.2, dtype="float32")
QUIET = np.zeros(1600, dtype="float32")
WAKE = np.full(1600, 0.9, dtype="float32")


def build(command_replies=("status",), dictation_replies=("status please",), wake=None, core_kw=None, end_words=(), gate=None):
    kw = {"no_speech_s": 0.3, "silence_end_s": 0.2, "max_s": 5.0, "min_speech_s": 0.1, **(core_kw or {})}
    core = ListenerCore(EnergyEndpointer(0.05), **kw)
    engines = {"command": FakeEngine("fast", command_replies, gate), "dictation": FakeEngine("slow", dictation_replies)}
    bus, calls = Bus(), []
    pipe = Pipeline(core, engines, bus, tts_stop=lambda: calls.append("tts-stop"), earcon=calls.append,
                    glossary=("worktree",), wake=wake, end_words=end_words)
    return pipe, bus, calls, engines


async def collect(queue, count, timeout=2.0):
    return [await asyncio.wait_for(queue.get(), timeout) for _ in range(count)]


def speak_cycle(pipe):
    pipe.trigger("hotkey")
    for frame in (LOUD, LOUD, QUIET, QUIET):
        pipe.feed(frame)


async def stop_task(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
