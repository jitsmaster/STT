import asyncio

import numpy as np
from fakes import FakeEngine

from stt.core import ListenerCore
from stt.endpointers import EnergyEndpointer
from stt.pipeline import Bus, Pipeline

LOUD = np.full(1600, 0.2, dtype="float32")
QUIET = np.zeros(1600, dtype="float32")
WAKE = np.full(1600, 0.9, dtype="float32")


def build(command_replies=("status",), dictation_replies=("status please",), wake=None):
    core = ListenerCore(EnergyEndpointer(0.05), no_speech_s=0.3, silence_end_s=0.2, max_s=5.0, min_speech_s=0.1)
    engines = {"command": FakeEngine("fast", command_replies), "dictation": FakeEngine("slow", dictation_replies)}
    bus, calls = Bus(), []
    pipe = Pipeline(core, engines, bus, tts_stop=lambda: calls.append("tts-stop"), earcon=calls.append,
                    glossary=("worktree",), wake=wake)
    return pipe, bus, calls, engines


async def collect(queue, count, timeout=2.0):
    return [await asyncio.wait_for(queue.get(), timeout) for _ in range(count)]


def speak_cycle(pipe):
    pipe.trigger("hotkey")
    for frame in (LOUD, LOUD, QUIET, QUIET):
        pipe.feed(frame)
