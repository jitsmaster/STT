import threading
import time

import numpy as np
from stt.engines.base import Hints, Transcript


class FakeEngine:
    """Returns the queued replies in order (an Exception reply is raised) and records the hints and audio it was given.
    With a `gate` (a threading.Event) every call blocks until the gate is set."""

    def __init__(self, name, replies, gate: threading.Event | None = None):
        self.name = name
        self.replies = list(replies)
        self.hints: list[Hints] = []
        self.audios: list = []
        self.gate = gate

    def transcribe(self, audio, hints):
        self.hints.append(hints)
        self.audios.append(audio)
        if self.gate is not None:
            self.gate.wait(5)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Transcript(reply, "en", 0.9, 0.01)


class ScriptEndpointer:
    """Speech or silence per frame from a pattern: 'S' is speech, anything else is silence."""

    def __init__(self, pattern):
        self.pattern, self.resets = list(pattern), 0

    def reset(self):
        self.resets += 1

    def is_speech(self, frame):
        return self.pattern.pop(0) == "S" if self.pattern else False


class ScriptStream:
    def __init__(self, spotter):
        self.spotter, self.pending = spotter, []

    def accept_waveform(self, rate, samples):
        self.spotter.accepted += 1
        self.pending.append(samples)


class ScriptSpotter:
    """A frame whose peak reaches the trigger level is the keyword; anything quieter is not."""

    def __init__(self, trigger=0.5):
        self.trigger, self.created, self.resets, self.accepted = trigger, 0, 0, 0

    def create_stream(self):
        self.created += 1
        return ScriptStream(self)

    def is_ready(self, stream):
        return bool(stream.pending)

    def decode_stream(self, stream):
        stream.last = stream.pending.pop(0)

    def get_result(self, stream):
        last, stream.last = getattr(stream, "last", None), None
        return "码农" if last is not None and float(np.max(np.abs(last))) >= self.trigger else ""

    def reset_stream(self, stream):
        self.resets += 1


class FakeSource:
    """A MicSource stand-in: frames are whatever the test pushes into `frames`; start() can be made to fail."""

    def __init__(self, fail=False):
        self.started, self.frames, self.starts, self.stops, self.fail = False, [], 0, 0, fail

    def start(self):
        if self.fail:
            raise OSError("device busy")
        self.started = True
        self.starts += 1

    def stop(self):
        self.started = False
        self.stops += 1

    def next_frame(self, timeout=0.1):
        if self.frames:
            return self.frames.pop(0)
        time.sleep(min(timeout, 0.01))
        return None
