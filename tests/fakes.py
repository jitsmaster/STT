import numpy as np
from stt.engines.base import Hints, Transcript


class FakeEngine:
    """Returns the queued replies in order and records the hints it was given."""

    def __init__(self, name, replies):
        self.name = name
        self.replies = list(replies)
        self.hints: list[Hints] = []

    def transcribe(self, audio, hints):
        self.hints.append(hints)
        return Transcript(self.replies.pop(0), "en", 0.9, 0.01)


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
