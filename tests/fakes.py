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
