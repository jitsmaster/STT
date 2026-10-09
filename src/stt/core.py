import numpy as np

from .audio import SAMPLE_RATE

EPS = 1e-9


class ListenerCore:
    def __init__(self, endpointer, *, no_speech_s=5.0, silence_end_s=0.8, max_s=30.0, min_speech_s=0.25):
        self.endpointer = endpointer
        self.no_speech_s, self.silence_end_s, self.max_s, self.min_speech_s = no_speech_s, silence_end_s, max_s, min_speech_s
        self.state = "idle"
        self._reset()

    def _reset(self):
        self._frames, self._speech, self._silence, self._waited, self._total = [], 0.0, 0.0, 0.0, 0.0
        self._source, self._context, self._window = "", None, self.no_speech_s

    def trigger(self, source, context=None, window_s=None):
        if self.state == "listening":
            return []
        self._reset()
        self._source, self._context = source, context
        if window_s is not None:
            self._window = window_s
        self.endpointer.reset()
        self.state = "listening"
        return [{"type": "listening", "source": source, "context": context}]

    def _idle(self, reason):
        ev = {"type": "idle", "reason": reason, "source": self._source, "context": self._context}
        self.state = "idle"
        self._reset()
        return [ev]

    def _end(self, reason):
        if self._speech == 0:
            return self._idle("no_speech")
        if self._speech < self.min_speech_s - EPS:
            return self._idle("too_short")
        ev = {"type": "utterance", "audio": np.concatenate(self._frames), "source": self._source,
              "context": self._context, "reason": reason}
        self.state = "idle"
        self._reset()
        return [ev]

    def feed(self, frame):
        if self.state != "listening":
            return []
        dur = len(frame) / SAMPLE_RATE
        self._frames.append(frame)
        self._total += dur
        if self.endpointer.is_speech(frame):
            self._speech += dur
            self._silence = 0.0
        elif self._speech > 0:
            self._silence += dur
        else:
            self._waited += dur
        if self._speech > 0 and self._silence >= self.silence_end_s - EPS:
            return self._end("silence")
        if self._speech == 0 and self._waited >= self._window - EPS:
            return self._end("timeout")
        if self._total >= self.max_s - EPS:
            return self._end("max")
        return []

    def finish(self):
        return self._end("finish") if self.state == "listening" else []

    def cancel(self):
        return self._idle("cancelled") if self.state == "listening" else []
