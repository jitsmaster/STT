import numpy as np

from .audio import SAMPLE_RATE

EPS = 1e-9


class ListenerCore:
    PRE_ROLL_S = 0.3  # leading silence kept in front of the first speech frame
    END_MIN_S, END_MAX_S = 0.15, 1.5  # speech length of a burst that may be a spoken end word

    def __init__(self, endpointer, *, no_speech_s=5.0, silence_end_s=0.8, max_s=30.0, min_speech_s=0.25,
                 end_check=False, end_gap_s=0.6, end_lead_gap_s=0.4):
        self.endpointer = endpointer
        self.no_speech_s, self.silence_end_s, self.max_s, self.min_speech_s = no_speech_s, silence_end_s, max_s, min_speech_s
        self.end_check, self.end_gap_s, self.end_lead_gap_s = end_check, end_gap_s, end_lead_gap_s
        self._burst_seq = 0  # ids keep counting across windows, so a stale id can never match a later window
        self.state = "idle"
        self._reset()

    def _reset(self):
        self._frames, self._speech, self._silence, self._waited, self._total = [], 0.0, 0.0, 0.0, 0.0
        self._lead = 0.0
        self._burst = None  # the latest speech burst: {"id", "start", "end", "speech", "checked"}; start/end index _frames
        self._source, self._context, self._window = "", None, self.no_speech_s

    @property
    def context(self):
        return self._context

    def _trimmed(self):
        """Drop the leading silence beyond PRE_ROLL_S: up to 5 s of waiting would dilute quiet speech in the
        loudness gate and give the engine a long silent head."""
        drop, frames = self._lead - self.PRE_ROLL_S, self._frames
        while len(frames) > 1 and drop >= len(frames[0]) / SAMPLE_RATE - EPS:
            drop -= len(frames[0]) / SAMPLE_RATE
            frames = frames[1:]
        return np.concatenate(frames)

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
        ev = {"type": "utterance", "audio": self._trimmed(), "source": self._source,
              "context": self._context, "reason": reason}
        self.state = "idle"
        self._reset()
        return [ev]

    def _track_burst(self, dur, gap):
        """A speech frame joins the current burst unless it follows a pause of end_lead_gap_s or more (or is the first)."""
        index = len(self._frames) - 1
        if self._burst is None or gap >= self.end_lead_gap_s - EPS:
            self._burst_seq += 1
            self._burst = {"id": self._burst_seq, "start": index, "speech": 0.0, "checked": False}
        self._burst["end"] = index + 1
        self._burst["speech"] += dur

    def _end_check(self):
        """The burst audio (with a short pre-roll) once the burst is followed by a long enough pause, at most once per
        burst and only when it is short enough to be a single spoken word."""
        burst = self._burst
        if burst is None or burst["checked"] or self._silence < max(self.end_gap_s, self.end_lead_gap_s) - EPS:
            return []
        burst["checked"] = True
        if not self.END_MIN_S - EPS <= burst["speech"] <= self.END_MAX_S + EPS:
            return []
        first, pre = burst["start"], 0.0
        while first > 0 and pre < self.PRE_ROLL_S - EPS:
            first -= 1
            pre += len(self._frames[first]) / SAMPLE_RATE
        return [{"type": "endcheck", "audio": np.concatenate(self._frames[first : burst["end"]]), "burst": burst["id"]}]

    def apply_end_word(self, burst_id):
        """The burst was the end word: drop it (and the silence after it) and end the utterance with what came before.
        Ignored (returns []) when the window moved on: another burst began, or the window is no longer listening."""
        burst = self._burst
        if self.state != "listening" or burst is None or burst["id"] != burst_id or not burst["checked"]:
            return []
        del self._frames[burst["start"] :]
        self._speech -= burst["speech"]
        if self._speech < EPS:
            self._speech = 0.0
        return self._end("endword")

    def feed(self, frame):
        if self.state != "listening":
            return []
        dur = len(frame) / SAMPLE_RATE
        self._frames.append(frame)
        self._total += dur
        if self.endpointer.is_speech(frame):
            if self._speech == 0:
                self._lead = self._total - dur
            if self.end_check:
                self._track_burst(dur, self._silence)
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
        return self._end_check() if self.end_check and self._speech > 0 else []

    def finish(self):
        return self._end("finish") if self.state == "listening" else []

    def cancel(self):
        return self._idle("cancelled") if self.state == "listening" else []
