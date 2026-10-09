import asyncio
import uuid
from collections import OrderedDict

from .audio import is_silent
from .engines.base import Hints, Transcript


class UnknownUtterance(LookupError):
    """The id is not (or no longer) among the kept utterances; a LookupError so callers can tell it from engine errors."""


class Bus:
    """Fans events out to subscribers; a subscriber that stops reading loses events instead of blocking."""

    def __init__(self):
        self._subs = set()

    def subscribe(self):
        q = asyncio.Queue(maxsize=100)
        self._subs.add(q)
        return q

    def unsubscribe(self, q):
        self._subs.discard(q)

    def publish(self, event):
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


class Pipeline:
    def __init__(self, core, engines, bus, *, tts_stop=lambda: None, earcon=lambda kind: None,
                 glossary=(), wake=None, keep=5):
        self.core, self.engines, self.bus = core, engines, bus
        self.tts_stop, self.earcon, self.glossary, self.wake, self.keep = tts_stop, earcon, tuple(glossary), wake, keep
        self._utterances = OrderedDict()  # uid -> (audio, context, source)
        self._lock = asyncio.Lock()
        self._tasks = set()

    @property
    def state(self):
        return self.core.state

    @property
    def context(self):
        """The context of the current listening window (None when idle or when the window has none)."""
        return self.core.context

    def _idle(self, reason, context):
        self.bus.publish({"type": "idle", "reason": reason, "context": context})

    def _handle(self, events):
        for ev in events:
            if ev["type"] == "listening":
                self.tts_stop()
                self.earcon("start")
                self.bus.publish({"type": "listening", "source": ev["source"], "context": ev["context"]})
            elif ev["type"] == "idle":
                self.earcon("cancel")
                self.wake and self.wake.reset()
                self._idle(ev["reason"], ev["context"])
            elif ev["type"] == "utterance":
                self.earcon("end")
                self.wake and self.wake.reset()
                self.bus.publish({"type": "transcribing", "context": ev["context"]})
                task = asyncio.get_running_loop().create_task(self._transcribe(ev))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def trigger(self, source, context=None, window_s=None) -> bool:
        events = self.core.trigger(source, context, window_s)
        self._handle(events)
        return bool(events)

    def finish(self) -> bool:
        events = self.core.finish()
        self._handle(events)
        return bool(events)

    def cancel(self) -> bool:
        events = self.core.cancel()
        self._handle(events)
        return bool(events)

    def feed(self, frame):
        if self.core.state == "listening":
            self._handle(self.core.feed(frame))
        elif self.wake is not None and self.wake.feed(frame):
            self.trigger("wake")

    async def _decode(self, kind, audio, language):
        engine = self.engines.get(kind) or self.engines["command"]
        if is_silent(audio):
            return Transcript("", "", 0.0, 0.0), engine.name
        async with self._lock:
            result = await asyncio.to_thread(engine.transcribe, audio, Hints(language=language, glossary=self.glossary))
        return result, engine.name

    @staticmethod
    def _event(uid, result, engine, source, context, revision):
        return {"type": "transcript", "id": uid, "text": result.text, "language": result.language,
                "confidence": result.confidence, "engine": engine, "seconds": result.seconds,
                "source": source, "context": context, "revision": revision}

    async def _transcribe(self, ev):
        uid = uuid.uuid4().hex[:8]
        self._utterances[uid] = (ev["audio"], ev["context"], ev["source"])
        while len(self._utterances) > self.keep:
            self._utterances.popitem(last=False)
        try:
            result, engine = await self._decode("command", ev["audio"], None)
        except Exception as exc:
            self.bus.publish({"type": "error", "message": "transcription failed: " + str(exc), "context": ev["context"]})
            self._idle("error", ev["context"])
            return
        self.bus.publish(self._event(uid, result, engine, ev["source"], ev["context"], 1))
        self._idle("done", ev["context"])

    async def transcribe_audio(self, audio, kind="command", language=None) -> dict:
        result, engine = await self._decode(kind, audio, language)
        return {"text": result.text, "language": result.language, "confidence": result.confidence,
                "engine": engine, "seconds": result.seconds}

    async def retranscribe(self, uid, kind="dictation", language=None) -> dict:
        if uid not in self._utterances:
            raise UnknownUtterance(uid)
        audio, context, source = self._utterances[uid]
        result, engine = await self._decode(kind, audio, language)
        event = self._event(uid, result, engine, source, context, 2)
        self.bus.publish(event)
        return event
