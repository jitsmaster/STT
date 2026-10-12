import math
import time

from stt.textnorm import to_simplified

from .base import Hints, Transcript

GLOSSARY_MIN_SECONDS = 2.0  # a shorter clip gets no prompt: small answers a one-word command with the prompt itself
NO_SPEECH_PROB = 0.6  # Whisper's own rule: a segment above both thresholds is silence it made words up for
MIN_AVG_LOGPROB = -1.0


def _is_hallucination(segment) -> bool:
    nsp, logprob = getattr(segment, "no_speech_prob", None), getattr(segment, "avg_logprob", None)
    return nsp is not None and logprob is not None and nsp > NO_SPEECH_PROB and logprob < MIN_AVG_LOGPROB


def _confidence(segments, language_probability: float) -> float:
    """exp(mean avg_logprob) x (1 - mean no_speech_prob) over the kept segments; the language probability only when
    the model reports no segment statistics (it says nothing about how well the words were recognised)."""
    stats = [(s.avg_logprob, getattr(s, "no_speech_prob", 0.0) or 0.0) for s in segments if getattr(s, "avg_logprob", None) is not None]
    if not segments:
        return 0.0
    if not stats:
        return float(language_probability)
    logprob = sum(p for p, _ in stats) / len(stats)
    no_speech = sum(n for _, n in stats) / len(stats)
    return max(0.0, min(1.0, math.exp(logprob) * (1.0 - no_speech)))


class FasterWhisperEngine:
    def __init__(self, model_name: str = "large-v3", compute_type: str = "int8", threads: int = 8, model=None):
        self.name = f"faster-whisper:{model_name}"
        if model is None:
            from faster_whisper import WhisperModel  # imported lazily: loading it is slow

            model = WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=threads)
        self._model = model

    def transcribe(self, audio, hints: Hints) -> Transcript:
        started = time.perf_counter()
        kwargs = {"beam_size": 5, "vad_filter": False, "condition_on_previous_text": False}
        if hints.language:
            kwargs["language"] = hints.language
        if hints.glossary and len(audio) >= GLOSSARY_MIN_SECONDS * 16000:
            kwargs["initial_prompt"] = ", ".join(hints.glossary)
        segments, info = self._model.transcribe(audio, **kwargs)
        kept = [s for s in segments if not _is_hallucination(s)]
        text = to_simplified(" ".join(s.text.strip() for s in kept).strip())
        confidence = _confidence(kept, info.language_probability)
        return Transcript(text, info.language, confidence, time.perf_counter() - started)
