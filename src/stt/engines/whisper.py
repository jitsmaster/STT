import time

from .base import Hints, Transcript


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
        if hints.glossary:
            kwargs["initial_prompt"] = ", ".join(hints.glossary)
        segments, info = self._model.transcribe(audio, **kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, info.language, float(info.language_probability), time.perf_counter() - started)
