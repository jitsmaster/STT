import re
import time

from stt.audio import SAMPLE_RATE

from .base import Hints, Transcript

_TAG = re.compile(r"<\|([^|>]*)\|>")


class SenseVoiceEngine:
    name = "sensevoice"

    # sherpa-onnx fixes SenseVoice's language when the recognizer is built (DSH's Node worker calls setConfig per recording;
    # the Python API has no such call), so a hint selects one of lazily built recognizers.
    LANGUAGES = ("zh", "en", "ja", "ko", "yue")

    def __init__(self, model_dir: str, num_threads: int = 4, recognizer=None, recognizer_factory=None):
        self._factory = recognizer_factory or (lambda language: self._build(model_dir, num_threads, language))
        self._recognizers = {"auto": recognizer} if recognizer is not None else {}

    @staticmethod
    def _build(model_dir: str, num_threads: int, language: str):
        import sherpa_onnx

        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_dir + "/model.int8.onnx",
            tokens=model_dir + "/tokens.txt",
            num_threads=num_threads,
            use_itn=True,
            language=language,
        )

    def _recognizer(self, language):
        key = language if language in self.LANGUAGES else "auto"
        if key not in self._recognizers:
            self._recognizers[key] = self._factory(key)
        return self._recognizers[key]

    def transcribe(self, audio, hints: Hints) -> Transcript:
        started = time.perf_counter()
        recognizer = self._recognizer(hints.language)
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        result = stream.result
        # The language is a tag such as <|zh|>: in result.lang, or (older builds) leading result.text.
        found = _TAG.search(getattr(result, "lang", "") or "") or _TAG.search(result.text)
        lang = found.group(1) if found else ""
        return Transcript(_TAG.sub("", result.text).strip(), lang, 0.0, time.perf_counter() - started)
