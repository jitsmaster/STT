import re
import time

from stt.audio import SAMPLE_RATE

from .base import Hints, Transcript

_TAG = re.compile(r"<\|([^|>]*)\|>")


class SenseVoiceEngine:
    name = "sensevoice"

    def __init__(self, model_dir: str, num_threads: int = 4, recognizer=None):
        if recognizer is None:
            import sherpa_onnx

            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_dir + "/model.int8.onnx",
                tokens=model_dir + "/tokens.txt",
                num_threads=num_threads,
                use_itn=True,
                language="auto",
            )
        self._r = recognizer

    def transcribe(self, audio, hints: Hints) -> Transcript:
        started = time.perf_counter()
        stream = self._r.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._r.decode_stream(stream)
        result = stream.result
        # The language is a tag such as <|zh|>: in result.lang, or (older builds) leading result.text.
        found = _TAG.search(getattr(result, "lang", "") or "") or _TAG.search(result.text)
        lang = found.group(1) if found else ""
        return Transcript(_TAG.sub("", result.text).strip(), lang, 0.0, time.perf_counter() - started)
