from types import SimpleNamespace

import numpy as np

from stt.engines.base import Hints
from stt.engines.sensevoice import SenseVoiceEngine


class StubStream:
    def __init__(self):
        self.result = SimpleNamespace(text="<|zh|><|NEUTRAL|><|Speech|><|withitn|>把 PR 6512 打开", lang="<|zh|>")

    def accept_waveform(self, rate, samples):
        self.rate, self.samples = rate, samples


class StubRecognizer:
    def create_stream(self):
        self.stream = StubStream()
        return self.stream

    def decode_stream(self, stream):
        pass


def test_strips_tags_and_reports_the_language():
    engine = SenseVoiceEngine("unused", recognizer=StubRecognizer())
    out = engine.transcribe(np.zeros(16000, dtype="float32"), Hints(language="en", glossary=("x",)))
    assert out.text == "把 PR 6512 打开"
    assert out.language == "zh"
    assert engine._r.stream.rate == 16000


def test_language_comes_from_result_lang_when_the_text_has_no_tags():
    """The real sherpa-onnx 1.13.8 result has clean text and lang='<|en|>'."""
    rec = StubRecognizer()
    rec.create_stream()
    rec.create_stream = lambda: SimpleNamespace(
        result=SimpleNamespace(text="open five", lang="<|en|>"), accept_waveform=lambda r, s: None
    )
    out = SenseVoiceEngine("unused", recognizer=rec).transcribe(np.zeros(1, dtype="float32"), Hints())
    assert (out.text, out.language) == ("open five", "en")
