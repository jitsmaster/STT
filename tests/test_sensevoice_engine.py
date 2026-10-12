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
    engine = SenseVoiceEngine("unused", recognizer_factory=lambda language: StubRecognizer())
    out = engine.transcribe(np.zeros(16000, dtype="float32"), Hints(language="en", glossary=("x",)))
    assert out.text == "把 PR 6512 打开"
    assert out.language == "zh"
    assert engine._recognizers["en"].stream.rate == 16000


def test_language_comes_from_result_lang_when_the_text_has_no_tags():
    """The real sherpa-onnx 1.13.8 result has clean text and lang='<|en|>'."""
    rec = StubRecognizer()
    rec.create_stream()
    rec.create_stream = lambda: SimpleNamespace(
        result=SimpleNamespace(text="open five", lang="<|en|>"), accept_waveform=lambda r, s: None
    )
    out = SenseVoiceEngine("unused", recognizer=rec).transcribe(np.zeros(1, dtype="float32"), Hints())
    assert (out.text, out.language) == ("open five", "en")


def test_a_language_hint_selects_a_recognizer_built_for_that_language():
    """sherpa-onnx fixes SenseVoice's language when the recognizer is built (DSH sets it per recording), so the hint picks one."""
    built = []

    def factory(language):
        built.append(language)
        rec = StubRecognizer()
        rec.language = language
        return rec

    engine = SenseVoiceEngine("unused", recognizer_factory=factory)
    engine.transcribe(np.zeros(1600, dtype="float32"), Hints(language="en"))
    engine.transcribe(np.zeros(1600, dtype="float32"), Hints(language="zh"))
    engine.transcribe(np.zeros(1600, dtype="float32"), Hints(language="en"))
    engine.transcribe(np.zeros(1600, dtype="float32"), Hints())
    assert built == ["en", "zh", "auto"]  # lazily, once each; no hint means auto


def test_an_unsupported_hint_falls_back_to_auto():
    built = []
    engine = SenseVoiceEngine("unused", recognizer_factory=lambda language: built.append(language) or StubRecognizer())
    engine.transcribe(np.zeros(1600, dtype="float32"), Hints(language="fr"))
    assert built == ["auto"]
