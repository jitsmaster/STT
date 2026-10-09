from types import SimpleNamespace

import numpy as np

from stt.engines.base import Hints
from stt.engines.whisper import FasterWhisperEngine


class StubModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        segments = [SimpleNamespace(text=" Rebase the "), SimpleNamespace(text="worktree. ")]
        return iter(segments), SimpleNamespace(language="en", language_probability=0.93)


def test_joins_segments_and_reports_language_and_probability():
    engine = FasterWhisperEngine(model=StubModel())
    out = engine.transcribe(np.zeros(16000, dtype="float32"), Hints())
    assert out.text == "Rebase the worktree."
    assert (out.language, out.confidence) == ("en", 0.93)


def test_language_and_glossary_are_forwarded_and_auto_leaves_language_unset():
    stub = StubModel()
    engine = FasterWhisperEngine(model=stub)
    engine.transcribe(np.zeros(1, dtype="float32"), Hints(language="zh", glossary=("worktree", "PR 6512")))
    assert stub.kwargs["language"] == "zh"
    assert stub.kwargs["initial_prompt"] == "worktree, PR 6512"
    engine.transcribe(np.zeros(1, dtype="float32"), Hints())
    assert "language" not in stub.kwargs and "initial_prompt" not in stub.kwargs
    assert stub.kwargs["condition_on_previous_text"] is False


def seg(text, no_speech=0.0, logprob=-0.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob)


class SegModel:
    def __init__(self, segments, language="en"):
        self.segments, self.language = segments, language

    def transcribe(self, audio, **kwargs):
        return iter(self.segments), SimpleNamespace(language=self.language, language_probability=0.99)


def test_segments_that_whisper_itself_thinks_are_not_speech_are_dropped():
    segments = [seg(" open five", 0.01, -0.2), seg(" Thank you.", 0.9, -1.5), seg(" You", 0.7, -0.3)]
    out = FasterWhisperEngine(model=SegModel(segments)).transcribe(np.zeros(1, dtype="float32"), Hints())
    assert out.text == "open five You"
    out = FasterWhisperEngine(model=SegModel([seg(" Thank you.", 0.9, -1.5)])).transcribe(np.zeros(1, dtype="float32"), Hints())
    assert (out.text, out.confidence) == ("", 0.0)


def test_confidence_reflects_the_transcript_not_the_language_guess():
    sure = FasterWhisperEngine(model=SegModel([seg("a", 0.0, -0.05)])).transcribe(np.zeros(1, dtype="float32"), Hints())
    shaky = FasterWhisperEngine(model=SegModel([seg("a", 0.0, -0.9)])).transcribe(np.zeros(1, dtype="float32"), Hints())
    assert sure.confidence > 0.9 and shaky.confidence < 0.5


def test_traditional_chinese_output_is_converted_to_simplified():
    out = FasterWhisperEngine(model=SegModel([seg("\u958b\u653e \u72c0\u614b PR 6512")], "zh")).transcribe(np.zeros(1, dtype="float32"), Hints())
    assert out.text == "\u5f00\u653e \u72b6\u6001 PR 6512"
