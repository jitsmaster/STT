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
