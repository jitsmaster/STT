import numpy as np
from fakes import FakeEngine

from stt.audio import SAMPLE_RATE, save_wav
from stt.bench.prompts import save_manifest
from stt.bench.run import render_markdown, run_bench, summarize


def speechlike(seconds=1.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 300 * t)).astype("float32")


ITEMS = [
    {"id": "cmd-en-10", "kind": "command", "lang": "en", "text": "option two", "intent": "option:2", "terms": [], "number": 2},
    {"id": "dict-01", "kind": "dictation", "lang": "en", "text": "rebase the worktree", "intent": None, "terms": ["worktree"], "number": None},
    {"id": "mix-01", "kind": "mixed", "lang": "mixed", "text": "把 PR 6512 打开", "intent": None, "terms": ["PR"], "number": None},
]


def make_samples(tmp_path, silent=()):
    for item in ITEMS:
        quiet = np.zeros(SAMPLE_RATE, dtype="float32")
        save_wav(tmp_path / (item["id"] + ".wav"), quiet if item["id"] in silent else speechlike())
    save_manifest(ITEMS, tmp_path / "manifest.json")


def test_scores_each_row_and_summarizes_per_kind(tmp_path):
    make_samples(tmp_path)
    engine = FakeEngine("fake", ["option 2", "rebase the work tree", "把 pr 6512 打开"])
    rows = run_bench(ITEMS, tmp_path, engine, glossary=[], oracle_lang=False)
    assert rows[0].intent_ok is True and rows[0].number_ok is True
    assert rows[1].ter > 0 and rows[1].term_rate == 1.0
    summary = summarize(rows)
    assert summary["command"]["intent"] == 1.0
    assert summary["dictation"]["n"] == 1


def test_silent_audio_is_not_sent_to_the_engine_and_scores_as_empty(tmp_path):
    make_samples(tmp_path, silent={"dict-01"})
    engine = FakeEngine("fake", ["option 2", "把 pr 6512 打开"])
    rows = run_bench(ITEMS, tmp_path, engine, glossary=[], oracle_lang=False)
    silent_row = next(r for r in rows if r.id == "dict-01")
    assert silent_row.got == "" and silent_row.ter == 1.0
    assert len(engine.hints) == 2


def test_oracle_language_and_glossary_reach_the_engine(tmp_path):
    make_samples(tmp_path)
    engine = FakeEngine("fake", ["a", "b", "c"])
    run_bench(ITEMS, tmp_path, engine, glossary=["worktree"], oracle_lang=True)
    assert engine.hints[0].language == "en" and engine.hints[0].glossary == ("worktree",)
    assert engine.hints[2].language is None  # mixed is never forced


def test_missing_samples_are_skipped_and_markdown_names_each_engine(tmp_path):
    save_manifest(ITEMS, tmp_path / "manifest.json")
    assert run_bench(ITEMS, tmp_path, FakeEngine("fake", []), [], False) == []
    md = render_markdown({"eng-a": {"command": {"n": 1, "ter": 0.1, "terms": None, "number": 1.0, "intent": 1.0, "median_seconds": 0.5}}})
    assert "eng-a" in md and "command" in md
