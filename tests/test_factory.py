import pytest
from fakes import FakeEngine

import stt.engines.factory as factory
from stt.config import load_config


def test_unknown_engine_spec_raises():
    with pytest.raises(ValueError, match="unknown engine"):
        factory.make_engine("nope:1")


def test_one_engine_serves_both_roles_unless_a_dictation_engine_is_set(monkeypatch):
    monkeypatch.setattr(factory, "make_engine", lambda spec: FakeEngine(spec, []))
    engines = factory.build_engines(load_config(None))
    assert engines["command"] is engines["dictation"]
    cfg = load_config(None).__class__(dictation_engine="sensevoice:models/sensevoice")
    split = factory.build_engines(cfg)
    assert split["command"].name != split["dictation"].name
