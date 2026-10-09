import pytest

from stt.config import default_state_dir, load_config


def test_defaults_are_loopback_with_the_wake_word_off():
    cfg = load_config(None)
    assert (cfg.host, cfg.port) == ("127.0.0.1", 8890)
    assert cfg.wake_enabled is False and cfg.dictation_engine is None


def test_toml_overrides(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("port = 9000\ndevice = 'soundcore'\nglossary = ['worktree']\nstate_dir = '" + str(tmp_path) + "'\n", encoding="utf-8")
    cfg = load_config(p)
    assert (cfg.port, cfg.device, cfg.glossary, cfg.state_dir) == (9000, "soundcore", ("worktree",), tmp_path)


def test_unknown_keys_and_non_loopback_hosts_are_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("prot = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)
    p.write_text("host = '0.0.0.0'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="loopback"):
        load_config(p)


def test_state_dir_follows_the_dsh_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("DSH_TTS_STATE_DIR", raising=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert default_state_dir() == tmp_path / "tts"
    monkeypatch.setenv("DSH_TTS_STATE_DIR", str(tmp_path / "x"))
    assert default_state_dir() == tmp_path / "x"
