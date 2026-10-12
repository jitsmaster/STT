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


@pytest.mark.parametrize("line", [
    'wake_enabled = "false"', 'glossary = "worktree"', 'max_utterance_s = "30"', 'port = "8890"', "port = true",
    "port = 8890.5", "hotkey = 5", "command_engine = 3", "device = true", "silence_end_s = 0", "no_speech_s = -1",
    "min_speech_s = 0", "state_dir = ''", "vad = 'nope'", "glossary = [1]", "wake_threshold = 0", "max_utterance_s = nan",
])
def test_wrong_types_and_out_of_range_values_are_rejected(tmp_path, line):
    p = tmp_path / "c.toml"
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_a_missing_explicit_config_file_is_an_error_not_the_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml", required=True)
    assert load_config(tmp_path / "nope.toml").port == 8890


def test_replacing_the_port_is_validated_again():
    import dataclasses

    from stt.config import validate

    with pytest.raises(ValueError):
        validate(dataclasses.replace(load_config(None), port=70000))


def test_relative_model_paths_resolve_against_the_config_file_not_the_working_directory(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("vad_model = 'models/v.onnx'\nwake_model_dir = 'k'\ncommand_engine = 'sensevoice:models/sv'\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.vad_model == str(tmp_path / "models" / "v.onnx")
    assert cfg.wake_model_dir == str(tmp_path / "k")
    assert cfg.command_engine == "sensevoice:" + str(tmp_path / "models" / "sv")
    assert load_config(None).command_engine == "faster-whisper:small"
