import dataclasses
import math
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def default_state_dir() -> Path:
    env = os.environ.get("DSH_TTS_STATE_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh") / "tts"


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8890
    state_dir: Path = field(default_factory=default_state_dir)
    device: str | int | None = None
    hotkey: str = "ctrl+alt+v"
    command_engine: str = "faster-whisper:large-v3"
    dictation_engine: str | None = None
    wake_enabled: bool = False
    wake_model_dir: str = "models/kws"
    wake_keywords: str = "keywords.txt"
    wake_threshold: float = 0.25
    wake_score: float = 1.5
    vad: str = "silero"  # "energy" opts in to the weaker loudness gate; otherwise a missing vad_model stops startup
    vad_model: str = "models/silero_vad.onnx"
    no_speech_s: float = 5.0
    silence_end_s: float = 0.8
    max_utterance_s: float = 30.0
    min_speech_s: float = 0.25
    glossary: tuple[str, ...] = ()


_STR_FIELDS = ("host", "hotkey", "command_engine", "wake_model_dir", "wake_keywords", "vad_model")
_POSITIVE_FIELDS = ("wake_threshold", "wake_score", "no_speech_s", "silence_end_s", "max_utterance_s", "min_speech_s")
_PATH_FIELDS = ("vad_model", "wake_model_dir", "wake_keywords")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(cfg: Config) -> Config:
    """Raise ValueError for a wrong type or an out-of-range value (TOML is untyped: "false" is a truthy string)."""
    for name in _STR_FIELDS:
        if not isinstance(getattr(cfg, name), str) or not getattr(cfg, name).strip():
            raise ValueError(name + " must be a non-empty string")
    if cfg.host not in LOOPBACK:
        raise ValueError("host must be a loopback address (127.0.0.1, localhost or ::1)")
    if isinstance(cfg.port, bool) or not isinstance(cfg.port, int) or not 1 <= cfg.port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    if not isinstance(cfg.state_dir, Path) or str(cfg.state_dir) in ("", "."):
        raise ValueError("state_dir must be a non-empty path")
    if cfg.device is not None and (isinstance(cfg.device, bool) or not isinstance(cfg.device, (int, str))):
        raise ValueError("device must be an index or a name fragment")
    if cfg.dictation_engine is not None and (not isinstance(cfg.dictation_engine, str) or not cfg.dictation_engine.strip()):
        raise ValueError("dictation_engine must be a non-empty string")
    if not isinstance(cfg.wake_enabled, bool):
        raise ValueError("wake_enabled must be true or false")
    if cfg.vad not in ("silero", "energy"):
        raise ValueError('vad must be "silero" or "energy"')
    for name in _POSITIVE_FIELDS:
        if not _number(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(name + " must be a positive number")
    if not isinstance(cfg.glossary, tuple) or not all(isinstance(t, str) for t in cfg.glossary):
        raise ValueError("glossary must be a list of strings")
    return cfg


def _absolute(value, base: Path):
    return value if not isinstance(value, str) or Path(value).is_absolute() else str(base / value)


def _resolve_paths(cfg: Config, base: Path) -> Config:
    """Model paths in a config file are relative to that file, so a daemon started from another working directory
    (for example by a plugin) finds them."""
    changes = {name: _absolute(getattr(cfg, name), base) for name in _PATH_FIELDS}
    for name in ("command_engine", "dictation_engine"):
        spec = getattr(cfg, name)
        kind, sep, arg = (spec or "").partition(":")
        if kind == "sensevoice" and sep and arg:
            changes[name] = kind + ":" + _absolute(arg, base)
    return dataclasses.replace(cfg, **changes)


def load_config(path=None, *, required: bool = False) -> Config:
    data = {}
    base = None
    if path is not None and Path(path).exists():
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        base = Path(path).resolve().parent
    elif required and path is not None:
        raise FileNotFoundError("config file not found: " + str(path))
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError("unknown config keys: " + ", ".join(sorted(unknown)))
    if isinstance(data.get("state_dir"), str) and data["state_dir"].strip():
        data["state_dir"] = Path(data["state_dir"])
    if isinstance(data.get("glossary"), list):
        data["glossary"] = tuple(data["glossary"])
    cfg = validate(Config(**data))
    return _resolve_paths(cfg, base) if base is not None else cfg
