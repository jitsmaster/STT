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
    vad_model: str = "models/silero_vad.onnx"
    no_speech_s: float = 5.0
    silence_end_s: float = 0.8
    max_utterance_s: float = 30.0
    min_speech_s: float = 0.25
    glossary: tuple[str, ...] = ()


def load_config(path=None) -> Config:
    data = {}
    if path is not None and Path(path).exists():
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError("unknown config keys: " + ", ".join(sorted(unknown)))
    if "state_dir" in data:
        data["state_dir"] = Path(data["state_dir"])
    if "glossary" in data:
        data["glossary"] = tuple(data["glossary"])
    cfg = Config(**data)
    if cfg.host not in LOOPBACK:
        raise ValueError("host must be a loopback address (127.0.0.1, localhost or ::1)")
    if not 1 <= cfg.port <= 65535:
        raise ValueError("port out of range")
    return cfg
