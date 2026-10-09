from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Hints:
    language: str | None = None  # "en" or "zh" to force a language; None = auto-detect
    glossary: tuple[str, ...] = ()  # terms the engine may bias towards (ignored if unsupported)


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str  # "en", "zh", ... or "" when unknown
    confidence: float  # 0.0-1.0; 0.0 when the engine reports none
    seconds: float = 0.0  # wall time spent decoding


class Engine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, hints: Hints) -> Transcript: ...
