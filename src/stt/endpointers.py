from typing import Protocol

import numpy as np

from .audio import rms


class Endpointer(Protocol):
    def reset(self) -> None: ...

    def is_speech(self, frame: np.ndarray) -> bool: ...


class EnergyEndpointer:
    """Loudness gate: the fallback when no VAD model is available, and the test endpointer."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def is_speech(self, frame) -> bool:
        return rms(frame) >= self.threshold
