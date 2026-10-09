import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE, rms


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


log = logging.getLogger("stt")


class SileroEndpointer:
    """Speech/non-speech per frame from a sherpa-onnx Silero VoiceActivityDetector."""

    def __init__(self, detector, window: int = 512):
        self.detector, self.window = detector, window
        self._carry = np.zeros(0, dtype="float32")

    def reset(self) -> None:
        self.detector.reset()
        self._carry = np.zeros(0, dtype="float32")

    def is_speech(self, frame) -> bool:
        data = np.concatenate([self._carry, frame])
        whole = len(data) - len(data) % self.window
        for start in range(0, whole, self.window):
            self.detector.accept_waveform(data[start : start + self.window])
        self._carry = data[whole:]
        while not self.detector.empty():  # we only need the speech flag, not the segments
            self.detector.pop()
        return bool(self.detector.is_speech_detected())


def make_vad(model_path: str, buffer_s: float = 30, min_silence_s: float = 0.25):
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = model_path
    config.silero_vad.min_silence_duration = min_silence_s
    config.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=buffer_s)


def make_endpointer(config, *, vad_factory=None):
    if config.vad == "energy":
        log.warning("vad = energy: using the loudness gate, which is worse in noise")
        return EnergyEndpointer()
    model = Path(config.vad_model)
    if not model.is_file():
        # A silent fallback would turn fan noise or a call into "utterances" that an engine then makes words of.
        raise FileNotFoundError(
            f"VAD model {model} not found: download silero_vad.onnx (see README) or set vad = \"energy\" in the config "
            "to accept the weaker loudness gate"
        )
    return SileroEndpointer((vad_factory or make_vad)(str(model)))
