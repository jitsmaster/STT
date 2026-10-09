import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
SILENCE_RMS = 0.003  # about -50 dBFS; below this an engine is not asked to transcribe


def load_wav(path) -> np.ndarray:
    """Read a WAV as float32 mono at SAMPLE_RATE (stereo is averaged, other rates are resampled)."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.size else np.zeros(0, dtype="float32")
    if rate != SAMPLE_RATE and mono.size:
        n = int(round(mono.size * SAMPLE_RATE / rate))
        mono = np.interp(np.linspace(0, mono.size - 1, n), np.arange(mono.size), mono)
    return mono.astype("float32")


def save_wav(path, audio: np.ndarray) -> None:
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS) -> bool:
    return rms(audio) < threshold
