import numpy as np
import soundfile as sf

from stt.audio import SAMPLE_RATE, is_silent, load_wav, save_wav


def tone(seconds, rate=SAMPLE_RATE, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 440 * t)).astype("float32")


def test_roundtrip_keeps_length_and_dtype(tmp_path):
    p = tmp_path / "a.wav"
    save_wav(p, tone(1.0))
    a = load_wav(p)
    assert a.dtype == np.float32
    assert abs(len(a) - SAMPLE_RATE) <= 1


def test_resamples_to_16k_and_mixes_stereo_to_mono(tmp_path):
    p = tmp_path / "b.wav"
    sf.write(str(p), np.stack([tone(1.0, 44100), tone(1.0, 44100)], axis=1), 44100)
    a = load_wav(p)
    assert a.ndim == 1
    assert abs(len(a) - SAMPLE_RATE) <= 1


def test_empty_file_loads_as_empty_and_counts_as_silent(tmp_path):
    p = tmp_path / "c.wav"
    sf.write(str(p), np.zeros(0, dtype="float32"), SAMPLE_RATE)
    a = load_wav(p)
    assert a.size == 0
    assert is_silent(a)


def test_quiet_noise_is_silent_but_speech_level_is_not():
    assert is_silent(tone(1.0, amp=0.0005))
    assert not is_silent(tone(1.0, amp=0.1))
