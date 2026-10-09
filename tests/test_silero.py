import dataclasses
import logging

import numpy as np

from stt.config import load_config
from stt.endpointers import EnergyEndpointer, SileroEndpointer, make_endpointer

FRAME = np.zeros(1600, dtype="float32")


class StubVad:
    """One script character per accepted window: 'S' speech, '.' silence. A finished speech run leaves a segment."""

    def __init__(self, script):
        self.script, self.lengths, self.speech, self.segments, self.resets = list(script), [], False, [], 0

    def accept_waveform(self, samples):
        self.lengths.append(len(samples))
        was, self.speech = self.speech, (self.script.pop(0) == "S" if self.script else False)
        if was and not self.speech:
            self.segments.append(object())

    def is_speech_detected(self):
        return self.speech

    def empty(self):
        return not self.segments

    def pop(self):
        self.segments.pop(0)

    def reset(self):
        self.resets += 1
        self.speech = False


def test_frames_reach_the_detector_as_512_sample_windows_with_the_remainder_carried_over():
    vad = StubVad("." * 20)
    ep = SileroEndpointer(vad)
    ep.is_speech(FRAME)
    ep.is_speech(FRAME)
    assert set(vad.lengths) == {512} and len(vad.lengths) == 6  # 3200 samples = 6 windows, 128 samples carried


def test_a_frame_is_speech_when_the_detector_says_so_at_the_end_of_it():
    ep = SileroEndpointer(StubVad("SSS" + "..." + "SSS"))
    assert [ep.is_speech(FRAME) for _ in range(3)] == [True, False, True]


def test_finished_segments_are_drained_so_the_detector_buffer_cannot_fill_up():
    vad = StubVad("SSS...")
    ep = SileroEndpointer(vad)
    ep.is_speech(FRAME)
    ep.is_speech(FRAME)
    assert vad.segments == []


def test_reset_resets_the_detector_and_drops_the_carried_samples():
    vad = StubVad("." * 20)
    ep = SileroEndpointer(vad)
    ep.is_speech(FRAME)  # leaves 64 samples carried
    ep.reset()
    before = len(vad.lengths)
    ep.is_speech(FRAME)
    assert vad.resets == 1 and len(vad.lengths) - before == 3  # 1600 // 512, not 3.2 windows' worth


def test_a_missing_vad_model_falls_back_to_the_loudness_gate_with_a_warning(caplog):
    cfg = dataclasses.replace(load_config(None), vad_model="no/such/model.onnx")
    with caplog.at_level(logging.WARNING, logger="stt"):
        ep = make_endpointer(cfg)
    assert isinstance(ep, EnergyEndpointer)
    assert "VAD model" in caplog.text


def test_an_existing_vad_model_gives_the_silero_endpointer(tmp_path):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"x")
    cfg = dataclasses.replace(load_config(None), vad_model=str(model))
    seen = []
    ep = make_endpointer(cfg, vad_factory=lambda path: seen.append(path) or StubVad("S"))
    assert isinstance(ep, SileroEndpointer) and seen == [str(model)]
