import numpy as np

from stt.audio import SAMPLE_RATE, save_wav
from stt.bench.wake import count_triggers, wake_report


class FakeStream:
    def accept_waveform(self, rate, samples):
        self.rate = rate

    def input_finished(self):
        pass


class FakeSpotter:
    """Reports hits keyword detections for every stream it is given."""

    def __init__(self, hits):
        self.hits, self.left = hits, hits

    def create_stream(self):
        self.left = self.hits
        return FakeStream()

    def is_ready(self, stream):
        return self.left > 0

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        if self.left > 0:
            self.left -= 1
            return "码农"
        return ""

    def reset_stream(self, stream):
        pass


def test_count_triggers_counts_every_detection():
    assert count_triggers(FakeSpotter(3), np.zeros(SAMPLE_RATE, dtype="float32")) == 3
    assert count_triggers(FakeSpotter(0), np.zeros(SAMPLE_RATE, dtype="float32")) == 0


def test_report_gives_detection_rate_and_false_triggers_per_hour(tmp_path):
    items = [
        {"id": "wake-01", "kind": "wake"}, {"id": "wake-02", "kind": "wake"},
        {"id": "ambient-01", "kind": "ambient"},
    ]
    for item in items:
        save_wav(tmp_path / (item["id"] + ".wav"), np.zeros(SAMPLE_RATE * 36, dtype="float32") + 0.1)
    report = wake_report(FakeSpotter(1), tmp_path, items)
    assert report["total"] == 2 and report["detected"] == 2 and report["rate"] == 1.0
    assert report["false_per_hour"] == 100.0  # 1 trigger in 36 s of ambient audio


def test_model_files_prefer_int8_encoder_and_joiner_with_the_fp32_decoder(tmp_path):
    from stt.bench.wake import model_files

    for name in (
        "encoder-e13-chunk-16-left-64.onnx", "encoder-e13-chunk-16-left-64.int8.onnx",
        "encoder-e13-chunk-8-left-64.int8.onnx", "decoder-e13-chunk-16-left-64.onnx",
        "joiner-e13-chunk-16-left-64.onnx", "joiner-e13-chunk-16-left-64.int8.onnx",
    ):
        (tmp_path / name).write_bytes(b"x")
    files = model_files(tmp_path)
    assert files["encoder"].name == "encoder-e13-chunk-16-left-64.int8.onnx"
    assert files["joiner"].name == "joiner-e13-chunk-16-left-64.int8.onnx"
    assert files["decoder"].name == "decoder-e13-chunk-16-left-64.onnx"
