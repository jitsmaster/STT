import io
import sys

from stt.bench import record


def test_list_devices_survives_a_cp1252_console(monkeypatch):
    """Device names (and the Chinese prompts) are not encodable in cp1252, the default for a redirected stdout."""
    monkeypatch.setattr(record.sd, "query_devices", lambda: "Microphone (Realtek\u00ae \u9ea6\u514b\u98ce)")
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert record.main(["--list-devices"]) == 0
    sys.stdout.flush()
    assert "Microphone" in raw.getvalue().decode("utf-8")


def tone(seconds=0.5, amp=0.1):
    np = __import__("numpy")
    return (amp * np.sin(np.linspace(0, 200, int(16000 * seconds)))).astype("float32")


DEVICES = [
    {"name": "Speakers", "max_input_channels": 0},
    {"name": "Headset (soundcore Space 2)", "max_input_channels": 1},
]


def test_an_output_only_device_is_refused_with_the_list_of_inputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(record.sd, "query_devices", lambda: DEVICES)
    assert record.main(["--out", str(tmp_path), "--device", "0"]) == 1
    err = capsys.readouterr().err
    assert "not an input device" in err and "soundcore" in err


def test_a_device_name_fragment_is_resolved_to_the_input_index(monkeypatch, tmp_path):
    monkeypatch.setattr(record.sd, "query_devices", lambda: DEVICES)
    seen = []
    monkeypatch.setattr(record, "record_until_enter", lambda device: seen.append(device) or tone())
    monkeypatch.setattr(record, "record_seconds", lambda seconds, device: seen.append(device) or __import__("numpy").zeros(1600, dtype="float32"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(record, "missing_samples", lambda items, out: items[:1])
    monkeypatch.setattr(record, "check_opens", lambda device: None)
    assert record.main(["--out", str(tmp_path), "--device", "SoundCore"]) == 0
    assert seen == [1]


def test_a_device_that_cannot_be_opened_is_reported_before_any_prompt(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(record.sd, "query_devices", lambda: DEVICES)

    def boom(device):
        raise record.sd.PortAudioError("Invalid device [PaErrorCode -9996]")

    monkeypatch.setattr(record, "check_opens", boom)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(AssertionError("prompted")))
    assert record.main(["--out", str(tmp_path), "--device", "soundcore"]) == 1
    err = capsys.readouterr().err
    assert "cannot be opened" in err and "Invalid device" in err and "--list-devices" in err


class FakeStream:
    """Stands in for sd.InputStream; delivers `frames` sample frames to the callback on enter."""

    def __init__(self, frames):
        self.frames = frames

    def __call__(self, **kwargs):
        self.callback = kwargs.get("callback")
        return self

    def __enter__(self):
        if self.frames and self.callback:
            self.callback(__import__("numpy").zeros((self.frames, 1), dtype="float32"), self.frames, None, None)
        return self

    def __exit__(self, *exc):
        return False


def test_a_device_that_opens_but_delivers_no_audio_is_refused(monkeypatch):
    monkeypatch.setattr(record.sd, "InputStream", FakeStream(0))
    monkeypatch.setattr(record.time, "sleep", lambda s: None)
    try:
        record.check_opens(1)
    except record.sd.PortAudioError as exc:
        assert "no audio" in str(exc)
    else:
        raise AssertionError("a silent stream was accepted")


def test_a_device_that_delivers_audio_passes_the_check(monkeypatch):
    monkeypatch.setattr(record.sd, "InputStream", FakeStream(800))
    monkeypatch.setattr(record.time, "sleep", lambda s: None)
    record.check_opens(1)


def run_with_takes(monkeypatch, tmp_path, takes):
    monkeypatch.setattr(record.sd, "query_devices", lambda: DEVICES)
    takes = iter(takes)
    monkeypatch.setattr(record, "record_until_enter", lambda device: next(takes))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(record, "missing_samples", lambda items, out: items[:1])
    monkeypatch.setattr(record, "check_opens", lambda device: None)
    return record.main(["--out", str(tmp_path), "--device", "soundcore"])


def test_a_clip_with_no_audio_is_not_saved_and_is_asked_again(monkeypatch, tmp_path, capsys):
    np = __import__("numpy")
    assert run_with_takes(monkeypatch, tmp_path, [np.zeros(0, dtype="float32"), tone()]) == 0
    out = capsys.readouterr().out
    assert "no audio was captured" in out and "record it again" in out.lower()
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_a_too_short_clip_is_asked_again(monkeypatch, tmp_path, capsys):
    assert run_with_takes(monkeypatch, tmp_path, [tone(0.1), tone(1.0)]) == 0
    assert "too short" in capsys.readouterr().out
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_a_silent_clip_is_asked_again_and_the_headset_is_named(monkeypatch, tmp_path, capsys):
    np = __import__("numpy")
    assert run_with_takes(monkeypatch, tmp_path, [np.zeros(16000, dtype="float32"), tone(1.0)]) == 0
    out = capsys.readouterr().out
    assert "too quiet" in out and "soundcore" in out
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_a_good_clip_reports_its_length_and_level(monkeypatch, tmp_path, capsys):
    assert run_with_takes(monkeypatch, tmp_path, [tone(1.0)]) == 0
    out = capsys.readouterr().out
    assert "ok" in out and "1.0 s" in out and "dBFS" in out
