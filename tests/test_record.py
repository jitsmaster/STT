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
    monkeypatch.setattr(record, "record_until_enter", lambda device: seen.append(device) or __import__("numpy").zeros(1600, dtype="float32"))
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
