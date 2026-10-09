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
