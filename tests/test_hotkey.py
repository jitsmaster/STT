import sys

import pytest

from stt.hotkey import MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, HotkeyListener, parse_hotkey


def test_the_default_hotkey_parses_to_ctrl_alt_v_without_repeat():
    assert parse_hotkey("ctrl+alt+v") == (MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 0x56)


def test_case_spaces_function_keys_and_named_keys():
    assert parse_hotkey("Ctrl + Shift + F9") == (MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, 0x78)
    assert parse_hotkey("alt+space")[1] == 0x20
    assert parse_hotkey("control+alt+7")[1] == ord("7")
    assert parse_hotkey("ctrl+alt+f24")[1] == 0x87


@pytest.mark.parametrize("bad", ["v", "ctrl+", "+v", "", "hyper+v", "ctrl+nope", "ctrl+f25", "ctrl+alt"])
def test_bad_hotkeys_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_hotkey(bad)


def test_the_default_does_not_collide_with_the_tts_stop_hotkey():
    assert parse_hotkey("ctrl+alt+v") != parse_hotkey("ctrl+alt+m")


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
def test_a_hotkey_that_another_listener_owns_fails_loudly():
    first, second = HotkeyListener("ctrl+alt+shift+f23", lambda: None, 1), HotkeyListener("ctrl+alt+shift+f23", lambda: None, 2)
    first.start()
    try:
        with pytest.raises(OSError, match="already in use"):
            second.start()
    finally:
        first.stop()
        second.stop()
