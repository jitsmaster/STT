import ctypes
import logging
import threading
from ctypes import wintypes

log = logging.getLogger("stt")

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x0001, 0x0002, 0x0004, 0x0008, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
MODIFIERS = {"alt": MOD_ALT, "ctrl": MOD_CONTROL, "control": MOD_CONTROL, "shift": MOD_SHIFT, "win": MOD_WIN}
NAMED_KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B}


def parse_hotkey(text) -> tuple[int, int]:
    parts = [p.strip().lower() for p in str(text).split("+")]
    if len(parts) < 2 or not all(parts):
        raise ValueError("hotkey needs at least one modifier and a key, for example ctrl+alt+v")
    *mods, key = parts
    flags = 0
    for mod in mods:
        if mod not in MODIFIERS:
            raise ValueError("unknown modifier " + repr(mod))
        flags |= MODIFIERS[mod]
    if key in MODIFIERS:
        raise ValueError("the last part must be a key, not the modifier " + repr(key))
    if len(key) == 1 and key.isascii() and key.isalnum():
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key in NAMED_KEYS:
        vk = NAMED_KEYS[key]
    else:
        raise ValueError("unknown key " + repr(key))
    return flags | MOD_NOREPEAT, vk


class HotkeyListener:
    def __init__(self, hotkey, callback, hotkey_id=1):
        self.hotkey, self.callback, self.id = hotkey, callback, hotkey_id
        self._thread, self._thread_id, self._error = None, None, None

    def start(self, timeout=5.0) -> None:
        modifiers, vk = parse_hotkey(self.hotkey)
        ready = threading.Event()

        def run():
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            self._thread_id = ctypes.WinDLL("kernel32").GetCurrentThreadId()
            if not user32.RegisterHotKey(None, self.id, modifiers, vk):
                self._error = OSError(
                    "hotkey " + self.hotkey + " is already in use by another program or not allowed (Windows error "
                    + str(ctypes.get_last_error()) + ")"
                )
                ready.set()
                return
            ready.set()
            msg = wintypes.MSG()
            try:
                while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                    if msg.message == WM_HOTKEY and msg.wParam == self.id:
                        try:
                            self.callback()
                        except Exception:
                            log.exception("hotkey callback failed")
            finally:
                user32.UnregisterHotKey(None, self.id)

        self._thread = threading.Thread(target=run, name="stt-hotkey", daemon=True)
        self._thread.start()
        if not ready.wait(timeout):
            raise OSError("the hotkey thread did not start")
        if self._error:
            raise self._error

    def stop(self) -> None:
        if self._thread is not None and self._thread_id and self._thread.is_alive():
            ctypes.WinDLL("user32").PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2)
        self._thread = None
