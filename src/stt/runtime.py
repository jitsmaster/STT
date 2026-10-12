import asyncio
import logging
import threading
from pathlib import Path

import numpy as np

from . import __version__
from .auth import load_or_create_secret
from .core import ListenerCore
from .endpointers import make_endpointer
from .engines.base import Hints
from .engines.factory import build_engines
from .hotkey import HotkeyListener
from .logs import install_token_redaction
from .mic import MicSource, capture_loop, resolve_device
from .pipeline import Bus, Pipeline
from .server import create_app
from .wake import WakeWatcher

log = logging.getLogger("stt")
EARCON_TONES = {"start": [(880, 70)], "end": [(660, 70)], "cancel": [(440, 90), (330, 90)]}


def touch_stop_flag(state_dir) -> None:
    folder = Path(state_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tts-stop.flag").touch()


EARCON_LEAD_S = 0.4  # near-silence that wakes a Bluetooth headset's output before the tone
EARCON_RATE = 16_000


def earcon_wav(kind) -> bytes:
    """The earcon as an in-memory WAV: a quiet lead-in (a 1-LSB-level signal, so the sink does not idle off) then the tones."""
    import io
    import wave

    import numpy as np

    parts = [np.where(np.arange(int(EARCON_LEAD_S * EARCON_RATE)) % 2 == 0, 8, -8).astype("<i2")]
    for freq, ms in EARCON_TONES.get(kind, ()):
        t = np.arange(int(EARCON_RATE * max(ms, 120) / 1000)) / EARCON_RATE
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
        parts.append((0.5 * 32767 * fade * np.sin(2 * np.pi * freq * t)).astype("<i2"))
        parts.append(np.zeros(int(EARCON_RATE * 0.04), dtype="<i2"))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(EARCON_RATE)
        w.writeframes(np.concatenate(parts).tobytes())
    return out.getvalue()


def beep(kind) -> None:
    def play():
        try:
            import winsound

            winsound.PlaySound(earcon_wav(kind), winsound.SND_MEMORY | winsound.SND_NODEFAULT)
        except Exception:
            log.debug("earcon %s failed", kind, exc_info=True)

    threading.Thread(target=play, name="stt-earcon", daemon=True).start()


def make_hotkey_handler(pipeline):
    def on_press():
        log.info("hotkey pressed, state %s", pipeline.state)
        if pipeline.state == "listening":
            pipeline.finish()
        else:
            pipeline.trigger("hotkey")

    return on_press


def list_input_devices(devices) -> list[str]:
    return [str(i) + ": " + d["name"] for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]


async def warm_up(engines) -> None:
    silence = np.zeros(16_000, dtype="float32")
    for engine in {id(e): e for e in engines.values()}.values():
        await asyncio.to_thread(engine.transcribe, silence, Hints())
    log.info("engines warm")


async def run_daemon(config) -> None:
    import sounddevice
    import uvicorn

    from .bench.wake import make_spotter

    secret = load_or_create_secret(config.state_dir)
    engines = build_engines(config)
    core = ListenerCore(make_endpointer(config), no_speech_s=config.no_speech_s, silence_end_s=config.silence_end_s,
                        max_s=config.max_utterance_s, min_speech_s=config.min_speech_s)
    wake = None
    if config.wake_enabled:
        wake = WakeWatcher(make_spotter(config.wake_model_dir, config.wake_keywords, config.wake_threshold, config.wake_score))
    bus = Bus()
    pipeline = Pipeline(core, engines, bus, tts_stop=lambda: touch_stop_flag(config.state_dir), earcon=beep,
                        glossary=config.glossary, wake=wake)
    info = {"version": __version__, "hotkey": config.hotkey, "wake": wake is not None,
            "engines": {role: engine.name for role, engine in engines.items()}}
    app = create_app(pipeline, bus, secret, info)
    source = MicSource(resolve_device(sounddevice.query_devices(), config.device))
    loop = asyncio.get_running_loop()
    on_press = make_hotkey_handler(pipeline)
    hotkey = HotkeyListener(config.hotkey, lambda: loop.call_soon_threadsafe(on_press))
    hotkey.start()  # raises OSError when the combination is taken: fail before the port opens
    try:
        await warm_up(engines)
        uvicorn_config = uvicorn.Config(app, host=config.host, port=config.port, log_level="info", access_log=False)
        install_token_redaction()  # after uvicorn configured its loggers: the WebSocket URL (with the token) is logged at INFO
        server = uvicorn.Server(uvicorn_config)
        capture = asyncio.create_task(capture_loop(pipeline, source, wake is not None))
        capture.add_done_callback(
            lambda t: not t.cancelled() and t.exception() and log.error("capture loop died: %r", t.exception())
        )
        try:
            await server.serve()
        finally:
            capture.cancel()
    finally:
        hotkey.stop()
