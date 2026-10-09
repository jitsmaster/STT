import asyncio
import logging
import queue
import time

import numpy as np

from .audio import SAMPLE_RATE

FRAME_SAMPLES = 1600  # 0.1 s
log = logging.getLogger("stt")


def resolve_device(devices, query):
    """Pick an input device index from sounddevice's device list; None means the system default."""
    if query is None:
        return None
    inputs = [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    listing = ", ".join(str(i) + ": " + d["name"] for i, d in inputs)
    if isinstance(query, int) or str(query).strip().isdigit():
        index = int(query)
        if any(i == index for i, _ in inputs):
            return index
        raise ValueError("device " + str(index) + " is not an input device; inputs are " + listing)
    needle = str(query).strip().lower()
    for i, d in inputs:
        if needle in d["name"].lower():
            return i
    raise ValueError("no input device matches " + repr(query) + "; inputs are " + listing)


class MicSource:
    def __init__(self, device=None, *, stream_factory=None, max_frames=100):
        self.device = device
        self._factory = stream_factory
        self._queue = queue.Queue(maxsize=max_frames)
        self._stream = None

    @property
    def started(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time, status):
        frame = np.array(indata[:, 0], dtype="float32")
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop the oldest rather than block the audio thread
                self._queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        factory = self._factory
        if factory is None:
            import sounddevice

            factory = sounddevice.InputStream
        stream = factory(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SAMPLES,
                         device=self.device, callback=self._callback)
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def next_frame(self, timeout=0.1):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def _fail(pipeline, message):
    """Report a microphone problem for the current window (if any) and close it."""
    pipeline.bus.publish({"type": "error", "message": message, "context": pipeline.context})
    pipeline.cancel()


async def capture_loop(pipeline, source, wake_enabled, *, poll_s=0.02, retry_s=1.0, stall_s=2.0):
    """Feed mic frames to the pipeline. The mic is open only while listening, unless the wake word is on.

    Supervised: an exception anywhere in an iteration (a failing stop(), the VAD, the spotter) is logged, reported
    as an error event, ends the current window and the loop carries on. A listening window whose frames stop
    arriving (device unplugged) is closed after stall_s seconds of wall-clock time."""
    last_error = None
    last_frame = time.monotonic()
    was_listening = False
    try:
        while True:
            try:
                listening = pipeline.state == "listening"
                if listening and not was_listening:
                    last_frame = time.monotonic()
                was_listening = listening
                want = wake_enabled or listening
                if not want:
                    if source.started:
                        source.stop()
                    await asyncio.sleep(poll_s)
                    continue
                if not source.started:
                    try:
                        source.start()
                        last_error = None
                    except Exception as exc:
                        message = "microphone unavailable: " + str(exc)
                        if message != last_error:
                            last_error = message
                            pipeline.bus.publish({"type": "error", "message": message, "context": pipeline.context})
                        pipeline.cancel()
                        await asyncio.sleep(retry_s)
                        continue
                frame = await asyncio.to_thread(source.next_frame, 0.1)
                if frame is not None:
                    last_frame = time.monotonic()
                    pipeline.feed(frame)
                elif listening and time.monotonic() - last_frame > stall_s:
                    _fail(pipeline, "no audio from the microphone for %.1f s" % stall_s)
                    source.stop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("capture loop iteration failed")
                _fail(pipeline, "microphone loop failed: " + str(exc))
                try:
                    source.stop()
                except Exception:
                    log.exception("closing the microphone failed")
                await asyncio.sleep(retry_s)
    finally:
        if source.started:
            source.stop()
