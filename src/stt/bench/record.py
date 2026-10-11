import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from stt.audio import SAMPLE_RATE, SILENCE_RMS, rms, save_wav
from stt.mic import resolve_device

from .prompts import build_manifest, missing_samples, save_manifest


def record_until_enter(device):
    frames = []

    def callback(indata, n, t, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=callback):
        input("  recording... press Enter to stop ")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, dtype="float32")


def check_opens(device):
    """Open the input for a moment, so a device that vanished, cannot be opened or opens but delivers no audio (a Bluetooth
    headset that is not streaming) fails before the first prompt instead of saving empty clips."""
    frames = [0]

    def callback(indata, n, t, status):
        frames[0] += n

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=callback):
        time.sleep(0.6)
    if frames[0] == 0:
        raise sd.PortAudioError("the device opened but delivered no audio (headset not streaming: reconnect it, check the "
                                "Windows sound input level, close apps using the mic)")


MIN_SECONDS = 0.3


def clip_problem(audio):
    """Why a take cannot be kept, or None when it is usable (length and level of the raw audio, before saving)."""
    if not len(audio):
        return "no audio was captured (the device delivered nothing)"
    seconds = len(audio) / SAMPLE_RATE
    if seconds < MIN_SECONDS:
        return f"too short ({seconds:.1f} s; Enter was pressed too soon)"
    if rms(audio) < SILENCE_RMS:
        return "too quiet (nothing heard; is this the headset mic, and is it streaming?)"
    return None


def level_db(audio):
    return 20 * np.log10(max(rms(audio), 1e-9))


def record_seconds(seconds, device):
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record the bake-off sample set (resumable).")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--device", default=None, help="input device index or a name fragment such as soundcore (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args(argv)
    # Device names and the Chinese prompts are not encodable in the cp1252 of a redirected stdout.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.list_devices:
        print(sd.query_devices())
        return 0
    try:
        device = resolve_device(sd.query_devices(), args.device)
    except ValueError as exc:  # an output device, an index that moved, or an unknown name
        print("record: " + str(exc), file=sys.stderr)
        return 1
    try:
        check_opens(device)
    except sd.PortAudioError as exc:
        print(f"record: input device {device} cannot be opened: {exc}\nrun with --list-devices and pick another "
              "(Bluetooth entries named Hands-Free under WDM-KS cannot be used; reconnect the headset)", file=sys.stderr)
        return 1
    device_name = sd.query_devices()[device]["name"]
    print(f"Recording from input device {device}: {device_name}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = build_manifest()
    save_manifest(items, out / "manifest.json")
    todo = missing_samples(items, out)
    print(f"{len(todo)} of {len(items)} samples to record. Ctrl+C stops; rerun to resume.")
    for item in todo:
        print(f"\n[{item['id']}] ({item['kind']}) say: {item['text']}")
        if item["kind"] == "ambient":
            input(f"  Press Enter, then do this for {item['seconds']} s ")
            audio = record_seconds(item["seconds"], device)
        else:
            while True:
                input("  Press Enter to start ")
                audio = record_until_enter(device)
                problem = clip_problem(audio)
                if problem is None:
                    print(f"  ok: {len(audio) / SAMPLE_RATE:.1f} s, level {level_db(audio):.0f} dBFS")
                    break
                print(f"  NOT SAVED, {problem}. Record it again (Ctrl+C to stop and check {device_name}).")
        save_wav(out / (item["id"] + ".wav"), audio)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
