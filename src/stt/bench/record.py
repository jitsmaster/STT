import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from stt.audio import SAMPLE_RATE, save_wav
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
    """Open and close the input once, so a device that vanished or is not usable fails before the first prompt."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device):
        pass


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
            input("  Press Enter to start ")
            audio = record_until_enter(device)
        save_wav(out / (item["id"] + ".wav"), audio)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
