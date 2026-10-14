"""Hands-free recorder for the wake phrase takes (no Enter key): a beep, then a 4 s window to say the phrase.

Usage: .venv\\Scripts\\python scripts\\record_wake_phrase.py [--device soundcore] [--takes 20] [--out samples/wakephrase]
The microphone stream stays open for the whole run (a Bluetooth headset needs a moment to start streaming), takes are
cut from it after each beep. Takes that are too quiet are listed at the end so they can be redone (rerun resumes).
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from stt.audio import SAMPLE_RATE, save_wav
from stt.bench.prompts import save_manifest
from stt.bench.record import clip_problem, level_db
from stt.mic import resolve_device

PHRASE = "码农，开始干活儿"
TAKE_S = 4.0
GAP_S = 1.2
STYLES = ["normal voice", "normal voice", "normal voice", "normal voice", "a bit faster",
          "a bit faster", "a bit quieter", "a bit quieter", "a little further from the mic", "casual, like talking to a colleague"]


def beep():
    import winsound
    winsound.Beep(1000, 180)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="soundcore")
    ap.add_argument("--takes", type=int, default=20)
    ap.add_argument("--out", default="samples/wakephrase")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = resolve_device(sd.query_devices(), args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [{"id": f"wake-{i:02d}", "kind": "wake", "text": PHRASE} for i in range(1, args.takes + 1)]
    save_manifest(items, out / "manifest.json")
    todo = [it for it in items if not (out / (it["id"] + ".wav")).exists()]
    print(f"Recording from {sd.query_devices()[device]['name']}; {len(todo)} takes to do. After each beep say: {PHRASE}", flush=True)
    chunks, lock, flowing = [], threading.Lock(), threading.Event()

    def callback(indata, n, t, status):
        with lock:
            chunks.append(indata[:, 0].copy())
        flowing.set()

    def grab(seconds):
        with lock:
            start = sum(len(c) for c in chunks)
        time.sleep(seconds)
        with lock:
            data = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        return data[start:start + int(seconds * SAMPLE_RATE)]

    bad = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=callback):
        if not flowing.wait(5):
            print("the headset delivered no audio; reconnect it and rerun", flush=True)
            return 1
        time.sleep(2.0)
        for n, it in enumerate(todo):
            style = STYLES[int(it["id"].split("-")[1]) % len(STYLES)]
            print(f"[{it['id']}] ({style}) ...", flush=True)
            beep()
            audio = grab(TAKE_S)
            problem = clip_problem(audio)
            if problem:
                print(f"  NOT SAVED: {problem}", flush=True)
                bad.append(it["id"])
            else:
                save_wav(out / (it["id"] + ".wav"), audio)
                print(f"  ok, level {level_db(audio):.0f} dBFS", flush=True)
            time.sleep(GAP_S)
    print("done" + (f"; redo by rerunning: {', '.join(bad)}" if bad else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
