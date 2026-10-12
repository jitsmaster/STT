"""Sweep wake-word keyword spellings, thresholds and scores over the recorded sample sets.

Usage: .venv\\Scripts\\python scripts\\wake_sweep.py [laptop|headset ...]
Prints detection rate and false triggers per hour for each setting (ambient clips are the only negatives).
"""
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

from stt.audio import SAMPLE_RATE, load_wav
from stt.bench.prompts import load_manifest
from stt.bench.wake import make_spotter

ROOT = Path(__file__).resolve().parent.parent
PAD = np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)

SPELLINGS = {
    "base": ["m ǎ n óng"],
    "toneless": ["m a n ong"],
    "tones4x4": [f"m {a} n {o}" for a in "ǎāáà" for o in ("óng", "ōng", "ǒng", "òng")],
    "mid": ["m ǎ n óng", "m ā n óng", "m à n óng", "m ǎ n ōng", "m a n ong"],
}


def keyword_file(lines, boost=None, thr=None):
    body = "\n".join(f"{l}{f' :{boost}' if boost else ''}{f' #{thr}' if thr else ''} @码农" for l in lines)
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(body + "\n")
    f.close()
    return f.name


def triggers(spotter, audio):
    stream = spotter.create_stream()
    stream.accept_waveform(SAMPLE_RATE, np.concatenate([audio, PAD]))
    stream.input_finished()
    hits = 0
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        if spotter.get_result(stream):
            hits += 1
            spotter.reset_stream(stream)
    return hits


def evaluate(spotter, sets):
    det = tot = amb_hits = 0
    amb_s = 0.0
    for name in sets:
        d = ROOT / "samples" / name
        for item in load_manifest(d / "manifest.json"):
            wav = d / (item["id"] + ".wav")
            if item["kind"] not in ("wake", "ambient") or not wav.exists():
                continue
            audio = load_wav(wav)
            h = triggers(spotter, audio)
            if item["kind"] == "wake":
                tot += 1
                det += h > 0
            else:
                amb_hits += h
                amb_s += len(audio) / SAMPLE_RATE
    return det, tot, amb_hits, amb_s


def main(argv):
    sets = argv or ["laptop", "headset"]
    rows = []
    import os
    spells = os.environ.get("WAKE_SPELLS", ",".join(SPELLINGS)).split(",")
    thrs = [float(x) for x in os.environ.get("WAKE_THR", "0.05,0.1,0.15,0.25").split(",")]
    scores = [float(x) for x in os.environ.get("WAKE_SCORE", "1,2,3").split(",")]
    for spell, thr, score in itertools.product(spells, thrs, scores):
        kf = keyword_file(SPELLINGS[spell])
        sp = make_spotter(str(ROOT / "models" / "kws"), kf, thr, score)
        det, tot, ah, asec = evaluate(sp, sets)
        fph = ah / (asec / 3600) if asec else float("nan")
        rows.append((det / tot, fph, spell, thr, score, det, tot, ah))
        print(f"{spell:9} thr={thr:<5} score={score:<4} detect={det}/{tot} ({det/tot:.0%}) ambient_hits={ah} ~{fph:.1f}/h", flush=True)
    best = sorted(rows, key=lambda r: (-(r[0] if r[1] <= 1 else r[0] - 1), r[1]))[:5]
    print("\nBEST")
    for r in best:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1:])
