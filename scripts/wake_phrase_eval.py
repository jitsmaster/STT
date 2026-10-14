"""Detection rate and false triggers per hour of the wake phrase over a grid of thresholds and scores.

Usage: .venv\\Scripts\\python scripts\\wake_phrase_eval.py [--keywords keywords_phrase.txt] [--positives samples/wakephrase]
Positives: every wake-*.wav in --positives. Negatives: every other clip of samples/headset (speech and ambient), the
only speech that is not the phrase. Prints the settings that reach >= 90% detection with <= 1 false trigger an hour.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

from stt.audio import SAMPLE_RATE, load_wav
from stt.bench.prompts import load_manifest
from stt.bench.wake import count_triggers, make_spotter

PAD = np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)


def load_set(positives, negatives_dir):
    pos = [np.concatenate([load_wav(p), PAD]) for p in sorted(Path(positives).glob("wake-*.wav"))]
    neg = []
    root = Path(negatives_dir)
    for it in load_manifest(root / "manifest.json"):
        wav = root / (it["id"] + ".wav")
        if it["kind"] != "wake" and wav.exists():
            neg.append(np.concatenate([load_wav(wav), PAD]))
    return pos, neg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", default="keywords_phrase.txt")
    ap.add_argument("--positives", default="samples/wakephrase")
    ap.add_argument("--negatives", default="samples/headset")
    ap.add_argument("--model-dir", default="models/kws")
    args = ap.parse_args(argv)
    pos, neg = load_set(args.positives, args.negatives)
    neg_s = sum(len(a) for a in neg) / SAMPLE_RATE
    print(f"{len(pos)} positive clips, {len(neg)} negative clips ({neg_s:.0f} s)")
    rows = []
    for thr, score in itertools.product([0.5, 0.35, 0.25, 0.15, 0.1, 0.05], [1.0, 1.5, 2.0, 3.0]):
        sp = make_spotter(args.model_dir, args.keywords, thr, score)
        det = sum(count_triggers(sp, a) > 0 for a in pos)
        fa = sum(count_triggers(sp, a) for a in neg)
        rate = det / len(pos) if pos else 0.0
        fph = fa / (neg_s / 3600) if neg_s else float("nan")
        rows.append((rate, fph, thr, score, det, fa))
        print(f"thr={thr:<5} score={score:<4} detect={det}/{len(pos)} ({rate:.0%}) false={fa} (~{fph:.1f}/h)", flush=True)
    ok = [r for r in rows if r[0] >= 0.9 and r[1] <= 1]
    print("\nMeets the bar (>=90% detect, <=1 false/h):")
    for r in sorted(ok, key=lambda r: (r[1], -r[0]))[:5] or ["none"]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
