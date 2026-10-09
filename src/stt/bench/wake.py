import argparse
from pathlib import Path

from stt.audio import SAMPLE_RATE, load_wav

from .prompts import load_manifest


def count_triggers(spotter, audio) -> int:
    stream = spotter.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    stream.input_finished()
    hits = 0
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        if spotter.get_result(stream):
            hits += 1
            spotter.reset_stream(stream)
    return hits


def wake_report(spotter, samples_dir, items) -> dict:
    detected = total = 0
    ambient_hits = 0
    ambient_seconds = 0.0
    for item in items:
        wav = Path(samples_dir) / (item["id"] + ".wav")
        if item["kind"] not in ("wake", "ambient") or not wav.exists():
            continue
        audio = load_wav(wav)
        hits = count_triggers(spotter, audio)
        if item["kind"] == "wake":
            total += 1
            detected += hits > 0
        else:
            ambient_hits += hits
            ambient_seconds += len(audio) / SAMPLE_RATE
    return {
        "total": total, "detected": detected, "rate": detected / total if total else None,
        "false_per_hour": ambient_hits / (ambient_seconds / 3600) if ambient_seconds else None,
    }


def model_files(model_dir) -> dict:
    """The chunk-16 files of the model archive. Only the encoder and joiner have int8 variants (the archive
    ships the decoder as fp32 only), and a plain glob would match both variants: prefer int8 explicitly."""
    d = Path(model_dir)

    def pick(part: str) -> Path:
        int8 = sorted(d.glob(f"{part}-*chunk-16*.int8.onnx"))
        plain = sorted(p for p in d.glob(f"{part}-*chunk-16*.onnx") if not p.name.endswith(".int8.onnx"))
        found = int8 or plain
        if not found:
            raise FileNotFoundError(f"no {part}-*chunk-16*.onnx in {d}")
        return found[0]

    return {"encoder": pick("encoder"), "decoder": pick("decoder"), "joiner": pick("joiner"), "tokens": d / "tokens.txt"}


def make_spotter(model_dir: str, keywords_file: str, threshold: float = 0.25, score: float = 1.5):
    """Build a sherpa-onnx KeywordSpotter (arguments checked against the installed 1.13.8 signature)."""
    import sherpa_onnx

    f = model_files(model_dir)
    return sherpa_onnx.KeywordSpotter(
        tokens=str(f["tokens"]),
        encoder=str(f["encoder"]),
        decoder=str(f["decoder"]),
        joiner=str(f["joiner"]),
        keywords_file=keywords_file,
        keywords_score=score,
        keywords_threshold=threshold,
        num_threads=2,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--model-dir", default="models/kws")
    ap.add_argument("--keywords", default="keywords.txt")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--score", type=float, default=1.5)
    args = ap.parse_args(argv)
    items = load_manifest(Path(args.samples) / "manifest.json")
    spotter = make_spotter(args.model_dir, args.keywords, args.threshold, args.score)
    print(wake_report(spotter, args.samples, items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
