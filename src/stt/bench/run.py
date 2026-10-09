import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from stt.audio import is_silent, load_wav
from stt.engines.base import Hints, Transcript
from stt.engines.factory import make_engine

from .intents import match_intent
from .prompts import load_manifest
from .scoring import number_ok, term_hits, token_error_rate

SCORED_KINDS = ("command", "dictation", "mixed")
GLOSSARY = ["worktree", "PR", "dsh-plugins", "sessionController", "Kokoro", "pnpm", "rebase", "develop", "faster-whisper"]


@dataclass
class Row:
    engine: str
    id: str
    kind: str
    lang: str
    expected: str
    got: str
    detected: str
    ter: float
    term_rate: float | None
    number_ok: bool | None
    intent_ok: bool | None
    seconds: float


def run_bench(items, samples_dir, engine, glossary, oracle_lang) -> list[Row]:
    rows = []
    for item in items:
        if item["kind"] not in SCORED_KINDS:
            continue
        wav = Path(samples_dir) / (item["id"] + ".wav")
        if not wav.exists():
            continue
        audio = load_wav(wav)
        if is_silent(audio):
            result = Transcript("", "", 0.0, 0.0)
        else:
            lang = item["lang"] if oracle_lang and item["lang"] in ("en", "zh") else None
            result = engine.transcribe(audio, Hints(language=lang, glossary=tuple(glossary)))
        rows.append(Row(
            engine=engine.name, id=item["id"], kind=item["kind"], lang=item["lang"],
            expected=item["text"], got=result.text, detected=result.language,
            ter=token_error_rate(item["text"], result.text),
            term_rate=term_hits(result.text, item["terms"]) if item["terms"] else None,
            number_ok=number_ok(result.text, item["number"]) if item.get("number") else None,
            intent_ok=(match_intent(result.text) == item["intent"]) if item.get("intent") else None,
            seconds=result.seconds,
        ))
    return rows


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows) -> dict:
    out = {}
    for kind in SCORED_KINDS:
        rs = [r for r in rows if r.kind == kind]
        if not rs:
            continue
        out[kind] = {
            "n": len(rs),
            "ter": _mean(r.ter for r in rs),
            "terms": _mean(r.term_rate for r in rs),
            "number": _mean(r.number_ok for r in rs),
            "intent": _mean(r.intent_ok for r in rs),
            "median_seconds": statistics.median(r.seconds for r in rs),
        }
    return out


def _fmt(value, pct=False):
    if value is None:
        return "-"
    return f"{value * 100:.0f}%" if pct else f"{value:.2f}"


def render_markdown(summaries: dict) -> str:
    lines = ["| engine | kind | n | token error | glossary hit | number ok | intent ok | median s |",
             "|---|---|---|---|---|---|---|---|"]
    for engine, by_kind in summaries.items():
        for kind, s in by_kind.items():
            lines.append(f"| {engine} | {kind} | {s['n']} | {_fmt(s['ter'], True)} | {_fmt(s['terms'], True)} | "
                         f"{_fmt(s['number'], True)} | {_fmt(s['intent'], True)} | {_fmt(s['median_seconds'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--engines", required=True, help="comma list, e.g. faster-whisper:large-v3,sensevoice:models/sensevoice")
    ap.add_argument("--glossary", action="store_true", help="bias engines with the glossary")
    ap.add_argument("--oracle-lang", action="store_true", help="force the known language instead of auto-detecting")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    items = load_manifest(Path(args.samples) / "manifest.json")
    summaries, all_rows = {}, []
    suffix = (" +glossary" if args.glossary else "") + (" +oracle-lang" if args.oracle_lang else "")
    for spec in args.engines.split(","):
        try:
            engine = make_engine(spec)
        except ValueError as exc:
            raise SystemExit(str(exc))
        rows = run_bench(items, args.samples, engine, GLOSSARY if args.glossary else [], args.oracle_lang)
        summaries[engine.name + suffix] = summarize(rows)
        all_rows += [asdict(r) for r in rows]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat() + ("-g" if args.glossary else "-n") + ("o" if args.oracle_lang else "a")
    (out / (stamp + ".json")).write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(summaries)
    (out / (stamp + ".md")).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
