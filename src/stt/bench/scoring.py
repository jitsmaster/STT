import re
import unicodedata

_CJK = re.compile(r"[\u4e00-\u9fff]")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s]", " ", t)  # \w keeps digits, letters and Chinese characters
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    """Each Chinese character is one token; other runs split on whitespace into words."""
    out: list[str] = []
    for part in normalize(text).split():
        buf = ""
        for ch in part:
            if _CJK.match(ch):
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
            else:
                buf += ch
        if buf:
            out.append(buf)
    return out


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def token_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _squash(s: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", normalize(s))


def term_hits(hypothesis: str, terms: list[str]) -> float:
    """Fraction of terms present. A term matches a run of whole tokens, so 'dsh plugins' hits
    'dsh-plugins' and 'PR' does not hit inside 'approve'."""
    if not terms:
        return 1.0
    toks = [t for t in (_squash(t) for t in tokens(hypothesis)) if t]

    def present(term: str) -> bool:
        want = _squash(term)
        for i in range(len(toks)):
            acc = ""
            for j in range(i, len(toks)):
                acc += toks[j]
                if acc == want:
                    return True
                if len(acc) >= len(want):
                    break
        return False

    return sum(present(t) for t in terms) / len(terms)


_EN = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_ZH = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
       "七": 7, "八": 8, "九": 9, "十": 10}
_PINYIN = {"ling": 0, "yi": 1, "er": 2, "liang": 2, "san": 3, "si": 4, "wu": 5,
           "liu": 6, "qi": 7, "ba": 8, "jiu": 9, "shi": 10}


def numbers_in(text: str) -> list[int]:
    out: list[int] = []
    for tok in tokens(text):
        if tok.isdigit():
            out.append(int(tok))
        elif tok in _EN:
            out.append(_EN[tok])
        elif tok in _ZH:
            out.append(_ZH[tok])
        elif tok in _PINYIN:
            out.append(_PINYIN[tok])
    return out


def number_ok(heard: str, expected: int) -> bool:
    return expected in numbers_in(heard)
