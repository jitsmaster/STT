import re

_CJK = re.compile(r"[\u3400-\u9fff]")


def to_simplified(text: str) -> str:
    """Traditional to Simplified Chinese (characters only, no regional word swaps). Whisper answers in either script;
    the command grammar and the scoring use Simplified."""
    if not _CJK.search(text):
        return text
    from zhconv import convert

    return convert(text, "zh-hans")
