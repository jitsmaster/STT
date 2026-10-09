import re

from .scoring import normalize, numbers_in

# Order matters: the more specific pattern comes first.
_RULES = [
    ("repeat_question", r"\brepeat (the )?question\b|重复问题"),
    ("repeat", r"\brepeat\b|重复"),
    ("status", r"\bstatus\b|\bwaiting\b|状态"),
    ("mute", r"\bmute\b|静音"),
    ("cancel", r"\bcancel\b|取消"),
    ("approve", r"\bapprove\b|通过"),
    ("deny", r"\bdeny\b|拒绝"),
    ("yes", r"^(yes|yeah)\b"),
    ("read", r"^read\b"),
    ("option", r"\boption\b|选"),
    ("stop", r"\bstop\b|停止"),
    ("open", r"\bopen\b|打开"),
    ("new", r"\bnew\b|新建"),
]
_NUMBERED = {"option", "stop", "open"}


def match_intent(text: str) -> str:
    t = normalize(text)
    for intent, pattern in _RULES:
        if re.search(pattern, t):
            if intent in _NUMBERED:
                nums = numbers_in(t)
                return f"{intent}:{nums[0]}" if nums else intent
            return intent
    return "unknown"
