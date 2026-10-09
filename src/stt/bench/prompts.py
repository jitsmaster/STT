import json
from pathlib import Path

from .scoring import numbers_in

COMMANDS_EN = [
    ("status", "status"), ("what's waiting", "status"), ("mute", "mute"), ("repeat", "repeat"),
    ("repeat question", "repeat_question"), ("cancel", "cancel"), ("yes", "yes"),
    ("approve", "approve"), ("deny", "deny"), ("option two", "option:2"), ("option three", "option:3"),
    ("stop three", "stop:3"), ("open five", "open:5"), ("new in dsh plugins", "new"), ("read", "read"),
]
COMMANDS_ZH = [
    ("状态", "status"), ("静音", "mute"), ("重复", "repeat"), ("取消", "cancel"), ("通过", "approve"),
    ("拒绝", "deny"), ("选二", "option:2"), ("选三", "option:3"), ("停止三号", "stop:3"), ("新建会话", "new"),
]
DICTATION = [
    ("Rebase the worktree onto develop and run the tests", ["worktree", "develop"]),
    ("Open pull request PR 6512 against the develop branch", ["PR", "6512", "develop"]),
    ("Add a failing test for the session controller before changing the plugin", ["session controller"]),
    ("Check whether dsh-plugins hooks still speak the full response", ["dsh-plugins", "hooks"]),
    ("Run pnpm install in the worktree and report any peer dependency warnings", ["pnpm", "worktree"]),
    ("Why does the Kokoro server restart when the headset microphone opens", ["Kokoro"]),
    ("Merge the feature branch and delete the local worktree afterwards", ["worktree"]),
    ("Summarize the review comments on PR 6512 in three bullet points", ["PR", "6512"]),
    ("Use faster-whisper large v3 with int8 on the CPU", ["faster-whisper", "int8"]),
    ("Update the README and commit with a conventional message", ["README"]),
]
MIXED = [
    ("把 PR 6512 的 session 打开", ["PR", "6512", "session"]),
    ("给 dsh-plugins 新建一个 worktree", ["dsh-plugins", "worktree"]),
    ("帮我 rebase 到 develop 然后跑测试", ["rebase", "develop"]),
    ("告诉第三个 session 继续", ["session"]),
    ("这个 pull request 的 review comment 总结一下", ["pull request", "review comment"]),
]
WAKE_WORD = "码农"
WAKE_REPEATS = 20
AMBIENT = [
    "a normal phone or video call (talk to someone)",
    "music or a video playing",
    "you talking to yourself while working",
]
AMBIENT_SECONDS = 60


def _item(item_id, kind, lang, text, intent=None, terms=(), number=None, **extra):
    return {"id": item_id, "kind": kind, "lang": lang, "text": text, "intent": intent,
            "terms": list(terms), "number": number, **extra}


def build_manifest() -> list[dict]:
    items = []
    for lang, table in (("en", COMMANDS_EN), ("zh", COMMANDS_ZH)):
        for n, (text, intent) in enumerate(table, 1):
            nums = numbers_in(text)
            items.append(_item(f"cmd-{lang}-{n:02d}", "command", lang, text, intent, number=nums[0] if nums else None))
    for n, (text, terms) in enumerate(DICTATION, 1):
        items.append(_item(f"dict-{n:02d}", "dictation", "en", text, terms=terms))
    for n, (text, terms) in enumerate(MIXED, 1):
        items.append(_item(f"mix-{n:02d}", "mixed", "mixed", text, terms=terms))
    for n in range(1, WAKE_REPEATS + 1):
        items.append(_item(f"wake-{n:02d}", "wake", "zh", WAKE_WORD))
    for n, what in enumerate(AMBIENT, 1):
        items.append(_item(f"ambient-{n:02d}", "ambient", "any", what, seconds=AMBIENT_SECONDS))
    return items


def save_manifest(items, path) -> None:
    Path(path).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def missing_samples(items, samples_dir) -> list[dict]:
    return [i for i in items if not (Path(samples_dir) / (i["id"] + ".wav")).exists()]
