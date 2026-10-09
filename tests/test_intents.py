import pytest

from stt.bench.intents import match_intent


@pytest.mark.parametrize("text,intent", [
    ("status", "status"), ("What's waiting?", "status"), ("状态", "status"),
    ("mute", "mute"), ("静音", "mute"),
    ("repeat", "repeat"), ("repeat question", "repeat_question"), ("重复", "repeat"),
    ("cancel", "cancel"), ("取消", "cancel"),
    ("approve", "approve"), ("通过", "approve"), ("deny", "deny"), ("拒绝", "deny"),
    ("yes", "yes"), ("read", "read"),
    ("option two", "option:2"), ("选三", "option:3"),
    ("stop three", "stop:3"), ("停止三号", "stop:3"), ("open five", "open:5"),
    ("new in dsh plugins", "new"), ("新建会话", "new"),
    ("banana", "unknown"), ("", "unknown"),
])
def test_match_intent(text, intent):
    assert match_intent(text) == intent
