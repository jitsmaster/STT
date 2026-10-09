from stt.bench.intents import match_intent
from stt.bench.prompts import build_manifest, load_manifest, missing_samples, save_manifest


def test_manifest_has_the_agreed_counts_and_unique_ids():
    items = build_manifest()
    count = lambda kind: sum(1 for i in items if i["kind"] == kind)
    assert (count("command"), count("dictation"), count("mixed"), count("wake"), count("ambient")) == (25, 10, 5, 20, 3)
    assert len({i["id"] for i in items}) == len(items)


def test_every_command_prompt_matches_its_own_intent():
    for item in build_manifest():
        if item["kind"] == "command":
            assert match_intent(item["text"]) == item["intent"], item["id"]


def test_numbered_commands_carry_the_number():
    by_text = {i["text"]: i for i in build_manifest()}
    assert by_text["option two"]["number"] == 2
    assert by_text["停止三号"]["number"] == 3


def test_manifest_roundtrip_and_missing_samples(tmp_path):
    items = build_manifest()
    save_manifest(items, tmp_path / "manifest.json")
    assert load_manifest(tmp_path / "manifest.json") == items
    (tmp_path / (items[0]["id"] + ".wav")).write_bytes(b"x")
    todo = missing_samples(items, tmp_path)
    assert items[0] not in todo and len(todo) == len(items) - 1
