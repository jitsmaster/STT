from stt.auth import load_or_create_secret, token_ok


def test_secret_is_created_once_and_reused(tmp_path):
    a = load_or_create_secret(tmp_path / "state")
    assert len(a) == 64 and load_or_create_secret(tmp_path / "state") == a


def test_a_too_short_secret_file_is_replaced(tmp_path):
    (tmp_path / "stt-secret").write_text("abc", encoding="utf-8")
    assert len(load_or_create_secret(tmp_path)) == 64


def test_token_ok():
    assert token_ok("s" * 40, "s" * 40)
    assert not token_ok("s" * 40, "t" * 40)
    assert not token_ok("s" * 40, "") and not token_ok("s" * 40, None)
