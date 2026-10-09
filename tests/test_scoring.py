from stt.bench.scoring import (
    normalize, number_ok, numbers_in, term_hits, token_error_rate, tokens,
)


def test_normalize_lowercases_strips_punctuation_and_fullwidth():
    assert normalize("Rebase, onto  DEVELOP!") == "rebase onto develop"
    assert normalize("ＰＲ６５１２") == "pr6512"


def test_tokens_split_chinese_per_character_and_keep_latin_words():
    assert tokens("把 PR 6512 的 session 打开") == ["把", "pr", "6512", "的", "session", "打", "开"]


def test_token_error_rate():
    assert token_error_rate("a b c d", "a b c d") == 0.0
    assert token_error_rate("a b c d", "a x c") == 0.5
    assert token_error_rate("", "") == 0.0
    assert token_error_rate("", "x") == 1.0


def test_term_hits_ignore_case_hyphens_and_word_splits():
    assert term_hits("check dsh plugins hooks", ["dsh-plugins", "hooks"]) == 1.0
    assert term_hits("the session controller", ["sessionController"]) == 1.0
    assert term_hits("run PNPM install", ["pnpm", "worktree"]) == 0.5
    assert term_hits("anything", []) == 1.0


def test_short_term_does_not_match_inside_another_word():
    assert term_hits("please approve it", ["PR"]) == 0.0
    assert term_hits("open the pr now", ["PR"]) == 1.0


def test_numbers_as_digits_words_chinese_and_san():
    for heard in ["three", "3", "三", "san", "Three.", "第三个"]:
        assert number_ok(heard, 3), heard
    assert not number_ok("tree", 3)
    assert number_ok("option 2", 2)
    assert numbers_in("stop 三号") == [3]
