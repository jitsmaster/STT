import numpy as np
from fakes import ScriptEndpointer

from stt.core import ListenerCore
from stt.endpointers import EnergyEndpointer

FRAME = np.zeros(1600, dtype="float32")


def make(pattern, **kw):
    kw = {"no_speech_s": 0.3, "silence_end_s": 0.2, "max_s": 1.0, "min_speech_s": 0.15, **kw}
    return ListenerCore(ScriptEndpointer(pattern), **kw)


def feed(c, n):
    out = []
    for _ in range(n):
        out += c.feed(FRAME)
    return out


def test_speech_then_silence_yields_one_utterance_with_everything_heard():
    c = make("SSS..")
    assert c.trigger("hotkey", {"target": "s1"}) == [{"type": "listening", "source": "hotkey", "context": {"target": "s1"}}]
    (ev,) = feed(c, 5)
    assert ev["type"] == "utterance" and ev["reason"] == "silence"
    assert len(ev["audio"]) == 5 * 1600 and ev["context"] == {"target": "s1"} and ev["source"] == "hotkey"
    assert c.state == "idle"


def test_no_speech_inside_the_window_goes_idle_and_a_custom_window_wins():
    c = make("...")
    c.trigger("api")
    assert [(e["type"], e["reason"]) for e in feed(c, 3)] == [("idle", "no_speech")]
    c = make("..")
    c.trigger("api", window_s=0.2)
    assert [e["reason"] for e in feed(c, 2)] == ["no_speech"]


def test_a_click_is_too_short_to_be_an_utterance():
    c = make("S..")
    c.trigger("api")
    assert [e["reason"] for e in feed(c, 3)] == ["too_short"]


def test_a_long_monologue_is_cut_at_the_maximum():
    c = make("S" * 20, max_s=0.4)
    c.trigger("api")
    (ev,) = feed(c, 4)
    assert ev["type"] == "utterance" and ev["reason"] == "max"


def test_feeding_while_idle_and_triggering_while_listening_do_nothing():
    c = make("SS")
    assert feed(c, 2) == []
    c.trigger("api")
    assert c.trigger("hotkey") == [] and c.endpointer.resets == 1


def test_finish_ends_early_and_cancel_discards():
    c = make("SSS")
    c.trigger("hotkey")
    feed(c, 3)
    (ev,) = c.finish()
    assert ev["type"] == "utterance" and ev["reason"] == "finish"
    assert c.finish() == []
    c.trigger("hotkey")
    assert [(e["type"], e["reason"]) for e in c.cancel()] == [("idle", "cancelled")]
    assert c.cancel() == []


def test_energy_endpointer_uses_loudness():
    e = EnergyEndpointer(0.05)
    assert e.is_speech(np.full(1600, 0.2, dtype="float32")) and not e.is_speech(FRAME)


def test_leading_silence_is_trimmed_to_a_short_pre_roll_so_it_cannot_dilute_quiet_speech():
    c = make("....." + "SSS" + "..", no_speech_s=1.0)
    c.trigger("api")
    (ev,) = feed(c, 10)
    assert ev["reason"] == "silence"
    assert len(ev["audio"]) == (3 + 3 + 2) * 1600  # 0.3 s pre-roll, the speech, the trailing silence


# --- end-word support: trailing burst check -------------------------------------------------------------------------

def make_end(pattern, **kw):
    kw = {"no_speech_s": 1.0, "silence_end_s": 15.0, "max_s": 60.0, "end_check": True, **kw}
    return make(pattern, **kw)


def feed_each(c, n):
    return [c.feed(FRAME) for _ in range(n)]


def self_trigger(c):
    c.trigger("hotkey")
    return c


def endchecks(per_frame):
    return [(i, e) for i, evs in enumerate(per_frame) for e in evs if e["type"] == "endcheck"]


def test_a_short_isolated_burst_is_checked_once_after_end_gap_s_of_silence_and_the_window_stays_open():
    c = make_end(".." + "SS" + "." * 8)
    c.trigger("hotkey")
    found = endchecks(feed_each(c, 12))
    assert [i for i, _ in found] == [9]  # 4 frames up to the burst's end, then the 6th silent frame (0.6 s)
    ev = found[0][1]
    assert ev == {"type": "endcheck", "audio": ev["audio"], "burst": 1}
    assert len(ev["audio"]) == (2 + 2) * 1600  # 0.2 s of lead-in (the pre-roll would allow 0.3 s) plus the burst
    assert c.state == "listening"


def test_the_burst_audio_is_just_the_burst_with_at_most_the_pre_roll_in_front():
    c = make_end("SSSS" + "." * 5 + "SS" + "." * 7)
    c.trigger("hotkey")
    frames = [np.full(1600, i + 1, dtype="float32") for i in range(18)]
    (ev,) = [e for f in frames for e in c.feed(f) if e["type"] == "endcheck"]
    assert ev["burst"] == 2
    assert [int(v) for v in ev["audio"][::1600]] == [7, 8, 9, 10, 11]  # 3 frames of pre-roll, then the 2 burst frames


def test_bursts_outside_015_to_15_seconds_of_speech_are_never_checked():
    assert endchecks(feed_each(self_trigger(make_end("S" + "." * 9)), 10)) == []  # 0.1 s: a click
    assert endchecks(feed_each(self_trigger(make_end("S" * 16 + "." * 9)), 25)) == []  # 1.6 s: dictation
    assert len(endchecks(feed_each(self_trigger(make_end("S" * 2 + "." * 8)), 10))) == 1  # 0.2 s: a word


def test_every_burst_is_checked_once_with_its_own_id():
    c = make_end("SS" + "." * 10 + "SS" + "." * 10)
    c.trigger("hotkey")
    found = endchecks(feed_each(c, 24))
    assert [(i, e["burst"]) for i, e in found] == [(7, 1), (19, 2)]


def test_a_pause_of_half_a_second_before_a_word_makes_it_a_burst_of_its_own():
    c = make_end("SSSS" + "." * 5 + "SS" + "." * 7)
    c.trigger("hotkey")
    found = endchecks(feed_each(c, 18))
    assert [e["burst"] for _, e in found] == [2]  # burst 1 was followed by only 0.5 s of silence: no check
    assert len(found[0][1]["audio"]) == (3 + 2) * 1600


def test_words_with_short_gaps_between_them_are_one_burst_and_a_trailing_word_is_not_checked_alone():
    c = make_end("SS" + ".." + "SS" + "." * 8)  # "pass it" ... "over" with only 0.2 s between them
    c.trigger("hotkey")
    found = endchecks(feed_each(c, 14))
    assert [e["burst"] for _, e in found] == [1] and len(found[0][1]["audio"]) == 6 * 1600


def test_end_checks_are_off_unless_asked_for_and_leave_the_events_alone():
    c = make("SS" + "." * 12, silence_end_s=15.0, max_s=60.0, no_speech_s=1.0)
    c.trigger("hotkey")
    assert endchecks(feed_each(c, 14)) == []


def test_an_end_word_alone_is_the_same_as_no_speech():
    c = make_end("SS" + "." * 8)
    c.trigger("hotkey", {"target": "s1"})
    ((_, ev),) = endchecks(feed_each(c, 10))
    out = c.apply_end_word(ev["burst"])
    assert out == [{"type": "idle", "reason": "no_speech", "source": "hotkey", "context": {"target": "s1"}}]
    assert c.state == "idle"


def test_speech_before_the_end_word_becomes_an_utterance_without_the_burst():
    c = make_end("SSSS" + "." * 5 + "SS" + "." * 7)
    c.trigger("hotkey", {"target": "s1"})
    ((_, ev),) = endchecks(feed_each(c, 18))
    (out,) = c.apply_end_word(ev["burst"])
    assert out["type"] == "utterance" and out["reason"] == "endword"
    assert out["source"] == "hotkey" and out["context"] == {"target": "s1"}
    assert len(out["audio"]) == (4 + 5) * 1600  # the first burst and the pause, not the end word or what followed it
    assert c.state == "idle"


def test_a_blip_before_the_end_word_is_too_short():
    c = make_end("S" + "." * 5 + "SS" + "." * 7)
    c.trigger("hotkey")
    ((_, ev),) = endchecks(feed_each(c, 15))
    assert [e["reason"] for e in c.apply_end_word(ev["burst"])] == ["too_short"]


def test_a_stale_or_unknown_burst_id_changes_nothing_and_the_window_goes_on():
    c = make_end("SS" + "." * 7 + "S" + "." * 3)
    c.trigger("hotkey")
    ((_, ev),) = endchecks(feed_each(c, 10))  # the check fired after 0.6 s; then the user spoke again
    feed_each(c, 3)
    assert c.apply_end_word(ev["burst"]) == [] and c.apply_end_word(99) == []
    assert c.state == "listening"
    assert [e["type"] for e in c.finish()] == ["utterance"]  # nothing was dropped


def test_a_burst_that_was_not_checked_yet_cannot_be_applied_and_neither_can_one_from_an_earlier_window():
    c = make_end("SS" + "." * 3 + "SS" + "." * 8 + "SS" + "." * 8)
    c.trigger("hotkey")
    feed_each(c, 5)
    assert c.apply_end_word(1) == []
    c.cancel()
    c.trigger("hotkey")
    ((_, ev),) = endchecks(feed_each(c, 4 + 6))
    assert ev["burst"] > 1 and c.apply_end_word(1) == []  # ids keep counting across windows
    assert c.state == "listening"


def test_applying_an_end_word_while_not_listening_does_nothing():
    c = make_end("SS" + "." * 8)
    assert c.apply_end_word(1) == []
    c.trigger("hotkey")
    ((_, ev),) = endchecks(feed_each(c, 10))
    c.cancel()
    assert c.apply_end_word(ev["burst"]) == []
