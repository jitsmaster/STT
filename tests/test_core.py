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
