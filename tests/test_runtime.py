import asyncio

import pytest
from helpers import LOUD, QUIET, build, collect

from stt.cli import build_parser, main
from stt.runtime import list_input_devices, make_hotkey_handler, touch_stop_flag


def run(coro):
    return asyncio.run(coro)


def test_touch_stop_flag_creates_the_flag_in_a_new_state_dir_and_can_repeat(tmp_path):
    state = tmp_path / "a" / "tts"
    touch_stop_flag(state)
    touch_stop_flag(state)
    assert (state / "tts-stop.flag").is_file()


def test_a_second_hotkey_press_ends_the_utterance_instead_of_starting_another():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        press = make_hotkey_handler(pipe)
        press()
        assert pipe.state == "listening"
        pipe.feed(LOUD)
        pipe.feed(LOUD)
        press()
        events = await collect(q, 4)
        assert [e["type"] for e in events] == ["listening", "transcribing", "transcript", "idle"]
        assert calls.count("start") == 1
        press()
        assert pipe.state == "listening"  # idle again, so the next press starts a new window

    run(scenario())


def test_a_second_press_with_nothing_said_just_closes_the_window():
    async def scenario():
        pipe, bus, calls, engines = build()
        q = bus.subscribe()
        press = make_hotkey_handler(pipe)
        press()
        pipe.feed(QUIET)
        press()
        events = await collect(q, 2)
        assert [e["type"] for e in events] == ["listening", "idle"] and events[1]["reason"] == "no_speech"
        assert engines["command"].hints == []

    run(scenario())


def test_list_input_devices_shows_inputs_only_with_their_indexes():
    devices = [{"name": "Speakers", "max_input_channels": 0}, {"name": "Mic Array", "max_input_channels": 2}]
    assert list_input_devices(devices) == ["1: Mic Array"]


def test_the_parser_knows_serve_and_devices_and_rejects_the_rest():
    args = build_parser().parse_args(["serve", "--config", "c.toml", "--port", "9001"])
    assert (args.command, args.config, args.port) == ("serve", "c.toml", 9001)
    assert build_parser().parse_args(["devices"]).command == "devices"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["frobnicate"])


def test_a_bad_config_is_reported_on_stderr_with_exit_code_1(tmp_path, capsys):
    bad = tmp_path / "c.toml"
    bad.write_text("prot = 1\n", encoding="utf-8")
    assert main(["serve", "--config", str(bad)]) == 1
    assert "unknown config keys" in capsys.readouterr().err


def test_a_config_path_given_on_the_command_line_must_exist(tmp_path, capsys):
    assert main(["serve", "--config", str(tmp_path / "missing.toml")]) == 1
    assert "not found" in capsys.readouterr().err
