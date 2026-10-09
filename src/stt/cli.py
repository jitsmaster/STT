import argparse
import asyncio
import dataclasses
import logging
import sys

from .config import load_config, validate


def build_parser():
    ap = argparse.ArgumentParser(prog="stt", description="Local speech-to-text daemon")
    sub = ap.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the daemon")
    serve.add_argument("--config", help="config file (default: config.toml when it exists); a path given here must exist")
    serve.add_argument("--port", type=int)
    sub.add_parser("devices", help="list microphone input devices")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "devices":
            import sounddevice

            from .runtime import list_input_devices

            print("\n".join(list_input_devices(sounddevice.query_devices())))
            return 0
        config = load_config(args.config or "config.toml", required=bool(args.config))
        if args.port is not None:
            config = validate(dataclasses.replace(config, port=args.port))
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        from .runtime import run_daemon

        asyncio.run(run_daemon(config))
    except (ValueError, OSError) as exc:
        print("stt: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0
