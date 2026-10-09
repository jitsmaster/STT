import asyncio
import json
import sys

import websockets

from stt.config import default_state_dir


async def main(port):
    secret = (default_state_dir() / "stt-secret").read_text(encoding="utf-8").strip()
    async with websockets.connect("ws://127.0.0.1:" + str(port) + "/events?token=" + secret) as ws:
        async for message in ws:
            print(json.dumps(json.loads(message), ensure_ascii=False))


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8890))
