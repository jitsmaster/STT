import asyncio
import io
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field

from .audio import load_wav
from .auth import token_ok
from .pipeline import UnknownUtterance

LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}
MAX_WAV_BYTES = 10 * 1024 * 1024
Kind = Literal["command", "dictation"]
Lang = Literal["en", "zh"]


def host_ok(value) -> bool:
    """True when a Host header names a loopback address (any port)."""
    host = (value or "").strip().lower()
    if host.startswith("["):
        host = host[1 : host.find("]")] if "]" in host else ""
    elif host.count(":") == 1:
        host = host.split(":")[0]
    return host in LOOPBACK_NAMES


class ListenBody(BaseModel):
    context: dict[str, Any] | None = None
    window_s: float | None = Field(default=None, gt=0, le=60)


class RetranscribeBody(BaseModel):
    id: str
    kind: Kind = "dictation"
    language: Lang | None = None


async def _until_disconnect(ws: WebSocket) -> None:
    try:
        while True:
            if (await ws.receive())["type"] == "websocket.disconnect":
                return
    except Exception:
        return


def create_app(pipeline, bus, secret, info) -> FastAPI:
    app = FastAPI(title="stt", docs_url=None, redoc_url=None, openapi_url=None)

    def host_guard(request: Request):
        if not host_ok(request.headers.get("host")):
            raise HTTPException(403, "host not allowed")

    def guard(request: Request):
        host_guard(request)
        auth = request.headers.get("authorization", "")
        if not (auth[:7].lower() == "bearer " and token_ok(secret, auth[7:].strip())):
            raise HTTPException(401, "missing or wrong token")

    @app.get("/health", dependencies=[Depends(host_guard)])
    async def health():
        return {"ok": True, "version": info.get("version"), "state": pipeline.state}

    @app.post("/listen", dependencies=[Depends(guard)])
    async def listen(body: ListenBody | None = None):
        body = body or ListenBody()
        return {"started": pipeline.trigger("api", body.context, body.window_s)}

    @app.post("/finish", dependencies=[Depends(guard)])
    async def finish():
        return {"finished": pipeline.finish()}

    @app.post("/cancel", dependencies=[Depends(guard)])
    async def cancel():
        return {"cancelled": pipeline.cancel()}

    @app.post("/transcribe", dependencies=[Depends(guard)])
    async def transcribe(request: Request, kind: Kind = "command", language: Lang | None = None):
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_WAV_BYTES:
            raise HTTPException(413, "audio too large")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_WAV_BYTES:
                raise HTTPException(413, "audio too large")
            chunks.append(chunk)
        try:
            audio = load_wav(io.BytesIO(b"".join(chunks)))
        except Exception:
            raise HTTPException(400, "body is not a readable WAV file")
        try:
            return await pipeline.transcribe_audio(audio, kind, language)
        except Exception as exc:
            raise HTTPException(502, "engine failed: " + str(exc))

    @app.post("/retranscribe", dependencies=[Depends(guard)])
    async def retranscribe(body: RetranscribeBody):
        try:
            return await pipeline.retranscribe(body.id, body.kind, body.language)
        except UnknownUtterance:
            raise HTTPException(404, "unknown or expired utterance id")
        except Exception as exc:
            raise HTTPException(502, "engine failed: " + str(exc))

    @app.websocket("/events")
    async def events(ws: WebSocket):
        if not host_ok(ws.headers.get("host")) or not token_ok(secret, ws.query_params.get("token", "")):
            await ws.close(code=1008)
            return
        await ws.accept()
        queue = bus.subscribe()
        reader = asyncio.create_task(_until_disconnect(ws))
        try:
            await ws.send_json({"type": "hello", **info, "state": pipeline.state})
            while True:
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({getter, reader}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    break
                await ws.send_json(getter.result())
        except Exception:
            pass  # the client went away mid-send
        finally:
            reader.cancel()
            bus.unsubscribe(queue)

    return app
