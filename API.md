# STT daemon API

Base URL `http://127.0.0.1:8890` (loopback only). Every request must carry a loopback `Host` header (`127.0.0.1`, `localhost` or `[::1]`, any port), otherwise 403 (HTTP) or close code 1008 (WebSocket).

## Authentication

Shared secret in `<state_dir>\stt-secret`, created on first start (64 hex characters). `state_dir` is `DSH_TTS_STATE_DIR`, else `DSH_HOME\tts`, else `~\.dsh\tts`: the same folder the plugin uses. HTTP: `Authorization: Bearer <secret>`. WebSocket: `?token=<secret>`. Only `GET /health` is open.

## Endpoints

| Method and path | Body or query | Success | Errors |
|---|---|---|---|
| GET /health | none | `{"ok": true, "version": "...", "state": "idle"|"listening"}` | 403 |
| POST /listen | `{"context": {...}?, "window_s": 0 < x <= 60 ?}` | `{"started": true}`; `false` when already listening | 401, 403, 422 |
| POST /finish | none | `{"finished": bool}`: end the utterance now (what "done" and the second hotkey press do) | 401, 403 |
| POST /cancel | none | `{"cancelled": bool}`: stop listening and discard | 401, 403 |
| POST /transcribe?kind=command|dictation&language=en|zh | raw WAV bytes, at most 10 MB | `{"text", "language", "confidence", "engine", "seconds"}` | 400 not a WAV, 401, 403, 413, 502 engine failed |
| POST /retranscribe | `{"id", "kind": "dictation"?, "language"?}` | a `transcript` event with `revision: 2` (also published on /events) | 401, 403, 404 unknown or expired id (only the last 5 utterances are kept), 502 |
| WS /events?token= | none | stream below | closed with 1008 when refused |

`context` is opaque JSON that comes back unchanged in the events of that listening window (the plugin uses it to carry the target session of a reply window). `window_s` is how long to wait for speech to begin (default 5 s, config `no_speech_s`). The hotkey and the wake word start windows with `context: null`.

## Events (WebSocket, one JSON object per message)

```json
{"type": "hello", "version": "0.0.1", "hotkey": "ctrl+alt+v", "wake": false, "engines": {"command": "...", "dictation": "..."}, "state": "idle"}
{"type": "listening", "source": "hotkey|wake|api", "context": null}
{"type": "transcribing", "context": null}
{"type": "transcript", "id": "a1b2c3d4", "text": "status", "language": "en", "confidence": 0.93, "engine": "...", "seconds": 0.4, "source": "hotkey", "context": null, "revision": 1}
{"type": "idle", "reason": "done|no_speech|too_short|cancelled|error", "context": null}
{"type": "error", "message": "...", "context": null}
```

A normal utterance is `listening`, `transcribing`, `transcript`, `idle(done)`. No speech inside the window is `listening`, `idle(no_speech)`. `transcript.text` can be empty (the engine heard nothing usable): ignore it. `revision` is 1 for the fast first pass and 2 after `/retranscribe`; `id` stays the same. `confidence` is 0.0 when the engine reports none: faster-whisper derives it from the segments (exp of the mean average log-probability, times one minus the mean no-speech probability; 0.0 when nothing was kept), SenseVoice reports none. Whisper output in Traditional Chinese is converted to Simplified, which is what the command grammar uses. A client that stops reading loses events instead of blocking the daemon, so reconnect and rely on `hello.state`.

## Behaviour a client must know

- Barge-in: when listening starts the daemon creates `tts-stop.flag` in the state dir; the Kokoro client stops the audio it is playing within about 50 ms. A read-out that is already queued behind it (waiting for the playback mutex) still plays afterwards, so a client that wants silence must hold its own queue while `listening` or `transcribing`.
- Earcons: a rising beep when listening starts, a lower beep when the utterance ends, two falling beeps on cancel or no speech.
- Microphone: opened when listening starts and closed afterwards (unless the wake word is on). The first ~0.2 s after the start beep can be lost, so begin speaking after the beep.
- Hotkey: a press starts listening; a press while listening ends the utterance.
- Dictation accuracy first: use `/retranscribe` on the `id` of a prompt or approval text to get the accurate engine's version before reading it back.
