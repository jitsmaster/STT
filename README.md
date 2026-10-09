# STT

Standalone local speech-to-text project: an engine bake-off harness (faster-whisper, SenseVoiceSmall, wake word 码农) and a background daemon that serves transcripts over a small authenticated HTTP/WebSocket API on 127.0.0.1.

Dev commands run from the project root with the venv: `.\.venv\Scripts\python -m pytest`

## Models (not committed, `models/` is gitignored)

### SenseVoiceSmall (sherpa-onnx)

Source: https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html (Python API: https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html).

- Archive: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2` (int8, about 160 MB)
- URL: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
- Extract and rename the folder to `models/sensevoice` (it holds `model.int8.onnx`, `tokens.txt`, `test_wavs/`).
- API (sherpa-onnx 1.13.8): `OfflineRecognizer.from_sense_voice(model, tokens, num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", debug=False, provider="cpu", language="", use_itn=False, rule_fsts="", rule_fars="", hr_dict_dir="", hr_rule_fsts="", hr_lexicon="")`. `stream.result` has `text` (tags already stripped), `lang` (`<|en|>`), `emotion`, `event`.

### Wake word model (sherpa-onnx keyword spotting)

Source: https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html

- Archive: `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2` (about 38 MB), from https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2, extracted to `models/kws`.
- Files (chunk-16 = 320 ms latency, chunk-8 = 160 ms): `encoder-epoch-13-avg-2-chunk-{16,8}-left-64[.int8].onnx`, `decoder-epoch-13-avg-2-chunk-{16,8}-left-64.onnx` (fp32 only), `joiner-epoch-13-avg-2-chunk-{16,8}-left-64[.int8].onnx`, `tokens.txt`, `en.phone`, `test_wavs/`. `stt.bench.wake.model_files` uses the chunk-16 int8 encoder and joiner with the fp32 decoder.
- `keywords.txt` is generated from `keywords_raw.txt` (`码农 @码农`; the original keyword after `@` must not contain spaces). Needs `pip install -e ".[kws-tools]"` (sentencepiece, pypinyin):

```powershell
.\.venv\Scripts\sherpa-onnx-cli text2token --tokens models\kws\tokens.txt --tokens-type phone+ppinyin --lexicon models\kws\en.phone keywords_raw.txt keywords.txt
```

  Result: `m ǎ n óng @码农`. (Set `PYTHONIOENCODING=utf-8` when redirecting.)
- API (sherpa-onnx 1.13.8): `KeywordSpotter(tokens, encoder, decoder, joiner, keywords_file, num_threads=2, sample_rate=16000, feature_dim=80, max_active_paths=4, keywords_score=1.0, keywords_threshold=0.25, num_trailing_blanks=1, provider="cpu", device=0)`, matching `make_spotter`.
- Smoke test done with the archive's own `test_wavs` and `keywords.txt` (every clip triggered); the 码农 clips come from the recorded sample set (Task 10).

## Daemon

A background process that listens (hotkey, optionally the 码农 wake word), endpoints speech with Silero VAD, transcribes, and serves the result over an authenticated loopback HTTP/WebSocket API. The contract is in [API.md](API.md).

1. Copy `config.example.toml` to `config.toml` (gitignored) and edit it. Pick the microphone: `.\.venv\Scripts\python -m stt devices` lists the inputs; set `device = "<index or name fragment>"`.
2. Needed in `models/`: `silero_vad.onnx` (https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx; without it the daemon falls back to a loudness gate and warns), the SenseVoice folder if used, and for the wake word the `kws` folder (see above).
3. Run: `.\scripts\run-stt.ps1` (or `.\.venv\Scripts\python -m stt serve --config config.toml [--port N]`).
4. Watch the events of a running daemon: `.\.venv\Scripts\python scripts\watch_events.py [port]`.
5. Default hotkey `ctrl+alt+v`: press to listen, press again to end the utterance. The default port is 8890; the secret is created in `<state_dir>\stt-secret`.
