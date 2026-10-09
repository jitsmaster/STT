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
