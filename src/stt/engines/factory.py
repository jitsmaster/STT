def make_engine(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "faster-whisper":
        from .whisper import FasterWhisperEngine

        return FasterWhisperEngine(arg or "large-v3")
    if kind == "sensevoice":
        from .sensevoice import SenseVoiceEngine

        return SenseVoiceEngine(arg or "models/sensevoice")
    raise ValueError("unknown engine " + repr(spec))


def build_engines(config) -> dict:
    command = make_engine(config.command_engine)
    same = config.dictation_engine in (None, config.command_engine)
    return {"command": command, "dictation": command if same else make_engine(config.dictation_engine)}
