import hmac
import secrets
from pathlib import Path


def load_or_create_secret(state_dir, name="stt-secret") -> str:
    path = Path(state_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= 32:
            return value
    value = secrets.token_hex(32)
    path.write_text(value, encoding="utf-8")
    return value


def token_ok(expected: str, given) -> bool:
    return bool(given) and hmac.compare_digest(expected.encode(), str(given).encode())
