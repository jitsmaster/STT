import logging
import re

_TOKEN = re.compile(r"(token=)[^&\s\"']+")


class TokenRedactor(logging.Filter):
    """uvicorn logs the WebSocket URL with its query string even when access_log is off, and the query holds the secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if "token=" in message:
            record.msg, record.args = _TOKEN.sub(r"\1<redacted>", message), ()
        return True


def install_token_redaction() -> None:
    """Redact tokens on the loggers and on their handlers (a filter on a logger skips records from child loggers)."""
    redactor = TokenRedactor()
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "stt"):
        logger = logging.getLogger(name)
        for target in [logger, *logger.handlers]:
            if not any(isinstance(f, TokenRedactor) for f in target.filters):
                target.addFilter(redactor)
