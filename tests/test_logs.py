import logging

from stt.logs import install_token_redaction


def test_the_websocket_token_never_reaches_a_log_handler():
    """uvicorn logs the WebSocket URL, query string included, even with access_log=False."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("uvicorn.error")
    handler = Capture()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    secret = "ab12" * 16
    install_token_redaction()
    try:
        log.info('%s - "WebSocket %s" [accepted]', "127.0.0.1:5000", "/events?token=" + secret)
        log.info("plain message with token=" + secret + "&x=1")
    finally:
        log.removeHandler(handler)
    assert records and all(secret not in r for r in records)
    assert "/events?token=<redacted>" in records[0] and "x=1" in records[1]
