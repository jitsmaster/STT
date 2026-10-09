from .audio import SAMPLE_RATE


class WakeWatcher:
    """Feeds mic frames to a keyword spotter and says whether the keyword was heard in them."""

    def __init__(self, spotter):
        self.spotter = spotter
        self.stream = spotter.create_stream()

    def feed(self, frame) -> bool:
        self.stream.accept_waveform(SAMPLE_RATE, frame)
        hit = False
        while self.spotter.is_ready(self.stream):
            self.spotter.decode_stream(self.stream)
            if self.spotter.get_result(self.stream):
                hit = True
                self.spotter.reset_stream(self.stream)
        return hit

    def reset(self) -> None:
        self.stream = self.spotter.create_stream()
