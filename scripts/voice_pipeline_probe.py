"""Clock-paced CPU audio sink for overlap experiments, without opening audio devices."""

from __future__ import annotations

import queue
import threading
import time


class PacedSink:
    """Consume CPU PCM duration on a separate thread; this is not physical playback."""

    def __init__(self) -> None:
        self.pending: queue.Queue = queue.Queue(maxsize=2)
        self.cancelled = threading.Event()
        self.origin = time.perf_counter()
        self.events: list[dict] = []
        self.thread = threading.Thread(target=self.consume, daemon=True)
        self.thread.start()

    def consume(self) -> None:
        """Record the same scheduling boundaries a blocking CPU PCM player would use."""
        while not self.cancelled.is_set():
            item = self.pending.get()
            if item is None:
                break
            name, pcm, rate, ready = item
            start = time.perf_counter() - self.origin
            cancelled = self.cancelled.wait(len(pcm) / rate)
            self.events.append(
                {
                    "audio": name,
                    "ready_seconds": ready,
                    "start_seconds": start,
                    "end_seconds": time.perf_counter() - self.origin,
                    "duration_seconds": len(pcm) / rate,
                    "cancelled": cancelled,
                }
            )

    def offer(self, name: str, pcm, rate: int) -> None:
        """Keep audio on CPU; never invoke a second GPU inference or retain GPU tensors."""
        self.pending.put((name, pcm, rate, time.perf_counter() - self.origin), timeout=10)

    def finish(self) -> list[dict]:
        """Drain the bounded queue; short benchmark clips complete within 60 seconds."""
        self.pending.put(None, timeout=10)
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            self.stop()
            raise TimeoutError("Paced audio sink did not finish")
        return self.events

    def stop(self) -> None:
        """Cancel a pending duration wait and release queued CPU audio."""
        self.cancelled.set()
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
        self.pending.put_nowait(None)
        self.thread.join(timeout=2)


def playback_schedule(arrivals: list[float], durations: list[float]) -> list[dict]:
    """Calculate optimistic playback gaps from measured CPU audio arrivals."""
    if len(arrivals) != len(durations):
        raise ValueError("Each audio arrival needs its own duration")
    scheduled, ended = [], 0.0
    for arrival, duration in zip(arrivals, durations):
        if duration <= 0 or arrival < 0:
            raise ValueError("Invalid audio timing")
        start = max(arrival, ended)
        gap = max(0.0, arrival - ended) if scheduled else 0.0
        scheduled.append(
            {
                "arrival": arrival,
                "start": start,
                "end": start + duration,
                "gap_before_seconds": gap,
                "duration_seconds": duration,
            }
        )
        ended = start + duration
    return scheduled
