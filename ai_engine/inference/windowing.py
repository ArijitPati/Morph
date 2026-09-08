"""
Windowing utilities for Morph real-time inference.

Splits continuous audio into short fixed-length windows (3–4 s)
for per-window 132-feature extraction and XGBoost inference.

Preserves the existing 132-feature pipeline — this module only
handles time-domain slicing and boundary bookkeeping.

Typical usage:

    from ai_engine.inference.windowing import slice_audio

    windows = slice_audio(y, sr=16000, window_sec=4.0, hop_sec=4.0)
    for start, end in windows:
        chunk = y[start:end]
        ...

For streaming (WebSocket), use StreamingBuffer:

    from ai_engine.inference.windowing import StreamingBuffer

    buf = StreamingBuffer(sr=16000, window_sec=4.0, hop_sec=4.0)
    buf.append(new_samples)
    while buf.has_window():
        chunk, meta = buf.next_window()
        # run inference on chunk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_engine.features.extraction import SR as DEFAULT_SR


DEFAULT_WINDOW_SEC: float = 4.0
DEFAULT_HOP_SEC: float = 4.0
MIN_WINDOW_SEC: float = 1.0  # partial tail shorter than this is dropped


@dataclass(frozen=True)
class WindowSlice:
    """One non-overlapping (or overlapping) slice of audio."""

    index: int  # 0-based
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float
    is_partial: bool


def get_window_boundaries(
    total_samples: int,
    sr: int = DEFAULT_SR,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float | None = None,
    min_window_sec: float = MIN_WINDOW_SEC,
    include_partial: bool = True,
) -> list[WindowSlice]:
    """
    Compute window boundaries for a fixed-length audio array.

    Parameters
    ----------
    total_samples : int
        Total number of PCM samples.
    sr : int
        Sample rate (default 16000).
    window_sec : float
        Window length in seconds (default 4.0).
    hop_sec : float | None
        Hop / stride in seconds.  If None, defaults to window_sec
        (non-overlapping).  Use 2.0 for 50 % overlap with 4 s windows.
    min_window_sec : float
        Minimum duration for a trailing partial window to be kept.
        Partial tails shorter than this are dropped.
    include_partial : bool
        Whether to include the trailing partial window.

    Returns
    -------
    list[WindowSlice]
    """
    if hop_sec is None:
        hop_sec = window_sec

    window_samples = int(round(window_sec * sr))
    hop_samples = int(round(hop_sec * sr))
    min_window_samples = int(round(min_window_sec * sr))

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_sec and hop_sec must be > 0")

    slices: list[WindowSlice] = []
    idx = 0
    start = 0

    while start + window_samples <= total_samples:
        end = start + window_samples
        slices.append(
            WindowSlice(
                index=idx,
                start_sample=start,
                end_sample=end,
                start_sec=round(start / sr, 4),
                end_sec=round(end / sr, 4),
                duration_sec=window_sec,
                is_partial=False,
            )
        )
        idx += 1
        start += hop_samples

    # Trailing partial
    if include_partial and start < total_samples:
        remaining = total_samples - start
        # Only keep if the remaining segment is long enough
        # NOTE: for non-overlapping case, start already moved; remaining < window
        if remaining >= min_window_samples:
            duration = remaining / sr
            slices.append(
                WindowSlice(
                    index=idx,
                    start_sample=start,
                    end_sample=total_samples,
                    start_sec=round(start / sr, 4),
                    end_sec=round(total_samples / sr, 4),
                    duration_sec=duration,
                    is_partial=True,
                )
            )

    return slices


def slice_audio(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float | None = None,
    min_window_sec: float = MIN_WINDOW_SEC,
    include_partial: bool = True,
) -> list[tuple[int, int]]:
    """
    Convenience wrapper — returns (start, end) sample pairs.

    Kept for backwards-compat with diagnostic scripts that used
    manual slicing.  Prefer get_window_boundaries() for richer
    metadata.
    """
    slices = get_window_boundaries(
        len(y),
        sr=sr,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_window_sec=min_window_sec,
        include_partial=include_partial,
    )
    return [(s.start_sample, s.end_sample) for s in slices]


# ---------------------------------------------------------------------------
# Streaming buffer — accumulates PCM chunks and emits fixed windows
# ---------------------------------------------------------------------------


class StreamingBuffer:
    """
    Accumulates streaming PCM and emits fixed-length windows.

    Thread-agnostic (caller handles locking).  Works for the WebSocket
    path where audio_chunk messages arrive incrementally.

    Example
    -------
    buf = StreamingBuffer(sr=16000, window_sec=4.0, hop_sec=4.0)
    buf.append(chunk_from_websocket)   # np.ndarray float32 mono
    while buf.has_window():
        window, meta = buf.next_window()
        # inference on window
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        window_sec: float = DEFAULT_WINDOW_SEC,
        hop_sec: float | None = None,
        min_window_sec: float = MIN_WINDOW_SEC,
    ) -> None:
        if hop_sec is None:
            hop_sec = window_sec
        self.sr = sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.min_window_sec = min_window_sec
        self.window_samples = int(round(window_sec * sr))
        self.hop_samples = int(round(hop_sec * sr))
        self.min_window_samples = int(round(min_window_sec * sr))
        self._buf = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0  # samples ever emitted
        self._window_index = 0

    def append(self, samples: np.ndarray) -> None:
        """Append new mono PCM samples (float32, any amplitude)."""
        if samples.size == 0:
            return
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, arr])

    def has_window(self) -> bool:
        """True if at least one full window is ready."""
        return len(self._buf) >= self.window_samples

    def has_partial(self) -> bool:
        """True if a tail partial window meets min duration."""
        return 0 < len(self._buf) < self.window_samples and len(self._buf) >= self.min_window_samples

    def pending_samples(self) -> int:
        return len(self._buf)

    def pending_sec(self) -> float:
        return len(self._buf) / self.sr

    def total_consumed_sec(self) -> float:
        return self._total_consumed / self.sr

    def next_window(self) -> tuple[np.ndarray, WindowSlice] | None:
        """
        Consume and return the next full window.

        Returns None if not enough samples buffered.
        """
        if not self.has_window():
            return None

        chunk = self._buf[: self.window_samples].copy()
        start_sample = self._total_consumed
        end_sample = start_sample + self.window_samples

        meta = WindowSlice(
            index=self._window_index,
            start_sample=start_sample,
            end_sample=end_sample,
            start_sec=round(start_sample / self.sr, 4),
            end_sec=round(end_sample / self.sr, 4),
            duration_sec=self.window_sec,
            is_partial=False,
        )

        # Slide by hop
        self._buf = self._buf[self.hop_samples :]
        self._total_consumed += self.hop_samples
        self._window_index += 1

        return chunk, meta

    def flush_partial(self) -> tuple[np.ndarray, WindowSlice] | None:
        """
        Emit the trailing partial window if it meets min duration.

        Call at end-of-stream (e.g. caller hangs up).
        """
        if not self.has_partial():
            return None
        chunk = self._buf.copy()
        n = len(chunk)
        start_sample = self._total_consumed
        end_sample = start_sample + n
        meta = WindowSlice(
            index=self._window_index,
            start_sample=start_sample,
            end_sample=end_sample,
            start_sec=round(start_sample / self.sr, 4),
            end_sec=round(end_sample / self.sr, 4),
            duration_sec=n / self.sr,
            is_partial=True,
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self._total_consumed += n
        self._window_index += 1
        return chunk, meta

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0
        self._window_index = 0
