"""
Streaming audio windowing for live Morph detection.

Bridges WebSocket chunk accumulation and the file-oriented inference
pipeline: mono audio chunks (any sample rate) are accumulated at their
native rate, and each fixed-length window is resampled to the model's
native SR=16000 in a SINGLE `librosa.resample` call before being handed
to `detect_from_array`.

Why whole-window (not per-chunk) resampling: the training feature
extractor resampled each entire file once via ``librosa.load(..., sr=SR)``.
Resampling each WebSocket chunk independently produces edge discontinuities
at chunk boundaries that measurably perturb CQT/CQCC/pitch features and
can flip predictions (observed on real audio). Resampling a full window at
once reproduces the training-time signal path.

Default window/hop (4 s / 1 s) matches the Deep-Voice training-chunk
length and yields ~1 prediction/second after the first window fills.
"""

from __future__ import annotations

from typing import List, Optional

import librosa
import numpy as np

from ai_engine.features.extraction import SR

# Default streaming window configuration (seconds).
DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_HOP_SECONDS = 1.0


class AudioWindowBuffer:
    """
    Accumulate mono float32 audio chunks into fixed-length windows at SR.

    Parameters
    ----------
    window_seconds : float
        Length of each detection window (default 4.0).
    hop_seconds : float
        Advance between consecutive windows (default 1.0). Windows overlap
        when hop < window; a fresh window is emitted every `hop_seconds`
        of consumed audio.
    target_sr : int
        Sample rate the windows are produced at (and what the model needs).
        Default: ai_engine.features.extraction.SR (16000).
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        hop_seconds: float = DEFAULT_HOP_SECONDS,
        target_sr: int = SR,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if hop_seconds <= 0 or hop_seconds > window_seconds:
            raise ValueError("hop_seconds must be in (0, window_seconds]")
        if target_sr <= 0:
            raise ValueError("target_sr must be > 0")

        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.target_sr = int(target_sr)

        # Buffer lives in the incoming (native) sample domain.
        self._native_sr: Optional[int] = None
        self._buffer = np.zeros(0, dtype=np.float32)

    # -- Streaming API ---------------------------------------------------

    def add_chunk(self, samples: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Append a mono audio chunk and return any completed windows.

        Parameters
        ----------
        samples : np.ndarray
            1-D float32/int16 audio samples (mono).
        sr : int
            Sample rate of `samples`. The first non-empty chunk fixes the
            buffer's native rate; later chunks with a different rate are
            resampled to it before buffering.

        Returns
        -------
        list[np.ndarray]
            Every window completed by this chunk, as float32 mono at
            ``target_sr`` (ready for ``detect_from_array``). May be empty.
        """
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []

        if self._native_sr is None:
            self._native_sr = int(sr)
        elif int(sr) != self._native_sr:
            # Rare: mixed-rate stream. Match the established buffer rate.
            samples = librosa.resample(
                samples, orig_sr=int(sr), target_sr=self._native_sr
            ).astype(np.float32)

        native_sr = self._native_sr
        self._buffer = np.concatenate([self._buffer, samples])

        window_samples = round(self.window_seconds * native_sr)
        hop_samples = round(self.hop_seconds * native_sr)

        windows: List[np.ndarray] = []
        while len(self._buffer) >= window_samples:
            native_window = self._buffer[:window_samples]
            if native_sr != self.target_sr:
                # One-shot resample of the full window — same transform the
                # training extractor applied to whole files.
                native_window = librosa.resample(
                    native_window,
                    orig_sr=int(native_sr),
                    target_sr=self.target_sr,
                ).astype(np.float32)
            windows.append(native_window)
            self._buffer = self._buffer[hop_samples:]
        return windows

    def flush(self, min_seconds: float = 1.0) -> List[np.ndarray]:
        """
        Drain any trailing audio as a final, possibly shorter window.

        Used at the end of a stream so the last partial window is still
        classified. Returns [] if the remainder is shorter than
        ``min_seconds`` (to avoid classifying pure-silence tails as a
        meaningful window).
        """
        if self._native_sr is None:
            return []
        min_samples = round(min_seconds * self._native_sr)
        if len(self._buffer) < min_samples:
            return []

        window = self._buffer.copy()
        if self._native_sr != self.target_sr:
            window = librosa.resample(
                window,
                orig_sr=int(self._native_sr),
                target_sr=self.target_sr,
            ).astype(np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        return [window]

    # -- State -----------------------------------------------------------

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently buffered (at the native rate)."""
        return len(self._buffer)

    @property
    def buffered_seconds(self) -> float:
        if self._native_sr is None:
            return 0.0
        return len(self._buffer) / self._native_sr

    def reset(self) -> None:
        """Drop all buffered audio (e.g. on stream error)."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._native_sr = None