"""
Tests for ai_engine.inference.streaming — window buffering + resampling.
"""

from pathlib import Path

import librosa
import numpy as np
import pytest

from ai_engine.features.extraction import SR
from ai_engine.inference.pipeline import detect_from_array
from ai_engine.inference.streaming import AudioWindowBuffer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestWindowing:
    def test_emits_window_after_enough_audio(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=0.5, target_sr=SR)
        chunk = np.zeros(SR // 2, dtype=np.float32)  # 0.5 s
        assert buf.add_chunk(chunk, SR) == []
        windows = buf.add_chunk(chunk, SR)  # now 1.0 s
        assert len(windows) == 1
        assert windows[0].shape == (SR,)

    def test_hop_advances_buffer(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=0.5, target_sr=SR)
        for _ in range(4):  # 2.0 s total, hop 0.5
            buf.add_chunk(np.zeros(SR // 2, dtype=np.float32), SR)
        # 4 chunks -> after first window (2 chunks) buffer drops by hop
        # (0.5 s), so we should have produced ceil((2.0 - 1.0)/0.5) = 2 windows
        assert buf.buffered_samples == SR // 2

    def test_non_overlapping(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=1.0, target_sr=SR)
        windows = buf.add_chunk(np.zeros(2 * SR, dtype=np.float32), SR)
        assert len(windows) == 2
        assert buf.buffered_samples == 0

    def test_flush_returns_trailing_partial_window(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=1.0, target_sr=SR)
        buf.add_chunk(np.zeros(SR // 2, dtype=np.float32), SR)
        assert buf.flush(min_seconds=1.0) == []  # below min length, not drained
        assert buf.buffered_samples == SR // 2
        out = buf.flush(min_seconds=0.3)
        assert len(out) == 1
        assert out[0].shape == (SR // 2,)
        assert buf.buffered_samples == 0

    def test_flush_discards_tiny_tail(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=1.0, target_sr=SR)
        buf.add_chunk(np.zeros(100, dtype=np.float32), SR)
        assert buf.flush(min_seconds=1.0) == []


class TestResampling:
    def test_chunk_at_48k_is_resampled_to_16k(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=1.0, target_sr=SR)
        windows = buf.add_chunk(np.zeros(48000, dtype=np.float32), 48000)
        assert any(w.shape == (SR,) for w in windows)
        assert buf.target_sr == 16000

    def test_resampled_energy_preserved(self):
        # A 1 kHz sine at 48k should still be non-silent after resample.
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=1.0, target_sr=SR)
        t = np.arange(16000) / SR
        sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        windows = buf.add_chunk(sine, SR)
        assert windows[0].shape == (SR,)
        rms = np.sqrt(np.mean(windows[0] ** 2))
        assert 0.2 < rms < 0.6

    def test_empty_chunk_returns_nothing(self):
        buf = AudioWindowBuffer(window_seconds=1.0, hop_seconds=1.0, target_sr=SR)
        assert buf.add_chunk(np.zeros(0, dtype=np.float32), SR) == []

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            AudioWindowBuffer(window_seconds=0)
        with pytest.raises(ValueError):
            AudioWindowBuffer(window_seconds=2.0, hop_seconds=3.0)


class TestDetectEndToEnd:
    def test_window_predicts_known_sample(self):
        # A real 16k sample detection through the streaming buffer.
        wav = PROJECT_ROOT / "data/raw/gary_stafford/real/gs_00202.wav"
        y, sr = librosa.load(str(wav), sr=SR, mono=True)
        buf = AudioWindowBuffer(window_seconds=4.0, hop_seconds=4.0, target_sr=SR)
        windows = buf.add_chunk(y, sr)
        assert windows
        result = detect_from_array(windows[0], sr=SR)
        assert result.label == 0  # gary real -> REAL
        assert result.label_str == "REAL"