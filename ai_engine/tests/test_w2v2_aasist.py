"""
Unit tests for W2V2-AASIST wrapper — preprocessing, fixed length, padding, score conversion.
"""
import numpy as np
import pytest

from ai_engine.inference.w2v2_aasist import TARGET_LEN, TARGET_SR, W2V2AASISTModel, pad


def test_pad_short_tiles():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = pad(x, max_len=7)
    # Should tile: [1,2,3,1,2,3,1]
    assert out.shape[0] == 7
    assert np.allclose(out, np.array([1, 2, 3, 1, 2, 3, 1]))

def test_pad_long_truncates():
    x = np.arange(100, dtype=np.float32)
    out = pad(x, max_len=10)
    assert out.shape[0] == 10
    assert np.allclose(out, np.arange(10))

def test_pad_exact():
    x = np.arange(TARGET_LEN, dtype=np.float32)
    out = pad(x, max_len=TARGET_LEN)
    assert out.shape[0] == TARGET_LEN
    assert np.allclose(out, x)

def test_preprocess_mono_16k_fixed_length():
    # Stereo -> mono, resample, fixed length
    sr = 8000
    y_stereo = np.random.randn(2, 16000).astype(np.float32)
    # Our preprocess expects 1D; simulate mono conversion via wrapper
    from ai_engine.inference.w2v2_aasist import _preprocess_raw
    # Provide mono for this test
    y_mono = np.random.randn(8000).astype(np.float32)
    out = _preprocess_raw(y_mono, sr=8000)
    assert out.shape[0] == TARGET_LEN
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))

def test_preprocess_short_and_long():
    from ai_engine.inference.w2v2_aasist import _preprocess_raw
    # Short: 0.5 sec at 16k -> 8000 samples -> pad to 64600
    y_short = np.random.randn(8000).astype(np.float32)
    out = _preprocess_raw(y_short, sr=16000)
    assert out.shape[0] == TARGET_LEN
    # Long: 10 sec -> truncate
    y_long = np.random.randn(160000).astype(np.float32)
    out2 = _preprocess_raw(y_long, sr=16000)
    assert out2.shape[0] == TARGET_LEN
    # Content for long should be first 64600
    assert np.allclose(out2, y_long[:TARGET_LEN])

def test_preprocess_handles_different_sr():
    from ai_engine.inference.w2v2_aasist import _preprocess_raw
    y_22k = np.random.randn(22050).astype(np.float32)
    out = _preprocess_raw(y_22k, sr=22050)
    assert out.shape[0] == TARGET_LEN

def test_model_loading():
    m = W2V2AASISTModel()
    assert m.is_loaded
    assert m.version == "w2v2_aasist"
    assert m.target_len == TARGET_LEN
    assert m.target_sr == TARGET_SR

def test_score_conversion_bonafide_vs_spoof():
    m = W2V2AASISTModel()
    # Logits: [spoof, bonafide] — higher bonafide should give lower P_fake
    logits_spoof = np.array([5.0, -5.0])  # strongly spoof
    p_real, p_fake = m._logits_to_probs(logits_spoof)
    assert p_fake > 0.9
    assert p_real < 0.1
    assert abs((p_real + p_fake) - 1.0) < 1e-6

    logits_bonafide = np.array([-5.0, 5.0])  # strongly bonafide
    p_real2, p_fake2 = m._logits_to_probs(logits_bonafide)
    assert p_fake2 < 0.1
    assert p_real2 > 0.9
    # P_fake should be monotonic with bonafide logit
    assert p_fake > p_fake2

def test_score_conversion_single_logit():
    m = W2V2AASISTModel()
    # Single logit via sigmoid
    p_real, p_fake = m._logits_to_probs(np.array([10.0]))
    assert p_fake < 0.01
    p_real2, p_fake2 = m._logits_to_probs(np.array([-10.0]))
    assert p_fake2 > 0.99

def test_output_format_predict_proba():
    m = W2V2AASISTModel()
    y = np.random.randn(16000).astype(np.float32) * 0.01
    p_real, p_fake = m.predict_proba(y, sr=16000)
    assert 0.0 <= p_real <= 1.0
    assert 0.0 <= p_fake <= 1.0
    assert abs((p_real + p_fake) - 1.0) < 1e-6
    pred = m.predict(y, sr=16000)
    assert pred in (0, 1)
    # Threshold 0.5 should match P_fake
    assert (pred == 1) == (p_fake >= 0.5)

def test_predict_from_file(tmp_path):
    import soundfile as sf
    sr = 16000
    y = np.random.randn(sr).astype(np.float32) * 0.01
    p = tmp_path / "test.wav"
    sf.write(str(p), y, sr)
    m = W2V2AASISTModel()
    pred, p_real, p_fake = m.predict_from_file(p)
    assert pred in (0, 1)
    assert 0 <= p_real <= 1 and 0 <= p_fake <= 1

def test_fixed_input_length_inference():
    """Ensure inference always uses 64600 regardless of input duration."""
    m = W2V2AASISTModel()
    for dur in [0.5, 1.0, 4.0, 8.0]:
        y = np.random.randn(int(TARGET_SR * dur)).astype(np.float32)
        # Should not raise, and preprocess gives fixed len
        y_proc = m.preprocess(y, TARGET_SR)
        assert y_proc.shape[0] == TARGET_LEN
        # Inference should succeed
        p_real, p_fake = m.predict_proba(y, TARGET_SR)
        assert np.isfinite(p_real) and np.isfinite(p_fake)
