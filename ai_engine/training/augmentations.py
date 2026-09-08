"""
Audio augmentation pipeline for Morph robustness training.

Simulates realistic speaker/microphone/channel effects to make the model
robust to mic-recorded audio without destroying synthetic-speech artifacts.

All augmentations operate on raw audio arrays at 16 kHz mono.
The augmented audio is then fed through the SAME production feature extraction
pipeline (extraction.py → 132 features → XGBoost).

Design principles:
    - Keep augmentations realistic (matching real-world mic recording effects)
    - Do NOT destroy the characteristics that distinguish synthetic from real speech
    - Each augmentation is independently toggleable
    - Randomized parameters for diversity across training epochs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    """Configuration for a single augmentation pipeline run."""

    # Speaker/mic frequency response simulation
    apply_eq: bool = True
    eq_lowcut_hz: float = 200.0      # Low-frequency rolloff (speaker limit)
    eq_highcut_hz: float = 6000.0    # High-frequency rolloff (mic/speaker limit)
    eq_order: int = 4                 # Filter order
    eq_randomize: bool = True         # Randomize cutoffs per sample

    # Room reverberation simulation
    apply_reverb: bool = True
    reverb_r60_db: float = 0.3        # RT60 in seconds (very mild)
    reverb_mix: float = 0.15          # Wet/dry mix (0=dry, 1=fully wet)
    reverb_randomize: bool = True

    # Background noise
    apply_noise: bool = True
    noise_snr_db_range: Tuple[float, float] = (20.0, 40.0)
    noise_randomize: bool = True

    # Volume/gain variation
    apply_gain: bool = True
    gain_db_range: Tuple[float, float] = (-6.0, 6.0)
    gain_randomize: bool = True

    # Resampling variation (simulates different DACs/resamplers)
    apply_resample_jitter: bool = True
    resample_sr_range: Tuple[int, int] = (15800, 16200)
    resample_randomize: bool = True

    # Codec-like degradation (mild, non-destructive)
    apply_codec: bool = True
    codec_bit_depth_options: Tuple[int, ...] = (12, 14, 16)
    codec_randomize: bool = True


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def apply_bandpass_filter(
    y: np.ndarray,
    sr: int,
    lowcut: float = 200.0,
    highcut: float = 6000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter simulating speaker/mic frequency response.

    Real speakers and microphones have limited bandwidth. This removes
    very low and very high frequencies, matching the spectral rolloff,
    bandwidth, and centroid shifts observed in the diagnostic.
    """
    nyquist = sr / 2
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)

    b, a = scipy_signal.butter(order, [low, high], btype="band")
    y_filtered = scipy_signal.filtfilt(b, a, y)

    # Clip to prevent overflow
    y_filtered = np.clip(y_filtered, -1.0, 1.0)
    return y_filtered.astype(np.float32)


def apply_reverberation(
    y: np.ndarray,
    sr: int,
    r60_seconds: float = 0.3,
    mix: float = 0.15,
) -> np.ndarray:
    """
    Simulate mild room reverberation using a simple exponential decay impulse response.

    This models the effect of room reflections that blur temporal features
    and reduce the contrast in spectral features.
    """
    if r60_seconds <= 0 or mix <= 0:
        return y

    # Generate exponential decay IR
    # r60 is the time for 60 dB decay → decay constant = r60 / (60 / (20*log10(e)))
    decay_samples = int(r60_seconds * sr)
    if decay_samples < 1:
        return y

    t = np.arange(decay_samples) / sr
    # Exponential decay: e^(-t / tau), where tau = r60 / 6.91
    tau = r60_seconds / 6.91
    ir = np.exp(-t / tau)
    ir[0] = 1.0  # Direct path

    # Normalize IR
    ir = ir / np.sqrt(np.sum(ir ** 2))

    # Convolve (use 'same' to keep length)
    reverb_tail = np.convolve(y, ir, mode="full")[:len(y)]

    # Mix dry and wet
    y_reverb = (1 - mix) * y + mix * reverb_tail

    # Normalize to prevent clipping
    peak = np.max(np.abs(y_reverb))
    if peak > 1.0:
        y_reverb = y_reverb / peak * 0.99

    return y_reverb.astype(np.float32)


def apply_background_noise(
    y: np.ndarray,
    sr: int,
    snr_db: float = 30.0,
) -> np.ndarray:
    """
    Add realistic background noise (white + brownian noise mixture).

    Simulates ambient noise from a real recording environment.
    """
    signal_power = np.mean(y ** 2)
    if signal_power < 1e-10:
        return y

    # Mix of white and brown noise for realistic ambient sound
    noise_white = np.random.randn(len(y)).astype(np.float32)
    noise_brown = np.cumsum(np.random.randn(len(y))).astype(np.float32)
    noise_brown = noise_brown / (np.max(np.abs(noise_brown)) + 1e-10)
    noise = 0.7 * noise_white + 0.3 * noise_brown

    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return y

    # Scale noise to achieve desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
    y_noisy = y + noise * noise_scale

    y_noisy = np.clip(y_noisy, -1.0, 1.0)
    return y_noisy.astype(np.float32)


def apply_gain_variation(
    y: np.ndarray,
    gain_db: float = 0.0,
) -> np.ndarray:
    """
    Apply volume/gain change.

    Simulates different recording levels and speaker volumes.
    """
    if abs(gain_db) < 0.01:
        return y

    gain_linear = 10 ** (gain_db / 20)
    y_gained = y * gain_linear
    y_gained = np.clip(y_gained, -1.0, 1.0)
    return y_gained.astype(np.float32)


def apply_resample_jitter(
    y: np.ndarray,
    sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Simulate resampling to a slightly different rate and back.

    This models the effect of different DACs, sound cards, and software
    resamplers that produce slightly different sample rates.
    """
    if target_sr == sr:
        return y

    # Downsample to target_sr (integer ratio approximation)
    # Use simple linear interpolation for speed
    ratio = target_sr / sr
    n_out = int(len(y) * ratio)
    indices = np.linspace(0, len(y) - 1, n_out)
    y_down = np.interp(indices, np.arange(len(y)), y)

    # Upsample back to original sr
    n_back = len(y)
    indices_back = np.linspace(0, len(y_down) - 1, n_back)
    y_up = np.interp(indices_back, np.arange(len(y_down)), y_down)

    y_up = np.clip(y_up, -1.0, 1.0)
    return y_up.astype(np.float32)


def apply_quantization(
    y: np.ndarray,
    bit_depth: int = 16,
) -> np.ndarray:
    """
    Apply quantization (bit-depth reduction).

    Simulates the effect of low-bit-depth codecs and compression artifacts.
    Mild quantization preserves speech characteristics while adding
    realistic codec-like degradation.
    """
    levels = 2 ** bit_depth
    y_quantized = np.round(y * (levels / 2)) / (levels / 2)
    y_quantized = np.clip(y_quantized, -1.0, 1.0)
    return y_quantized.astype(np.float32)


# ---------------------------------------------------------------------------
# Composite augmentation pipeline
# ---------------------------------------------------------------------------

def augment_audio(
    y: np.ndarray,
    sr: int = 16000,
    config: Optional[AugConfig] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Apply the full augmentation pipeline to an audio signal.

    Returns the augmented audio and a dict of applied parameters
    (for reproducibility and logging).

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal (16 kHz recommended).
    sr : int
        Sample rate.
    config : AugConfig, optional
        Augmentation configuration. Uses default if None.
    rng : np.random.RandomState, optional
        Random state for reproducibility.

    Returns
    -------
    y_aug : np.ndarray
        Augmented audio signal.
    params : dict
        Applied augmentation parameters.
    """
    if config is None:
        config = AugConfig()
    if rng is None:
        rng = np.random.RandomState()

    params = {}
    y_aug = y.copy().astype(np.float32)

    # 1. Bandpass filter (speaker/mic frequency response)
    if config.apply_eq:
        if config.eq_randomize:
            lowcut = rng.uniform(
                config.eq_lowcut_hz * 0.5,
                config.eq_lowcut_hz * 2.0,
            )
            highcut = rng.uniform(
                config.eq_highcut_hz * 0.5,
                config.eq_highcut_hz * 1.5,
            )
        else:
            lowcut = config.eq_lowcut_hz
            highcut = config.eq_highcut_hz

        y_aug = apply_bandpass_filter(y_aug, sr, lowcut, highcut, config.eq_order)
        params["eq_lowcut"] = round(lowcut, 1)
        params["eq_highcut"] = round(highcut, 1)

    # 2. Reverberation
    if config.apply_reverb:
        if config.reverb_randomize:
            r60 = rng.uniform(0.05, config.reverb_r60_db * 3)
            mix = rng.uniform(0.05, config.reverb_mix * 2)
        else:
            r60 = config.reverb_r60_db
            mix = config.reverb_mix

        y_aug = apply_reverberation(y_aug, sr, r60, mix)
        params["reverb_r60"] = round(r60, 4)
        params["reverb_mix"] = round(mix, 4)

    # 3. Background noise
    if config.apply_noise:
        if config.noise_randomize:
            snr_db = rng.uniform(*config.noise_snr_db_range)
        else:
            snr_db = (config.noise_snr_db_range[0] + config.noise_snr_db_range[1]) / 2

        y_aug = apply_background_noise(y_aug, sr, snr_db)
        params["noise_snr_db"] = round(snr_db, 1)

    # 4. Gain variation
    if config.apply_gain:
        if config.gain_randomize:
            gain_db = rng.uniform(*config.gain_db_range)
        else:
            gain_db = (config.gain_db_range[0] + config.gain_db_range[1]) / 2

        y_aug = apply_gain_variation(y_aug, gain_db)
        params["gain_db"] = round(gain_db, 1)

    # 5. Resampling jitter
    if config.apply_resample_jitter:
        if config.resample_randomize:
            target_sr = rng.randint(*config.resample_sr_range)
        else:
            target_sr = (config.resample_sr_range[0] + config.resample_sr_range[1]) // 2

        y_aug = apply_resample_jitter(y_aug, sr, target_sr)
        params["resample_sr"] = int(target_sr)

    # 6. Quantization (codec-like degradation)
    if config.apply_codec:
        if config.codec_randomize:
            bit_depth = rng.choice(config.codec_bit_depth_options)
        else:
            bit_depth = config.codec_bit_depth_options[1]  # middle option

        y_aug = apply_quantization(y_aug, bit_depth)
        params["quant_bits"] = int(bit_depth)

    # Final normalization to prevent any overflow
    peak = np.max(np.abs(y_aug))
    if peak > 0.99:
        y_aug = y_aug / peak * 0.99

    return y_aug, params


def augment_audio_batch(
    y: np.ndarray,
    sr: int = 16000,
    n_augmentations: int = 3,
    config: Optional[AugConfig] = None,
    seed: int = 42,
) -> List[Tuple[np.ndarray, dict]]:
    """
    Generate multiple augmented versions of a single audio signal.

    Each version uses different random parameters, providing diverse
    training examples from a single source file.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    n_augmentations : int
        Number of augmented versions to generate.
    config : AugConfig, optional
        Augmentation configuration.
    seed : int
        Base seed for reproducibility.

    Returns
    -------
    list of (np.ndarray, dict)
        Each element is (augmented_audio, applied_params).
    """
    results = []
    for i in range(n_augmentations):
        rng = np.random.RandomState(seed + i)
        y_aug, params = augment_audio(y, sr, config, rng)
        results.append((y_aug, params))
    return results
