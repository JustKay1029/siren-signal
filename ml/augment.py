"""
ml/augment.py — augmentation helpers used during training.

Augmentation happens on ADC samples (integers), not raw float audio,
so it stays in-distribution with the Uno's real input.

Functions:
  augment_chunks(adc, rng) -> list of chunk-feature arrays
      Returns N_AUG extra feature arrays from one audio signal.
  mix_traffic_noise(signal, noise_clips, snr_db, rng) -> np.ndarray
      Returns signal mixed with a random noise clip at the given SNR.
      Used for distance-proxy evaluation at known SNRs.
"""

import numpy as np
import librosa

from ml.config import (
    SR, GAIN_LOW, GAIN_HIGH, ADC_NOISE_LOW, ADC_NOISE_HIGH, N_AUG
)
from ml.features import audio_to_adc, extract_chunks


def augment_chunks(audio: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """
    Generate N_AUG augmented feature matrices from float audio.
    Each copy: random gain (log-uniform) + Gaussian ADC noise.
    Returns list of arrays, each shape (n_chunks, 5).
    """
    result = []
    for _ in range(N_AUG):
        gain = float(np.exp(rng.uniform(np.log(GAIN_LOW), np.log(GAIN_HIGH))))
        noise_sigma = float(rng.uniform(ADC_NOISE_LOW, ADC_NOISE_HIGH))
        adc = audio_to_adc(audio, gain=gain, noise_sigma=noise_sigma, rng=rng)
        chunks = extract_chunks(adc)
        if len(chunks) > 0:
            result.append(chunks)
    return result


def mix_traffic_noise(signal: np.ndarray, noise_clips: list[np.ndarray],
                      snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Mix signal with a random traffic noise clip at the given SNR (dB).
    Both are assumed to be mono float audio at SR.
    Returns the mixed signal as float array of same length as signal.
    """
    if not noise_clips:
        return signal.copy()

    noise = rng.choice(noise_clips)  # pick random noise clip

    # Tile or trim noise to match signal length
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    start = rng.integers(0, len(noise) - len(signal) + 1)
    noise = noise[start: start + len(signal)]

    sig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = signal + noise_scaled
    # Normalise to avoid clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.98
    return mixed.astype(np.float32)
