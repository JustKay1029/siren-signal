"""
ml/features.py — integer feature extractor, bit-exact port of siren_features.h.

Implements the streaming algorithm from siren-feature-parity:
  - EMA DC removal (integer, dc16 in fixed-point Q4)
  - Hysteresis zero-crossing pitch estimator
  - Per-window: pitch (Hz) and level (mean |x - dc|)
  - Per-chunk:  pmin, pmax, pmean, sweep, lvl  (all int32)

Public API:
  extract_chunks(adc_samples: array-like[int]) -> np.ndarray shape (n_chunks, 5)
      columns: pmin, pmax, pmean, sweep, lvl

adc_samples must be integer 0..1023 (ADC counts).
Use audio_to_adc() to convert float audio before calling.
"""

import numpy as np
from ml.config import SR, WIN, WINS_PER_CHUNK, HYST, DC_SHIFT, ADC_MID, ADC_BITS, GAIN_LOW, GAIN_HIGH, ADC_NOISE_LOW, ADC_NOISE_HIGH


def audio_to_adc(audio: np.ndarray, gain: float | None = None, noise_sigma: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Convert float audio (−1..1) to 10-bit ADC counts (0..1023).
    Optionally apply gain and Gaussian ADC noise for augmentation.
    Returns int32 array.
    """
    if gain is None:
        gain = 1.0
    adc = ADC_MID + 511 * audio * gain
    if noise_sigma > 0.0 and rng is not None:
        adc = adc + rng.standard_normal(len(adc)) * noise_sigma
    adc = np.clip(np.round(adc), 0, (1 << ADC_BITS) - 1).astype(np.int32)
    return adc


def extract_chunks(adc_samples) -> np.ndarray:
    """
    Run the streaming pitch/level algorithm over integer ADC samples.
    Returns integer array of shape (n_chunks, 5):
        [pmin, pmax, pmean, sweep, lvl]
    Rows with lvl < LEVEL_GATE are still returned; the caller/gate decides.
    """
    adc = np.asarray(adc_samples, dtype=np.int32)
    n = len(adc)

    # Streaming state — mirrors the C struct exactly
    dc16  = np.int32(ADC_MID << DC_SHIFT)   # Q4 fixed-point DC estimate
    state = -1                               # -1 = unknown (first edge just sets it)
    cross = np.int32(0)
    level_sum = np.int32(0)

    pitches  = []   # per-window pitch values (one list per chunk)
    levels   = []   # per-window level values

    win_pitches = []
    win_levels  = []

    chunks = []

    win_count  = 0  # samples in current window
    chunk_wins = 0  # windows in current chunk

    for x in adc:
        x = np.int32(x)

        # EMA DC removal (Q4 fixed-point, arithmetic right shift)
        dc16 += (np.int32(x << DC_SHIFT) - dc16) >> DC_SHIFT
        dc = np.int32(dc16 >> DC_SHIFT)

        level_sum += abs(x - dc)

        # Hysteresis zero-crossing
        if x > dc + HYST:
            if state == 0:
                cross += np.int32(1)
            if state != 1:
                state = 1
        elif x < dc - HYST:
            if state == 1:
                cross += np.int32(1)
            if state != 0:
                state = 0

        win_count += 1

        if win_count == WIN:
            # Window end
            # pitch = cross * SR / (2 * WIN)  = cross * 75 >> 2  (integer Hz)
            pitch = np.int32((int(cross) * 75) >> 2)
            level = np.int32(int(level_sum) >> 8)

            win_pitches.append(pitch)
            win_levels.append(level)

            cross = np.int32(0)
            level_sum = np.int32(0)
            win_count = 0
            chunk_wins += 1

            if chunk_wins == WINS_PER_CHUNK:
                # Chunk end
                p = np.array(win_pitches, dtype=np.int32)
                lv = np.array(win_levels, dtype=np.int32)

                pmin  = np.int32(p.min())
                pmax  = np.int32(p.max())
                pmean = np.int32(int(p.sum()) // WINS_PER_CHUNK)
                # sweep: mean |diff| over windows 1..36
                sweep = np.int32(int(np.abs(np.diff(p)).sum()) // (WINS_PER_CHUNK - 1))
                lvl   = np.int32(int(lv.sum()) // WINS_PER_CHUNK)

                chunks.append([pmin, pmax, pmean, sweep, lvl])

                win_pitches = []
                win_levels  = []
                chunk_wins  = 0

    if not chunks:
        return np.empty((0, 5), dtype=np.int32)
    return np.array(chunks, dtype=np.int32)
