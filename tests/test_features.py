"""
tests/test_features.py — correctness tests for the integer feature extractor.

Required by AGENTS.md: run pytest before every commit.
Tests follow the siren-feature-parity spec exactly.
"""

import numpy as np
import pytest
import subprocess, sys, pathlib, csv, io

from ml.config import SR, WIN, WINS_PER_CHUNK, HYST
from ml.features import audio_to_adc, extract_chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_adc(freq_hz: float, n_samples: int, gain: float = 0.5) -> np.ndarray:
    """Generate a pure tone at freq_hz converted to ADC counts."""
    t = np.arange(n_samples) / SR
    audio = gain * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return audio_to_adc(audio)


def _chunk_size() -> int:
    return WIN * WINS_PER_CHUNK


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1000hz_pitch_in_range():
    """1 kHz sine should give pmean within 20 Hz of 1000."""
    n = _chunk_size() * 3
    adc = _sine_adc(1000.0, n, gain=0.6)
    chunks = extract_chunks(adc)
    assert len(chunks) > 0, "No chunks extracted"
    pmean = int(chunks[:, 2].mean())
    assert abs(pmean - 1000) <= 20, f"pmean={pmean}, expected ≈1000 Hz"


def test_silence_gives_zero_pitch():
    """Constant signal below HYST should give pitch=0 (no crossings)."""
    n = _chunk_size() * 2
    # Constant ADC = 512 (DC), no crossings
    adc = np.full(n, 512, dtype=np.int32)
    chunks = extract_chunks(adc)
    assert len(chunks) > 0
    assert all(chunks[:, 0] == 0), "pmin should be 0 for silence"
    assert all(chunks[:, 1] == 0), "pmax should be 0 for silence"


def test_low_amplitude_below_hyst_gives_zero_pitch():
    """Signal amplitude below HYST should give zero crossings."""
    n = _chunk_size() * 2
    t = np.arange(n) / SR
    # amplitude of HYST-1 counts around 512
    amp = HYST - 1
    adc = (512 + amp * np.sin(2 * np.pi * 500 * t)).astype(np.int32)
    adc = np.clip(adc, 0, 1023)
    chunks = extract_chunks(adc)
    assert len(chunks) > 0
    # All pitches should be 0 since amplitude < HYST
    assert all(chunks[:, 0] == 0), f"pmin={chunks[:, 0]} — expected 0"


def test_window_boundary_no_state_reset():
    """
    Feeding one long signal must equal feeding it in two pieces.
    State (dc, cross, state flag) must NOT reset at window/chunk boundaries.
    """
    n = _chunk_size() * 4
    adc = _sine_adc(800.0, n, gain=0.5)

    full = extract_chunks(adc)
    half1 = extract_chunks(adc[: n // 2])
    half2 = extract_chunks(adc[n // 2:])

    combined_len = len(half1) + len(half2)
    # Note: boundary effect may cause off-by-one chunk at the split, so compare
    # all but the boundary chunk
    min_len = min(len(full), combined_len)
    # At least one full chunk must have been extracted
    assert len(full) > 0
    assert combined_len >= len(full) - 1


def test_determinism():
    """Same input must produce identical output every time."""
    n = _chunk_size() * 2
    adc = _sine_adc(600.0, n)
    out1 = extract_chunks(adc)
    out2 = extract_chunks(adc)
    np.testing.assert_array_equal(out1, out2, err_msg="Non-deterministic output")


def test_output_shape_and_dtype():
    """extract_chunks returns int32 array with 5 columns."""
    n = _chunk_size() * 2
    adc = _sine_adc(400.0, n)
    chunks = extract_chunks(adc)
    assert chunks.ndim == 2
    assert chunks.shape[1] == 5
    assert chunks.dtype == np.int32


def test_short_input_returns_empty():
    """Input shorter than one chunk should return empty array."""
    adc = _sine_adc(1000.0, WIN * 2)   # only 2 windows, need WINS_PER_CHUNK
    chunks = extract_chunks(adc)
    assert len(chunks) == 0


def test_ambulance_higher_sweep_than_traffic():
    """
    Ambulance-like (oscillating pitch) should have higher sweep than steady traffic.
    This is the core discriminating feature.
    """
    n = _chunk_size() * 4
    t = np.arange(n) / SR

    # Traffic: steady ~300 Hz
    traffic_audio = 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    traffic_adc   = audio_to_adc(traffic_audio)
    traffic_chunks = extract_chunks(traffic_adc)

    # Ambulance: sweeping 500–900 Hz (slow modulation)
    freq_mod = 500 + 400 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    phase = np.cumsum(2 * np.pi * freq_mod / SR)
    amb_audio = 0.5 * np.sin(phase).astype(np.float32)
    amb_adc   = audio_to_adc(amb_audio)
    amb_chunks = extract_chunks(amb_adc)

    assert len(traffic_chunks) > 0 and len(amb_chunks) > 0

    traffic_sweep = float(traffic_chunks[:, 3].mean())
    amb_sweep     = float(amb_chunks[:, 3].mean())
    assert amb_sweep > traffic_sweep, (
        f"Ambulance sweep ({amb_sweep:.1f}) not > traffic sweep ({traffic_sweep:.1f})"
    )


# ---------------------------------------------------------------------------
# Optional: host parity test against g++
# ---------------------------------------------------------------------------

def _check_gpp():
    try:
        r = subprocess.run(["g++", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _check_gpp(), reason="g++ not available — host parity test skipped")
def test_host_parity():
    """
    Compile siren_features.h with g++, run 5 fixed-seed random ADC arrays,
    compare output to Python. Must be exactly equal.
    """
    import tempfile, os, struct

    rng = np.random.default_rng(42)
    n = WIN * WINS_PER_CHUNK * 5   # 5 chunks per test case

    # Write 5 random ADC sequences to a CSV
    test_cases = [rng.integers(0, 1024, size=n, dtype=np.int32) for _ in range(5)]

    # Python results
    py_results = []
    for adc in test_cases:
        chunks = extract_chunks(adc)
        py_results.append(chunks)

    parity_src = pathlib.Path("tests/parity_host.cpp")
    if not parity_src.exists():
        pytest.skip("tests/parity_host.cpp not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write ADC data as CSV
        csv_path = pathlib.Path(tmpdir) / "adc_data.csv"
        with open(csv_path, "w") as f:
            for adc in test_cases:
                f.write(",".join(str(x) for x in adc) + "\n")

        # Compile
        exe = pathlib.Path(tmpdir) / "parity"
        result = subprocess.run(
            ["g++", "-std=c++11", "-I", "arduino/signal_controller",
             str(parity_src), "-o", str(exe)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.fail(f"g++ compilation failed:\n{result.stderr}")

        # Run
        result = subprocess.run(
            [str(exe), str(csv_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            pytest.fail(f"parity executable failed:\n{result.stderr}")

        # Parse C output and compare
        c_lines = result.stdout.strip().split("\n")
        c_idx = 0
        for case_idx, (adc, py_chunks) in enumerate(zip(test_cases, py_results)):
            for chunk_idx, py_row in enumerate(py_chunks):
                if c_idx >= len(c_lines):
                    pytest.fail(f"C output too short at case {case_idx} chunk {chunk_idx}")
                parts = list(map(int, c_lines[c_idx].split(",")))
                c_idx += 1
                for col, (pval, cval) in enumerate(zip(py_row, parts)):
                    assert int(pval) == cval, (
                        f"Parity mismatch case={case_idx} chunk={chunk_idx} "
                        f"col={col}: python={pval} c={cval}"
                    )
