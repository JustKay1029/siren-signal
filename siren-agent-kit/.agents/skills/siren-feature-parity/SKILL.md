---
name: siren-feature-parity
description: The exact integer feature-extraction specification (zero-crossing pitch, sweep, loudness) shared by the Python training code and the Arduino Uno C++ code, plus the tests that prove both sides give identical results. Use this skill whenever writing or changing ml/features.py, arduino siren_features.h, sampling rate or window constants, augmentation that touches raw samples, or when training accuracy looks fine but the Uno behaves differently.
---

# Siren feature parity

The model is only valid if the Uno computes features exactly the way Python did during training. This file is the specification. Python and C++ implement it independently and must agree on every integer.

## Constants (defined once in ml/config.py, generated into config.h)
SR = 9600 Hz (the Uno's analogRead rate at default prescaler is about 9.6 kHz; the probe sketch measures the real value, and if it differs, update SR and retrain)
WIN = 256 samples per window
WINS_PER_CHUNK = 37 (about 1 s)
HYST = 6 ADC counts
DC_SHIFT = 6
LEVEL_GATE = tune on real hardware (default 40)

## Input
Samples are 10-bit ADC counts 0..1023, centre about 512. Python converts audio x in [-1, 1] with adc = clip(round(512 + 511 * x * gain), 0, 1023). During training, gain is drawn log-uniformly from [0.03, 0.9] and Gaussian ADC noise of sigma 1 to 3 counts is added before rounding, to mimic real microphones at different distances.

## Algorithm (streaming; state persists across window and chunk boundaries; reset only at file start or power-up)
Per sample x:
```
dc16 += ((int32)(x << 4) - dc16) >> 6      // dc16 starts at 512 << 4; arithmetic shift
dc = dc16 >> 4
level_sum += abs(x - dc)
if (x > dc + HYST) { if (state == 0) cross++; state = 1; }
else if (x < dc - HYST) { if (state == 1) cross++; state = 0; }
// state starts at -1 (unknown): the first threshold hit sets it without counting
```
Per window end (every WIN samples):
```
pitch = (cross * 75) >> 2        // Hz, equals cross * SR / (2 * WIN)
level = level_sum >> 8
cross = 0; level_sum = 0
```
Per chunk end (every WINS_PER_CHUNK windows):
```
pmin, pmax = min, max of pitch over the chunk
pmean  = sum(pitch) / 37           // integer division
sweep  = sum(abs(pitch[i] - pitch[i-1]) for i in 1..36) / 36   // the first window of a chunk has no previous
lvl    = sum(level) / 37
```
Gate: if lvl < LEVEL_GATE the chunk is classified NOISE without running the tree. The tree itself takes only (pmin, pmax, pmean, sweep). Loudness is for the gate and for the near/far distance logic, never a tree feature.

## Rules
- Integer arithmetic only on both sides. Python ints or numpy int32; no float in the feature path.
- ml/features.py exposes extract_chunks(adc_samples) returning an integer array of shape (n_chunks, 5): pmin, pmax, pmean, sweep, lvl.
- arduino/signal_controller/siren_features.h exposes a streaming struct with `siren_feed(int16 sample)` that returns true when a chunk completes. Static memory under 40 bytes.
- Pure Python loops are fine. numba is optional.

## Tests (tests/test_features.py, must pass)
- 1000 Hz sine at moderate amplitude gives pitch within 20 Hz of 1000.
- Silence and amplitude below HYST give pitch 0.
- Window and chunk boundaries do not reset state: feeding one long signal equals feeding it in pieces.
- Same input gives identical output (determinism).
- If g++ is available, tests/parity_host.cpp includes siren_features.h, runs 5 fixed-seed random ADC arrays dumped to CSV by Python, and the outputs must be exactly equal. If g++ is not available, say "host parity test skipped" in the final message.
