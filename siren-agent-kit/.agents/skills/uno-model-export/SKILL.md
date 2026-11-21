---
name: uno-model-export
description: Convert a trained scikit-learn decision tree into integer-only C headers for an Arduino Uno, generate the siren_probe test sketch, verify the exported logic against Python, and check flash and SRAM use. Use this skill whenever exporting a model to C, writing Arduino headers or sketches for the classifier, calibrating the loudness gate, or answering whether something fits on the Uno.
---

# Uno model export

## Export rules (ml/export_c.py)
1. Read the fitted tree from ml/models/siren_tree.json (stored via tree_ arrays: children, feature, threshold, value). Emit nested if/else, never recursion or arrays of floats.
2. Features are integers, so a split `x <= t` with float t becomes `x <= floor(t)`. Feature order is fixed: pmin, pmax, pmean, sweep.
3. Generate arduino/signal_controller/siren_model.h containing: class constants (SIREN_NOISE 0, SIREN_AMBULANCE 1, SIREN_FIRE 2), and `static inline uint8_t siren_classify(int16_t pmin, int16_t pmax, int16_t pmean, int16_t sweep)`. Put a comment header with the training date, git commit, grouped-CV macro-F1 and held-out accuracy, and the text "GENERATED, DO NOT EDIT".
4. Generate config.h from ml/config.py (SR, WIN, WINS_PER_CHUNK, HYST, DC_SHIFT, LEVEL_GATE).
5. Generate siren_features.h from the specification in siren-feature-parity (streaming, static memory under 40 bytes).

## Equivalence proof (tests/test_export.py)
For 2000 random integer feature vectors within the observed training ranges, Python `tree.predict` must equal the exported C logic. Prefer compiling the header with g++ and comparing; if g++ is unavailable, evaluate the same generated rules through a Python function emitted by the same exporter and say so.

## Probe sketch (arduino/siren_probe/siren_probe.ino)
- Read A0 with analogRead in a tight loop, feed siren_feed(), and on each completed chunk print one line at 115200 baud: `pmin,pmax,pmean,sweep,lvl,class,sr_est`.
- Measure sr_est from micros() across a chunk so the team can confirm the real sampling rate. If it differs from SR by more than 3 percent, the final message must say that retraining with the measured SR is required.
- Add a calibration mode (hold a pin or send 'c' over Serial) that prints lvl continuously so LEVEL_GATE can be tuned against a real microphone in a quiet room, with phone-speaker sirens at 0.5 m and 2 m.
- Use no String class and no dynamic memory.

## Resource check
If arduino-cli is installed, run `arduino-cli compile --fqbn arduino:avr:uno arduino/siren_probe` and report flash and global-variable bytes. Limits for model plus feature code: 6 KB flash, 600 B static SRAM. If arduino-cli is missing, say "not compiled" and estimate nothing.

## Do not
- Edit signal_controller.ino.
- Use float, double, String, malloc, new, or recursion in anything under arduino/.
- Claim the model works on hardware. Only the probe sketch on a real board can show that.
