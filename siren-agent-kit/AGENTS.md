# Siren-aware adaptive traffic signal: ML + Arduino rules

## What this project is
College IoT lab project. An Arduino Uno listens to a microphone, classifies the sound as traffic/noise, ambulance or firetruck with a tiny decision tree trained offline in Python, and the signal controller adapts its timers. Training happens on a laptop. The Uno only runs exported integer if-else rules. Time is very short: prefer simple, verifiable and working over clever.

## Hard constraints (never violate)
1. Target is Arduino Uno (ATmega328P): 2048 B SRAM, 32 KB flash, no FPU. The model plus feature code may add at most 600 B static SRAM and 6 KB flash. Uno-side code: no float/double, no String, no malloc/new, no recursion. Use int16_t/int32_t only.
2. Model is a scikit-learn DecisionTreeClassifier with at most 16 leaves. No neural nets, no GPU code, no TensorFlow or PyTorch.
3. Train/deploy parity. The feature extractor is specified once (skill: siren-feature-parity). ml/features.py must reproduce the Uno C++ routine bit for bit on integer ADC samples (0..1023). All constants live in ml/config.py; arduino/signal_controller/config.h is generated from it. Never hand-edit generated files.
4. Honest evaluation. Split by source FILE (grouped), never by chunk. Never evaluate on augmented copies of training files. Seed is 42. Report held-out numbers only. Never fabricate or round up metrics; every number in any document must come from a script run.
5. Data. data/raw is read-only: never delete, move or modify it. Never commit audio (it is gitignored). Synthetic audio is allowed only for pipeline smoke tests, must be labelled synthetic, and its metrics are never reported as results.
6. Git. You may `git add` and `git commit` locally with clear messages. Never `git push`, never force, reset, rebase or touch remotes.
7. Do not modify arduino/signal_controller/signal_controller.ino (the owner writes it). You may write only generated headers next to it and files under arduino/siren_probe/.
8. Dependencies: numpy, scipy, scikit-learn, librosa, soundfile, matplotlib, pandas, pytest. numba only if it installs cleanly. Pin versions in requirements.txt. Ask before downloading any dataset over 1 GB.
9. Windows-friendly: use pathlib, no bash-only commands, run everything from the repo root as `python -m ml.<module>`.

## Directory structure
```
siren-signal/
  AGENTS.md
  .agents/skills/{siren-data-prep,siren-feature-parity,uno-model-export}/SKILL.md
  README.md  requirements.txt  .gitignore
  data/                      (gitignored)
    raw/{traffic,ambulance,firetruck}/   read-only audio
    processed/manifest.csv
  ml/
    config.py                single source of truth for constants
    prep_data.py  features.py  augment.py  train.py  evaluate.py  export_c.py
    models/                  siren_tree.json, metrics.json
  arduino/
    signal_controller/       siren_features.h, siren_model.h, config.h (generated), signal_controller.ino (DO NOT EDIT)
    siren_probe/siren_probe.ino
  tests/                     test_features.py, test_export.py, parity_host.cpp
  docs/                      data_report.md, ml_report.md, figures/
```

## Working rules
- Start with a plan of at most 15 lines and wait for approval before heavy work.
- Smoke test on at most 20 files per class before running on everything.
- Commit after each phase. Run pytest before every commit.
- Keep the code small and readable: a student must be able to explain every file. Short comments that say why, not what.
- If blocked for more than 10 minutes, stop and report instead of guessing.
- Be blunt about weaknesses. A weak honest result is better than a strong fake one.

## Definition of done
- `python -m pytest` passes and `python -m ml.train` reproduces the saved metrics with seed 42.
- Exported C compiles for the Uno (arduino-cli if available; otherwise say "not compiled") and fits the limits above.
- Python tree predictions equal the exported C logic on 2000 random feature vectors.
- docs/ml_report.md exists, is at most 2 pages, and uses only script-produced numbers.
