# Siren-aware Adaptive Traffic Signal

A college IoT project. An Arduino Uno listens to a microphone, classifies sound
as **traffic/noise**, **ambulance**, or **firetruck** using a tiny decision tree
trained offline in Python, and the signal controller adapts its green-phase timers.

## Directory structure

```
siren-signal/
  ml/              Python training pipeline
    config.py      Single source of truth for all constants
    prep_data.py   Scan data/raw, build manifest.csv
    features.py    Integer feature extractor (bit-exact with Uno)
    augment.py     Augmentation helpers
    train.py       Grid-search training with grouped CV
    evaluate.py    Figures: confusion matrix, accuracy vs SNR, scatter
    export_c.py    Generate C headers + probe sketch
    models/        siren_tree.json, metrics.json  (generated)
  arduino/
    signal_controller/   config.h, siren_model.h, siren_features.h (generated)
    siren_probe/         siren_probe.ino (generated calibration sketch)
  data/
    raw/           Read-only audio  (gitignored)
    processed/     manifest.csv     (gitignored)
  tests/
    test_features.py   Feature extractor correctness tests
    test_export.py     Python ↔ C equivalence check (2000 vectors)
    parity_host.cpp    Optional g++ parity harness
  docs/
    data_report.md     Auto-generated
    ml_report.md       Auto-generated after ml.evaluate
    figures/           confusion_matrix.png, accuracy_vs_snr.png, pitch_scatter.png

```

## Reproduce from scratch

```bash
# 0. Inspect environment
python --version  # need 3.11+

# 1. Install dependencies
pip install -r requirements.txt

# 2. Add audio to data/raw/{traffic,ambulance,firetruck}/
#    (public datasets or own phone recordings)

# 3. Data prep
python -m ml.prep_data

# 4. Train
python -m ml.train

# 5. Evaluate + figures
python -m ml.evaluate

# 6. Export to C
python -m ml.export_c

# 7. Run all tests
python -m pytest

# 8. (Optional) Compile for Uno — if arduino-cli is installed
arduino-cli compile --fqbn arduino:avr:uno arduino/siren_probe
```

## Hard constraints

- Target: Arduino Uno (ATmega328P) — 2 KB SRAM, 32 KB flash, no FPU
- Model: DecisionTreeClassifier, ≤16 leaves
- Features: 4 integer pitch features (pmin, pmax, pmean, sweep)
- No float/double/String/malloc on the Uno

## Important notes

- **data/raw is read-only** — never move or modify audio files
- **Generated files** (`siren_model.h`, `siren_features.h`, `config.h`, `siren_probe.ino`) — never edit by hand
- **signal_controller.ino** — do not edit (owned by hardware team)
- Numbers in the report must trace to script output (seed 42)
