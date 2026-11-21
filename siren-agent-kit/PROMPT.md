You are my ML + embedded engineer for a college IoT project. Work in this repo root.

FIRST: read AGENTS.md and the three skills in .agents/skills/ (siren-data-prep, siren-feature-parity, uno-model-export). Follow them exactly. If anything conflicts, AGENTS.md wins.

GOAL
Train a tiny siren classifier on my laptop (classes: traffic, ambulance, firetruck) and export it as integer-only C code that an Arduino Uno (2 KB RAM, no FPU) can run on live microphone audio sampled at about 9.6 kHz. A decision tree on four pitch features is the approved model. The result must be honest, reproducible, and easy to explain to a professor.

INPUTS
- Audio in data/raw/<class>/*.wav (public datasets plus my own phone clips). Inspect the real layout first; do not assume it.
- If data/raw is empty or incomplete, say so, run only the synthetic smoke test, and stop after phase 3.

PHASES (commit after each and tell me in one line when it finishes)
0. Plan: inspect the environment (Python version, installed packages, whether g++ and arduino-cli exist; the GPU is irrelevant). Post a plan of at most 15 lines with your assumptions, then wait for my "approved".
1. Scaffold: create the directory structure from AGENTS.md, requirements.txt, .gitignore, README.md and ml/config.py (single source of truth).
2. Data: follow siren-data-prep to produce data/processed/manifest.csv and docs/data_report.md (counts, durations, balance, problems found).
3. Features: implement ml/features.py as a bit-exact integer port of the Uno routine in siren-feature-parity, write the tests, and run a smoke test on at most 20 files per class. Fix problems before scaling up.
4. Train: augmentation (random gain, ADC noise, traffic-noise mixing to simulate distance), grouped cross-validation by source file, a small grid over tree depth and leaf size, choose by macro-F1 under the Uno size limits. Save ml/models/siren_tree.json and ml/models/metrics.json.
5. Evaluate on held-out files only. Produce in docs/figures/: confusion matrix, accuracy versus SNR (20, 10, 5, 0 dB) as a distance proxy, and a pitch-feature scatter plot. Print the tree as readable rules.
6. Export: follow uno-model-export to generate siren_model.h, siren_features.h, config.h and the siren_probe sketch. Prove Python and C logic agree. Compile for the Uno with arduino-cli if available and report flash and SRAM; otherwise say it was not checked.
7. Report: write docs/ml_report.md (max 2 pages, plain language: problem, data, method, results, limitations) using only numbers produced by scripts.

DONE WHEN
- python -m pytest passes and python -m ml.train reproduces the saved metrics with seed 42.
- The exported C is integer-only and within the Uno limits in AGENTS.md.
- Every number in the report traces to a script output.

FINAL MESSAGE FORMAT
1) held-out accuracy and macro-F1, 2) confusion matrix, 3) chosen tree parameters and the rules, 4) Uno flash and SRAM numbers (or "not checked"), 5) known weaknesses, stated bluntly, 6) exact commands to reproduce, 7) what I must do on real hardware next (microphone gain, LEVEL_GATE calibration, measuring the real sampling rate).

Do not push to git. Do not edit signal_controller.ino. Do not invent results. If blocked for more than 10 minutes, stop and ask me.
