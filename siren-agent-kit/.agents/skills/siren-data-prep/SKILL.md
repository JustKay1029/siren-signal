---
name: siren-data-prep
description: Organise, inspect, clean and validate siren and road-noise audio datasets (sireNNet, the Scientific Data siren dataset, or the user's own phone recordings) and build data/processed/manifest.csv with grouped train/test identifiers. Use this skill whenever the task touches audio files, dataset layout, class labels, class balance, duplicates, or how to split audio into train and test sets, even if the user does not say "data preparation".
---

# Siren data prep

## Goal
Turn whatever audio is in data/raw into a clean, leak-free manifest the training code can trust.

## Steps
1. Inspect first. List data/raw recursively and print the real folder layout, file counts and extensions. Do not assume the dataset layout; adapt to what is there.
2. Classes: traffic (background and road noise), ambulance, firetruck. If a police class exists, exclude it by default (config flag EXCLUDE_CLASSES) and say so in the report. Own clips go in the same class folders with an `own_` filename prefix and source=own.
3. For every file compute: path, class, label id, group (the source file stem; each original file is one group), duration_s, original sample rate, RMS, source (public or own), md5.
4. Validation checks. Flag and exclude, never silently drop: unreadable files, duration under 1.5 s, near-silent files (very low RMS), heavy clipping, exact duplicates (same md5; keep one). Write every exclusion with its reason to docs/data_report.md.
5. Load with librosa/soundfile as mono at SR from ml/config.py. If mp3 fails because ffmpeg is missing, report it and skip those files instead of installing system tools.
6. Class imbalance: do not delete data. Record counts per class in the manifest report and let the trainer use class_weight="balanced".
7. Leakage warning: public datasets often cut many short clips from the same long recording. If filenames suggest this (numbered sequences), note it in docs/data_report.md as a limitation, because it can make accuracy look better than it is. Own phone clips are the most honest test, so keep them identifiable.

## Output
- data/processed/manifest.csv with columns: path,class,label,group,duration_s,sr_orig,rms,source,md5
- docs/data_report.md: counts per class and source, total duration per class, exclusions with reasons, and limitations.

## Do not
- Modify or move anything in data/raw.
- Split by chunk. The unit of splitting is the file (group).
- Generate synthetic audio here except for an explicitly labelled smoke test.
