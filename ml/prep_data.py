"""
ml/prep_data.py — scan data/raw, validate files, build data/processed/manifest.csv.

Run:  python -m ml.prep_data
      python -m ml.prep_data --smoke   (at most 20 files per class)

Follows the siren-data-prep skill exactly.
"""

import argparse, hashlib, pathlib, sys
import numpy as np
import librosa
import soundfile as sf
import pandas as pd

from ml.config import (
    CLASSES, EXCLUDE_CLASSES, DATA_RAW, MANIFEST_CSV, DATA_PROCESSED,
    DOCS_DIR, SR
)

MIN_DURATION_S = 1.5
CLIPPING_THRESHOLD = 0.95   # fraction of samples at ±max → flagged as clipped
SILENCE_RMS_THRESHOLD = 0.001


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def rms_of(sig: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sig.astype(float) ** 2)))


def is_clipped(sig: np.ndarray, thr=CLIPPING_THRESHOLD) -> bool:
    peak = np.max(np.abs(sig))
    if peak == 0:
        return False
    fraction = np.mean(np.abs(sig) >= thr * peak)
    return fraction > 0.01   # >1 % of samples at peak


def scan_class(class_name: str, label: int, raw_root: pathlib.Path,
               max_per_class: int | None, exclusions: list) -> list[dict]:
    folder = raw_root / class_name
    if not folder.exists():
        return []

    rows = []
    seen_md5s: dict[str, pathlib.Path] = {}
    count = 0

    exts = {".wav", ".mp3", ".flac", ".ogg"}
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() not in exts:
            continue
        if max_per_class is not None and count >= max_per_class:
            break

        source = "own" if f.stem.startswith("own_") else "public"
        reason = None

        try:
            sig, sr_orig = librosa.load(f, sr=None, mono=True)
        except Exception as e:
            reason = f"unreadable: {e}"
            exclusions.append({"path": str(f), "reason": reason})
            continue

        dur = len(sig) / sr_orig
        if dur < MIN_DURATION_S:
            reason = f"too_short ({dur:.2f}s)"
            exclusions.append({"path": str(f), "reason": reason})
            continue

        file_rms = rms_of(sig)
        if file_rms < SILENCE_RMS_THRESHOLD:
            reason = "near_silent"
            exclusions.append({"path": str(f), "reason": reason})
            continue

        if is_clipped(sig):
            reason = "clipped"
            exclusions.append({"path": str(f), "reason": reason})
            continue

        m = md5(f)
        if m in seen_md5s:
            reason = f"duplicate_of {seen_md5s[m]}"
            exclusions.append({"path": str(f), "reason": reason})
            continue
        seen_md5s[m] = f

        rows.append({
            "path":       str(f.as_posix()),
            "class":      class_name,
            "label":      label,
            "group":      f.stem,          # one source file = one group for splitting
            "duration_s": round(dur, 3),
            "sr_orig":    int(sr_orig),
            "rms":        round(file_rms, 6),
            "source":     source,
            "md5":        m,
        })
        count += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Limit to 20 files per class (smoke test)")
    args = parser.parse_args()

    max_per_class = 20 if args.smoke else None

    raw_root = pathlib.Path(DATA_RAW)
    if not raw_root.exists():
        print(f"ERROR: {DATA_RAW} does not exist. Create data/raw/<class>/ and add audio.")
        sys.exit(1)

    pathlib.Path(DATA_PROCESSED).mkdir(parents=True, exist_ok=True)
    pathlib.Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

    rows = []
    exclusions = []

    for label, name in enumerate(CLASSES):
        if name in EXCLUDE_CLASSES:
            print(f"  Skipping excluded class: {name}")
            continue
        class_rows = scan_class(name, label, raw_root, max_per_class, exclusions)
        rows.extend(class_rows)
        print(f"  {name}: {len(class_rows)} files accepted")

    # Also check for extra (police etc.) classes and report them
    extra_classes = []
    for d in raw_root.iterdir():
        if d.is_dir() and d.name not in CLASSES and d.name not in EXCLUDE_CLASSES:
            extra_classes.append(d.name)
    if extra_classes:
        print(f"  WARNING: unexpected class folders found and skipped: {extra_classes}")

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_CSV, index=False)
    print(f"\nManifest saved: {MANIFEST_CSV} ({len(df)} rows)")

    # Data report
    _write_report(df, exclusions, extra_classes, args.smoke)


def _write_report(df: pd.DataFrame, exclusions: list, extra_classes: list, smoke: bool):
    lines = ["# Data Report\n"]
    if smoke:
        lines.append("> **Smoke test**: at most 20 files per class.\n")

    if df.empty:
        lines.append("No valid files found in data/raw.\n")
    else:
        lines.append("## Class summary\n")
        lines.append("| class | n_files | total_s | mean_s | sources |\n")
        lines.append("|-------|---------|---------|--------|---------|\n")
        for cls in df["class"].unique():
            sub = df[df["class"] == cls]
            sources = "/".join(sorted(sub["source"].unique()))
            lines.append(
                f"| {cls} | {len(sub)} | {sub['duration_s'].sum():.1f} | "
                f"{sub['duration_s'].mean():.1f} | {sources} |\n"
            )

    if exclusions:
        lines.append("\n## Excluded files\n")
        lines.append("| path | reason |\n")
        lines.append("|------|--------|\n")
        for e in exclusions:
            lines.append(f"| {e['path']} | {e['reason']} |\n")

    if extra_classes:
        lines.append(f"\n## Unknown class folders (skipped)\n{extra_classes}\n")

    # Leakage warning heuristic: numbered suffixes
    if not df.empty:
        numbered = df["group"].str.match(r".*_\d+$").sum()
        if numbered > 0:
            lines.append(
                "\n## Leakage warning\n"
                f"{numbered} files have numbered suffixes (e.g. clip_01, clip_02). "
                "These may be segments of the same longer recording. "
                "Accuracy numbers may be optimistic because train/test split "
                "is by filename group, not by original long-file identity. "
                "Own phone clips (own_ prefix) are the most honest test.\n"
            )

    report_path = pathlib.Path(DOCS_DIR) / "data_report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Data report saved: {report_path}")


if __name__ == "__main__":
    main()
