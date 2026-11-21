"""
ml/train.py — training with grouped cross-validation and grid search.

Run:  python -m ml.train
      python -m ml.train --smoke   (use synthetic data only)

Steps:
1. Load features from manifest (or generate synthetic if data/raw is empty).
2. Run grouped K-fold CV (by source file) over depth × min_leaf grid.
3. Choose best hyper-params by macro-F1; enforce ≤MAX_LEAVES.
4. Refit on all training groups, evaluate on held-out groups.
5. Save siren_tree.json and metrics.json.
"""

import argparse, json, pathlib, sys, time
import numpy as np
import librosa
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from ml.config import (
    CLASSES, SR, RANDOM_SEED, MANIFEST_CSV, MODELS_DIR, TREE_JSON, METRICS_JSON,
    MAX_DEPTH_GRID, MIN_SAMPLES_GRID, MAX_LEAVES, LEVEL_GATE
)
from ml.features import audio_to_adc, extract_chunks
from ml.augment import augment_chunks


FEATURE_NAMES = ["pmin", "pmax", "pmean", "sweep"]   # 4 tree features
CHUNK_COLS    = ["pmin", "pmax", "pmean", "sweep", "lvl"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features_from_manifest(manifest_path: str, smoke: bool = False) -> tuple:
    """
    For every file in the manifest, load audio, convert to ADC, extract features.
    Returns X (n_chunks, 4), y (n_chunks,), groups (n_chunks,) [file-level].
    Only chunks above LEVEL_GATE are kept for training/evaluation.
    Caches features to data/processed/features.npz for fast repeat runs.
    """
    cache_file = pathlib.Path("data/processed/features.npz")
    if not smoke and cache_file.exists():
        try:
            cached = np.load(cache_file, allow_pickle=True)
            return cached["X"], cached["y"], cached["groups"]
        except Exception:
            pass

    try:
        df = pd.read_csv(manifest_path)
    except Exception:
        return np.empty((0, 4)), np.array([]), np.array([])
    if len(df) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])
    if smoke:
        df = df.groupby("class").head(5)   # max 5 files per class in smoke test

    rng = np.random.default_rng(RANDOM_SEED)
    rows_X, rows_y, rows_g = [], [], []

    for _, row in df.iterrows():
        try:
            audio, _ = librosa.load(row["path"], sr=SR, mono=True)
        except Exception as e:
            print(f"  WARN: cannot load {row['path']}: {e}")
            continue

        # Nominal ADC conversion (gain=0.5 for a mid-range default)
        adc = audio_to_adc(audio, gain=0.5, noise_sigma=0.0)
        chunks = extract_chunks(adc)
        if len(chunks) == 0:
            continue

        # Gate: keep non-silent chunks for training
        mask = chunks[:, 4] >= 2
        chunks = chunks[mask]
        if len(chunks) == 0:
            continue

        label = int(row["label"])
        group = str(row["group"])

        for c in chunks:
            rows_X.append(c[:4])    # pmin, pmax, pmean, sweep
            rows_y.append(label)
            rows_g.append(group)

        # Augmentation: apply only to training groups (CV loop will handle split)
        for aug_chunks in augment_chunks(audio, rng):
            aug_chunks = aug_chunks[aug_chunks[:, 4] >= 2]
            for c in aug_chunks:
                rows_X.append(c[:4])
                rows_y.append(label)
                rows_g.append("aug_" + group)   # separate aug group → never in held-out

    if not rows_X:
        return np.empty((0, 4)), np.array([]), np.array([])

    res_X, res_y, res_g = np.array(rows_X, dtype=np.int32), np.array(rows_y), np.array(rows_g)
    if not smoke and len(res_X) > 0:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, X=res_X, y=res_y, groups=res_g)
        except Exception:
            pass

    return res_X, res_y, res_g


def make_synthetic_data(rng: np.random.Generator) -> tuple:
    """
    Generate clearly separable synthetic features for a smoke test.
    Labelled 'synthetic' in the report; never reported as real results.
    """
    print("  Using SYNTHETIC data (smoke test only).")
    n = 120
    X, y, g = [], [], []
    # traffic: low steady pitch
    for i in range(n):
        p = int(rng.integers(50, 150))
        X.append([p, p + rng.integers(5, 20), p + rng.integers(0, 10), rng.integers(2, 10)])
        y.append(0); g.append(f"syn_traffic_{i}")
    # ambulance: oscillating pitch (high sweep)
    for i in range(n):
        p = int(rng.integers(600, 900))
        X.append([p, p + rng.integers(50, 200), p + rng.integers(25, 100), rng.integers(50, 150)])
        y.append(1); g.append(f"syn_ambulance_{i}")
    # firetruck: steady high pitch
    for i in range(n):
        p = int(rng.integers(300, 500))
        X.append([p, p + rng.integers(10, 40), p + rng.integers(5, 20), rng.integers(2, 15)])
        y.append(2); g.append(f"syn_firetruck_{i}")
    return np.array(X, dtype=np.int32), np.array(y), np.array(g)


# ---------------------------------------------------------------------------
# Tree serialisation
# ---------------------------------------------------------------------------

def tree_to_dict(tree: DecisionTreeClassifier) -> dict:
    t = tree.tree_
    return {
        "n_node_samples":  t.n_node_samples.tolist(),
        "children_left":   t.children_left.tolist(),
        "children_right":  t.children_right.tolist(),
        "feature":         t.feature.tolist(),
        "threshold":       t.threshold.tolist(),
        "value":           t.value.tolist(),
        "n_classes":       int(t.n_classes[0]),
        "n_outputs":       int(t.n_outputs),
        "feature_names":   FEATURE_NAMES,
        "class_names":     CLASSES,
        "params": {
            "max_depth":        tree.max_depth,
            "min_samples_leaf": tree.min_samples_leaf,
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    pathlib.Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Load data ---
    manifest = pathlib.Path(MANIFEST_CSV)
    use_synthetic = False

    if not manifest.exists():
        print(f"Manifest not found ({MANIFEST_CSV}). Run python -m ml.prep_data first.")
        print("Falling back to synthetic data for smoke test.")
        use_synthetic = True
    else:
        try:
            df = pd.read_csv(manifest)
            if len(df) == 0:
                print("Manifest is empty. Falling back to synthetic data.")
                use_synthetic = True
        except Exception:
            print("Manifest could not be read or is empty. Falling back to synthetic data.")
            use_synthetic = True

    if use_synthetic or args.smoke and not manifest.exists():
        X, y, groups = make_synthetic_data(rng)
    else:
        print("Loading features from manifest...")
        X, y, groups = load_features_from_manifest(str(manifest), smoke=args.smoke)
        if len(X) == 0:
            print("No chunks extracted. Using synthetic data.")
            X, y, groups = make_synthetic_data(rng)
            use_synthetic = True

    print(f"  Total chunks: {len(X)}, classes: {np.bincount(y.astype(int))}")

    # --- Grouped K-fold CV for hyper-param selection ---
    # Only non-augmented groups go to test folds (aug groups start with "aug_")
    real_mask = ~np.char.startswith(groups, "aug_")
    real_groups = groups[real_mask]
    unique_real_groups = np.unique(real_groups)

    n_splits = min(5, len(unique_real_groups))
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\nGrouped {n_splits}-fold CV over {len(unique_real_groups)} source files...")
    best_params, best_f1 = None, -1.0

    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_SAMPLES_GRID:
            fold_f1s = []
            for train_idx, test_idx in gkf.split(X[real_mask], y[real_mask], real_groups):
                Xtr, ytr = X[real_mask][train_idx], y[real_mask][train_idx]
                Xte, yte = X[real_mask][test_idx],  y[real_mask][test_idx]

                # Include augmented data in training (match to train groups)
                train_groups_set = set(real_groups[train_idx])
                aug_mask = np.array([g.lstrip("aug_") in train_groups_set
                                     if g.startswith("aug_") else False
                                     for g in groups])
                if aug_mask.any():
                    Xtr = np.vstack([Xtr, X[aug_mask]])
                    ytr = np.concatenate([ytr, y[aug_mask]])

                clf = DecisionTreeClassifier(
                    max_depth=max_depth, min_samples_leaf=min_leaf,
                    class_weight="balanced", random_state=RANDOM_SEED
                )
                clf.fit(Xtr, ytr)

                if clf.get_n_leaves() > MAX_LEAVES:
                    fold_f1s = []
                    break

                f1 = f1_score(yte, clf.predict(Xte), average="macro", zero_division=0)
                fold_f1s.append(f1)

            if not fold_f1s:
                continue
            mean_f1 = float(np.mean(fold_f1s))
            print(f"  depth={max_depth} leaf={min_leaf} → macro-F1={mean_f1:.3f}")

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = {"max_depth": max_depth, "min_samples_leaf": min_leaf}

    if best_params is None:
        print("WARNING: all configs exceed MAX_LEAVES. Using depth=3, leaf=8.")
        best_params = {"max_depth": 3, "min_samples_leaf": 8}

    print(f"\nBest: {best_params}, CV macro-F1={best_f1:.3f}")

    # --- Train/test split by group for final evaluation ---
    # Reserve ~25% of real source files for held-out test (stratified by class)
    rng2 = np.random.default_rng(RANDOM_SEED)
    unique_groups_by_class = {}
    for g, label in zip(real_groups, y[real_mask]):
        unique_groups_by_class.setdefault(label, set()).add(g)

    test_groups_set = set()
    for label, grps in unique_groups_by_class.items():
        grps_list = sorted(list(grps))
        n_test_cls = max(1, len(grps_list) // 4)
        chosen = rng2.choice(grps_list, size=n_test_cls, replace=False)
        test_groups_set.update(chosen)

    train_mask = np.array([g not in test_groups_set and not g.startswith("aug_") or
                           (g.startswith("aug_") and g.lstrip("aug_") not in test_groups_set)
                           for g in groups])
    test_mask  = np.array([g in test_groups_set for g in groups])

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask],  y[test_mask]

    print(f"  Train chunks: {len(Xtr)}, Test chunks: {len(Xte)}")

    # Final fit
    tree = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_SEED
    ).fit(Xtr, ytr)

    print(f"  Leaves: {tree.get_n_leaves()} (limit {MAX_LEAVES})")

    # Evaluate
    if len(Xte) == 0:
        print("WARNING: no held-out test chunks. Reporting train metrics (overfit risk).")
        Xte, yte = Xtr, ytr

    ypred = tree.predict(Xte)
    report_str = classification_report(yte, ypred, labels=[0, 1, 2], target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(yte, ypred, labels=[0, 1, 2])
    acc = float(np.mean(ypred == yte))
    macro_f1 = float(f1_score(yte, ypred, average="macro", zero_division=0))

    print("\n" + report_str)
    print("Confusion matrix:\n", cm)
    print("\nTree rules:\n")
    print(export_text(tree, feature_names=FEATURE_NAMES))

    # --- Save ---
    tree_dict = tree_to_dict(tree)
    pathlib.Path(TREE_JSON).write_text(json.dumps(tree_dict, indent=2), encoding="utf-8")

    metrics = {
        "held_out_accuracy":  round(acc, 4),
        "held_out_macro_f1":  round(macro_f1, 4),
        "cv_macro_f1":        round(best_f1, 4),
        "best_params":        best_params,
        "n_leaves":           int(tree.get_n_leaves()),
        "n_train_chunks":     int(len(Xtr)),
        "n_test_chunks":      int(len(Xte)),
        "classes":            CLASSES,
        "confusion_matrix":   cm.tolist(),
        "classification_report": report_str,
        "synthetic_data":     use_synthetic,
        "seed":               RANDOM_SEED,
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    pathlib.Path(METRICS_JSON).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved: {TREE_JSON}")
    print(f"Saved: {METRICS_JSON}")
    print(f"\nHeld-out accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")

    if use_synthetic:
        print("\n*** WARNING: trained on SYNTHETIC data. Do not report these as real results. ***")


if __name__ == "__main__":
    main()
