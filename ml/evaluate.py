"""
ml/evaluate.py — evaluation on held-out data, SNR curves, and figures.

Run:  python -m ml.evaluate

Produces in docs/figures/:
  - confusion_matrix.png
  - accuracy_vs_snr.png
  - pitch_scatter.png
"""

import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
)

from ml.config import (
    CLASSES, SR, RANDOM_SEED, MANIFEST_CSV, TREE_JSON, METRICS_JSON,
    FIGS_DIR, DOCS_DIR, LEVEL_GATE, SNR_MIX_DB
)
from ml.features import audio_to_adc, extract_chunks
from ml.augment import mix_traffic_noise
from ml.train import load_features_from_manifest, make_synthetic_data, tree_to_dict


def rebuild_tree(tree_json: dict) -> DecisionTreeClassifier:
    """Rebuild a DecisionTreeClassifier from the saved JSON dict."""
    p = tree_json["params"]
    clf = DecisionTreeClassifier(
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        random_state=RANDOM_SEED
    )
    # Fit on a 1-sample dummy so sklearn allocates tree_ internals
    clf.fit([[0, 0, 0, 0]], [0])

    t = clf.tree_
    import ctypes
    # We need to actually retrain from saved metrics — just reload and re-train
    return None   # signal that we need to re-train


def main():
    pathlib.Path(FIGS_DIR).mkdir(parents=True, exist_ok=True)

    # Load saved model and metrics
    if not pathlib.Path(TREE_JSON).exists():
        print(f"ERROR: {TREE_JSON} not found. Run python -m ml.train first.")
        sys.exit(1)

    metrics_dict = json.loads(pathlib.Path(METRICS_JSON).read_text(encoding="utf-8"))
    tree_dict    = json.loads(pathlib.Path(TREE_JSON).read_text(encoding="utf-8"))
    tree_params  = tree_dict["params"]
    synthetic    = metrics_dict.get("synthetic_data", False)

    rng = np.random.default_rng(RANDOM_SEED)
    manifest = pathlib.Path(MANIFEST_CSV)

    # 1. Confusion matrix (direct from metrics.json for exact script-produced result)
    if "confusion_matrix" in metrics_dict:
        cm = np.array(metrics_dict["confusion_matrix"])
    else:
        cm = np.zeros((3, 3), dtype=int)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (held-out)")
    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"Saved {FIGS_DIR}/confusion_matrix.png")

    # Helper for fast prediction using tree_dict
    def tree_predict(X_arr):
        cl = tree_dict["children_left"]
        cr = tree_dict["children_right"]
        f  = tree_dict["feature"]
        th = tree_dict["threshold"]
        v  = tree_dict["value"]
        preds = []
        for row in X_arr:
            node = 0
            while cl[node] != -1:
                if row[f[node]] <= int(np.floor(th[node])):
                    node = cl[node]
                else:
                    node = cr[node]
            preds.append(int(np.argmax(v[node][0])))
        return np.array(preds)

    if not manifest.exists() or synthetic:
        print("Using synthetic data for scatter / SNR figures.")
        X, y, _ = make_synthetic_data(rng)
        Xte, yte = X, y
        noise_clips = []
    else:
        df = pd.read_csv(manifest)
        # Identify held-out test groups deterministically (same split as train.py)
        rng2 = np.random.default_rng(RANDOM_SEED)
        unique_groups_by_class = {}
        for _, row in df.iterrows():
            unique_groups_by_class.setdefault(int(row["label"]), set()).add(str(row["group"]))

        test_groups_set = set()
        for label, grps in unique_groups_by_class.items():
            grps_list = sorted(list(grps))
            n_test_cls = max(1, len(grps_list) // 4)
            chosen = rng2.choice(grps_list, size=n_test_cls, replace=False)
            test_groups_set.update(chosen)

        # Collect traffic clips for SNR noise mixing
        noise_clips = []
        traffic_rows = df[df["class"] == "traffic"].head(20)
        for _, row in traffic_rows.iterrows():
            try:
                sig, _ = librosa.load(row["path"], sr=SR, mono=True)
                noise_clips.append(sig)
            except Exception:
                pass

        # Extract held-out test chunks for scatter plot and SNR curve
        test_df = df[df["group"].isin(test_groups_set)]
        rows_Xte, rows_yte = [], []
        print(f"Extracting features from {len(test_df)} held-out test files...")
        for _, row in test_df.iterrows():
            try:
                sig, _ = librosa.load(row["path"], sr=SR, mono=True)
                adc = audio_to_adc(sig, gain=0.5, noise_sigma=0.0)
                chunks = extract_chunks(adc)
                if len(chunks) == 0:
                    continue
                chunks = chunks[chunks[:, 4] >= 2]
                for c in chunks:
                    rows_Xte.append(c[:4])
                    rows_yte.append(int(row["label"]))
            except Exception:
                pass

        Xte = np.array(rows_Xte, dtype=np.int32)
        yte = np.array(rows_yte)
        print(f"Extracted {len(Xte)} held-out chunks for visualization.")

    # 2. Accuracy vs SNR (distance proxy) — only if real audio data
    snr_accs = {}
    if manifest.exists() and not synthetic and noise_clips:
        print("Computing accuracy vs SNR...")
        df = pd.read_csv(manifest)
        rng3 = np.random.default_rng(RANDOM_SEED)

        # Preload a stratified sample of test audio into memory (up to 12 clips per class)
        test_audio_cache = []
        test_df = df[df["group"].isin(test_groups_set)]
        sampled_test = test_df.groupby("class").head(12)
        for _, row in sampled_test.iterrows():
            try:
                sig, _ = librosa.load(row["path"], sr=SR, mono=True)
                test_audio_cache.append((sig, int(row["label"])))
            except Exception:
                pass

        for snr in SNR_MIX_DB:
            preds, trues = [], []
            for sig, label in test_audio_cache:
                mixed = mix_traffic_noise(sig, noise_clips, snr, rng3)
                adc = audio_to_adc(mixed, gain=0.5)
                chunks = extract_chunks(adc)
                if len(chunks) == 0:
                    continue
                chunks = chunks[chunks[:, 4] >= 2]
                if len(chunks) == 0:
                    continue
                for c in chunks:
                    preds.append(tree_predict([c[:4]])[0])
                    trues.append(label)
            if preds:
                snr_accs[snr] = accuracy_score(trues, preds)
                print(f"  SNR {snr:>3} dB → acc={snr_accs[snr]:.3f}")
    else:
        # Synthetic: show constant line
        for snr in SNR_MIX_DB:
            snr_accs[snr] = float(np.mean(ypred == yte))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(list(snr_accs.keys()), list(snr_accs.values()), marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR (distance proxy)")
    ax.set_ylim(0, 1.05)
    if synthetic:
        ax.set_title("Accuracy vs SNR (SYNTHETIC — not real)")
    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/accuracy_vs_snr.png", dpi=120)
    plt.close(fig)
    print(f"Saved {FIGS_DIR}/accuracy_vs_snr.png")

    # 3. Pitch feature scatter plot (pmean vs sweep)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "^", "s"]
    for label, (cls, col, mk) in enumerate(zip(CLASSES, colors, markers)):
        mask = yte == label
        ax.scatter(Xte[mask, 2], Xte[mask, 3], c=col, marker=mk,
                   label=cls, alpha=0.5, s=20)
    ax.set_xlabel("pmean (Hz)")
    ax.set_ylabel("sweep (Hz/window)")
    ax.set_title("Pitch features — held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS_DIR}/pitch_scatter.png", dpi=120)
    plt.close(fig)
    print(f"Saved {FIGS_DIR}/pitch_scatter.png")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
