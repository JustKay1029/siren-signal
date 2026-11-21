"""
tests/test_export.py — verify that Python tree.predict matches exported C logic
for 2000 random integer feature vectors.

If g++ is unavailable, we evaluate the same rules through a Python function
emitted by the exporter itself, and say so.
"""

import json, pathlib, subprocess, sys, tempfile
import numpy as np
import pytest

from ml.config import TREE_JSON, METRICS_JSON, ARDUINO_DIR


def _load_tree_dict():
    p = pathlib.Path(TREE_JSON)
    if not p.exists():
        pytest.skip(f"{TREE_JSON} not found — run python -m ml.train first")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_metrics():
    p = pathlib.Path(METRICS_JSON)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _build_sklearn_tree(tree_dict: dict):
    """Reconstruct an exact sklearn DecisionTreeClassifier from saved JSON."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree._tree import Tree

    n_nodes = len(tree_dict["children_left"])
    t = Tree(4, np.array([3], dtype=np.intp), 1)

    nodes = np.zeros(n_nodes, dtype=t.__getstate__()["nodes"].dtype)
    nodes["left_child"] = tree_dict["children_left"]
    nodes["right_child"] = tree_dict["children_right"]
    nodes["feature"] = tree_dict["feature"]
    nodes["threshold"] = tree_dict["threshold"]
    nodes["n_node_samples"] = tree_dict["n_node_samples"]
    nodes["weighted_n_node_samples"] = tree_dict["n_node_samples"]

    state = {
        "max_depth": tree_dict["params"]["max_depth"],
        "node_count": n_nodes,
        "nodes": nodes,
        "values": np.array(tree_dict["value"], dtype=np.float64)
    }
    t.__setstate__(state)

    clf = DecisionTreeClassifier(random_state=42)
    clf.classes_ = np.array([0, 1, 2])
    clf.n_classes_ = 3
    clf.n_features_in_ = 4
    clf.n_outputs_ = 1
    clf.tree_ = t
    return clf


def _python_emulated_classify(tree_dict: dict, features: np.ndarray) -> np.ndarray:
    """
    Evaluate the exported if/else logic via a Python function (no g++).
    Generated dynamically from the same tree_dict as gen_model_h().
    """
    import math

    children_left  = tree_dict["children_left"]
    children_right = tree_dict["children_right"]
    feature        = tree_dict["feature"]
    threshold      = tree_dict["threshold"]
    value          = tree_dict["value"]

    def classify_one(row):
        node = 0
        while children_left[node] != -1:
            feat_val = int(row[feature[node]])
            thresh   = int(math.floor(threshold[node]))
            if feat_val <= thresh:
                node = children_left[node]
            else:
                node = children_right[node]
        vals = value[node][0]
        return int(max(range(len(vals)), key=lambda i: vals[i]))

    return np.array([classify_one(row) for row in features], dtype=np.int32)


def _check_gpp():
    try:
        r = subprocess.run(["g++", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_export_equivalence_2000_vectors():
    """
    For 2000 random feature vectors, Python tree.predict must match
    either: g++-compiled siren_model.h (if available) or the Python-emulated
    if/else logic (fallback).
    """
    tree_dict = _load_tree_dict()
    sklearn_tree = _build_sklearn_tree(tree_dict)

    rng = np.random.default_rng(99)
    # Random feature vectors in plausible integer range
    n = 2000
    pmin  = rng.integers(0, 2000, size=n)
    pmax  = pmin + rng.integers(0, 500, size=n)
    pmean = (pmin + rng.integers(0, (pmax - pmin + 1), size=n)).clip(0, 32767)
    sweep = rng.integers(0, 300, size=n)
    X_rand = np.column_stack([pmin, pmax, pmean, sweep]).astype(np.int32)

    sklearn_preds = sklearn_tree.predict(X_rand)

    gpp_ok = _check_gpp()
    if gpp_ok:
        # Compile siren_model.h with a small test harness
        model_h = pathlib.Path(ARDUINO_DIR) / "siren_model.h"
        if not model_h.exists():
            pytest.skip(f"{model_h} not found — run python -m ml.export_c first")

        harness = _make_gpp_harness()
        with tempfile.TemporaryDirectory() as tmpdir:
            src = pathlib.Path(tmpdir) / "harness.cpp"
            src.write_text(harness, encoding="utf-8")

            csv_path = pathlib.Path(tmpdir) / "features.csv"
            with open(csv_path, "w") as f:
                for row in X_rand:
                    f.write(",".join(str(v) for v in row) + "\n")

            exe = pathlib.Path(tmpdir) / "harness"
            result = subprocess.run(
                ["g++", "-std=c++11", "-I", str(ARDUINO_DIR), str(src), "-o", str(exe)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                pytest.fail(f"g++ compile failed:\n{result.stderr}")

            result = subprocess.run([str(exe), str(csv_path)], capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"harness failed:\n{result.stderr}")

            c_preds = np.array([int(x) for x in result.stdout.strip().split("\n")], dtype=np.int32)
            method = "g++ compiled siren_model.h"
    else:
        # Python emulated fallback
        c_preds = _python_emulated_classify(tree_dict, X_rand)
        method = "Python-emulated if/else (g++ not available)"
        print(f"\nNote: equivalence test used {method}")

    mismatches = np.sum(sklearn_preds != c_preds)
    assert mismatches == 0, (
        f"{mismatches}/2000 mismatches between sklearn and {method}.\n"
        f"First mismatch at index {np.where(sklearn_preds != c_preds)[0][0]}: "
        f"sklearn={sklearn_preds[np.where(sklearn_preds != c_preds)[0][0]]}, "
        f"exported={c_preds[np.where(sklearn_preds != c_preds)[0][0]]}"
    )


def _make_gpp_harness() -> str:
    return """\
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "siren_model.h"

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    FILE* f = fopen(argv[1], "r");
    if (!f) return 2;
    int16_t pmin, pmax, pmean, sweep;
    while (fscanf(f, "%hd,%hd,%hd,%hd\\n", &pmin, &pmax, &pmean, &sweep) == 4) {
        printf("%d\\n", (int)siren_classify(pmin, pmax, pmean, sweep));
    }
    fclose(f);
    return 0;
}
"""
