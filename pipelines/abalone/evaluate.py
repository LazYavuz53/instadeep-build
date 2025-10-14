"""Evaluation script for peptide binding classifiers."""
from __future__ import annotations

import json
import logging
import os
import tarfile
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())


BASE_DIR = "/opt/ml/processing"
MODEL_TAR_PATH = os.path.join(BASE_DIR, "model", "model.tar.gz")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
META_DIR = os.path.join(BASE_DIR, "metadata")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

TEST_CLEAN_PATH = os.path.join(CLEAN_DIR, "test_clean.csv")
META_PATH = os.path.join(META_DIR, "metadata.json")
MANIFEST_PATH = os.path.join(MODEL_DIR, "manifest.json")


def extract_model_tarball(model_tar_path: str, target_dir: str) -> None:
    """Extract the compressed model bundle produced by SageMaker training."""

    if not os.path.isfile(model_tar_path):
        raise FileNotFoundError(f"Model archive not found at {model_tar_path}")

    LOGGER.info("Extracting model artifacts from %s", model_tar_path)
    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path=target_dir)


def one_hot_encode_padded(seqs: List[str], max_len: int, aa_to_idx: Dict[str, int]) -> np.ndarray:
    """One-hot encode peptides with zero padding up to ``max_len``."""

    n = len(seqs)
    width = len(aa_to_idx)
    encoded = np.zeros((n, max_len * width), dtype=np.float32)

    for i, seq in enumerate(seqs):
        seq = str(seq).strip().upper()
        L = min(len(seq), max_len)
        for j in range(L):
            aa = seq[j]
            idx = aa_to_idx.get(aa)
            if idx is not None:
                encoded[i, j * width + idx] = 1.0

    return encoded


def safe_allele_name(allele: str) -> str:
    return allele.replace("*", "").replace(":", "_").replace("/", "-")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    extract_model_tarball(MODEL_TAR_PATH, MODEL_DIR)

    LOGGER.info("Loading cleaned test data from %s", TEST_CLEAN_PATH)
    test_df = pd.read_csv(TEST_CLEAN_PATH)

    LOGGER.info("Loading metadata from %s", META_PATH)
    with open(META_PATH, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    LOGGER.info("Loading training manifest from %s", MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    aa_vocab: List[str] = metadata.get("aa_vocab", [])
    if not aa_vocab:
        raise KeyError("Metadata missing 'aa_vocab'.")
    aa_to_idx = {aa: idx for idx, aa in enumerate(aa_vocab)}

    allele_max_len: Dict[str, int] = manifest.get("allele_max_len", {})
    allele_to_model: Dict[str, str] = {
        m["allele"]: os.path.join(MODEL_DIR, m["path"])
        for m in manifest.get("models", [])
    }

    LOGGER.info("Loaded %d test rows covering %d alleles", len(test_df), test_df["allele"].nunique())
    LOGGER.info("Found trained models for %d alleles", len(allele_to_model))

    have_labels = "hit" in test_df.columns and not test_df["hit"].isna().any()

    predictions: List[pd.DataFrame] = []
    metrics: List[Dict[str, object]] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_all: List[float] = []

    for allele in sorted(test_df["allele"].unique()):
        if allele not in allele_to_model:
            LOGGER.warning("Skipping allele %s: no trained model available", allele)
            continue

        allele_rows = test_df[test_df["allele"] == allele]
        max_len = allele_max_len.get(allele, metadata.get("max_seq_len", 15))
        LOGGER.info(
            "Evaluating allele=%s with %d samples (max_len=%d)",
            allele,
            len(allele_rows),
            max_len,
        )

        features = one_hot_encode_padded(allele_rows["peptide"].tolist(), max_len, aa_to_idx)
        model_path = allele_to_model[allele]
        clf = joblib.load(model_path)

        expected = getattr(clf, "n_features_in_", features.shape[1])
        if features.shape[1] != expected:
            LOGGER.warning(
                "Skipping allele=%s due to feature size mismatch (expected=%d, found=%d)",
                allele,
                expected,
                features.shape[1],
            )
            continue

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(features)[:, 1]
        elif hasattr(clf, "decision_function"):
            decision = clf.decision_function(features)
            decision_min = decision.min()
            decision_range = decision.max() - decision_min + 1e-12
            prob = (decision - decision_min) / decision_range
        else:
            prob = clf.predict(features).astype(float)

        pred = (prob >= 0.5).astype(int)

        allele_out = allele_rows[["peptide", "allele"]].copy()
        allele_out["y_prob"] = prob
        allele_out["y_pred"] = pred
        if have_labels:
            allele_out["hit"] = allele_rows["hit"].values
        predictions.append(allele_out)

        allele_filename = f"predictions_{safe_allele_name(allele)}.csv"
        allele_path = os.path.join(PREDICTIONS_DIR, allele_filename)
        allele_out.to_csv(allele_path, index=False)
        LOGGER.info("Saved per-allele predictions to %s", allele_path)

        if have_labels:
            y_true = allele_rows["hit"].astype(int).values
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, prob)
                except ValueError:
                    auc = float("nan")
                acc = accuracy_score(y_true, pred)
                f1 = f1_score(y_true, pred, zero_division=0)
                metrics.append(
                    {
                        "allele": allele,
                        "n_test": len(allele_rows),
                        "acc": acc,
                        "auc": auc,
                        "f1": f1,
                    }
                )
                y_true_all.extend(y_true.tolist())
                y_pred_all.extend(pred.tolist())
                y_prob_all.extend(prob.tolist())
            else:
                metrics.append(
                    {
                        "allele": allele,
                        "n_test": len(allele_rows),
                        "acc": None,
                        "auc": None,
                        "f1": None,
                    }
                )

    if predictions:
        combined_predictions = pd.concat(predictions, ignore_index=True)
    else:
        columns = ["peptide", "allele", "y_prob", "y_pred"]
        if have_labels:
            columns.append("hit")
        combined_predictions = pd.DataFrame(columns=columns)

    predictions_path = os.path.join(OUTPUT_DIR, "predictions_per_allele.csv")
    combined_predictions.to_csv(predictions_path, index=False)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics_per_allele.csv")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    overall_metrics = {"auc": None, "acc": None, "f1": None}
    if have_labels and y_true_all and len(set(y_true_all)) > 1:
        try:
            overall_metrics["auc"] = roc_auc_score(y_true_all, y_prob_all)
        except ValueError:
            overall_metrics["auc"] = float("nan")
        overall_metrics["acc"] = accuracy_score(y_true_all, y_pred_all)
        overall_metrics["f1"] = f1_score(y_true_all, y_pred_all, zero_division=0)
    else:
        overall_metrics = {"auc": None, "acc": None, "f1": None}

    overall_path = os.path.join(OUTPUT_DIR, "metrics_overall.json")
    with open(overall_path, "w", encoding="utf-8") as handle:
        json.dump(overall_metrics, handle, indent=2)

    LOGGER.info("Saved combined predictions to %s", predictions_path)
    LOGGER.info("Saved per-allele metrics to %s", metrics_path)
    LOGGER.info("Saved overall metrics to %s", overall_path)
    LOGGER.info("Overall metrics: %s", overall_metrics)

    accuracy_value = overall_metrics["acc"] if overall_metrics["acc"] is not None else 0.0
    evaluation_report = {
        "classification_metrics": {
            "accuracy": {"value": accuracy_value},
            "roc_auc": {"value": overall_metrics["auc"]},
            "f1": {"value": overall_metrics["f1"]},
        }
    }

    evaluation_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(evaluation_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation_report, handle)

    LOGGER.info("Evaluation report written to %s", evaluation_path)


if __name__ == "__main__":
    main()
