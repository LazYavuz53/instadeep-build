"""Evaluation script for per-allele classifiers."""
from __future__ import annotations

import json
import logging
import pathlib
import tarfile
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())


def one_hot_encode_padded(seqs: List[str], max_len: int, aa_to_idx: Dict[str, int]) -> np.ndarray:
    n = len(seqs)
    width = len(aa_to_idx)
    encoded = np.zeros((n, max_len * width), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = str(seq).strip().upper()
        for j, aa in enumerate(seq[:max_len]):
            idx = aa_to_idx.get(aa)
            if idx is not None:
                encoded[i, j * width + idx] = 1.0
    return encoded


def safe_predict_proba(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        if proba.ndim == 2:
            return proba[:, 0]
        return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1 / (1 + np.exp(-scores))
    predictions = model.predict(features)
    return predictions.astype(float)


def main(args: Iterable[str] | None = None) -> None:
    del args  # unused - script does not accept CLI args

    LOGGER.info("Extracting model artifacts")
    model_tar_path = "/opt/ml/processing/model/model.tar.gz"
    extract_dir = pathlib.Path("/opt/ml/processing/model_artifacts")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    manifest_path = extract_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest.json not found in model artifacts")

    LOGGER.info("Loading manifest from %s", manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    aa_vocab = manifest["aa_vocab"]
    aa_to_idx = {aa: idx for idx, aa in enumerate(aa_vocab)}
    allele_max_len = manifest.get("allele_max_len", {})

    test_path = pathlib.Path("/opt/ml/processing/test/test.csv")
    if not test_path.is_file():
        raise FileNotFoundError("Test split not found; expected /opt/ml/processing/test/test.csv")

    LOGGER.info("Loading test data from %s", test_path)
    test_df = pd.read_csv(test_path)
    if not {"peptide", "allele", "hit"}.issubset(test_df.columns):
        raise ValueError("Test data must contain columns: peptide, allele, hit")

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_proba_all: List[float] = []

    model_index = {entry["allele"]: entry["path"] for entry in manifest.get("models", [])}

    for allele, group in test_df.groupby("allele"):
        relative_path = model_index.get(allele)
        if relative_path is None:
            LOGGER.warning("No trained model for allele %s; skipping %d rows", allele, len(group))
            continue

        model_path = extract_dir / relative_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file missing: {model_path}")

        LOGGER.info("Evaluating allele %s using model %s", allele, model_path)
        model = joblib.load(model_path)
        max_len = allele_max_len.get(allele)
        if max_len is None:
            raise KeyError(f"Missing max_len for allele {allele} in manifest")

        features = one_hot_encode_padded(group["peptide"].tolist(), max_len, aa_to_idx)
        y_true = group["hit"].astype(int).to_numpy()
        probabilities = safe_predict_proba(model, features)
        predictions = (probabilities >= 0.5).astype(int)

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(predictions.tolist())
        y_proba_all.extend(probabilities.tolist())

    if not y_true_all:
        raise RuntimeError("No predictions were generated; ensure test set shares alleles with training data.")

    y_true_arr = np.asarray(y_true_all)
    y_pred_arr = np.asarray(y_pred_all)
    y_proba_arr = np.asarray(y_proba_all)

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    balanced_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)

    try:
        auc = roc_auc_score(y_true_arr, y_proba_arr)
    except ValueError:
        auc = float("nan")

    try:
        loss = log_loss(y_true_arr, np.clip(y_proba_arr, 1e-7, 1 - 1e-7))
    except ValueError:
        loss = float("nan")

    error_std = float(np.std(y_true_arr - y_pred_arr))

    report = {
        "classification_metrics": {
            "accuracy": {"value": float(accuracy), "standard_deviation": error_std},
            "balanced_accuracy": {"value": float(balanced_acc)},
            "roc_auc": {"value": float(auc)},
            "log_loss": {"value": float(loss)},
        }
    }

    output_dir = pathlib.Path("/opt/ml/processing/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    LOGGER.info("Evaluation report written to %s", output_path)


if __name__ == "__main__":
    main()
