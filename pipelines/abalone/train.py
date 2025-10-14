"""Training script for SageMaker training job."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())


@dataclass
class TrainingConfig:
    chunk_size: int
    epochs: int
    max_iter: int
    random_state: int
    train_file: str
    metadata_file: str


def parse_args(args: Iterable[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=53)
    parser.add_argument("--train-file", type=str, default="train_clean.csv")
    parser.add_argument("--metadata-file", type=str, default="metadata.json")
    parsed = parser.parse_args(args=args)
    return TrainingConfig(
        chunk_size=parsed.chunk_size,
        epochs=parsed.epochs,
        max_iter=parsed.max_iter,
        random_state=parsed.random_state,
        train_file=parsed.train_file,
        metadata_file=parsed.metadata_file,
    )


def resolve_metadata_path(metadata_dir: str, clean_dir: str, metadata_file: str) -> str:
    candidates = [
        os.path.join(metadata_dir, metadata_file),
        os.path.join(clean_dir, metadata_file),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Unable to locate metadata file '{metadata_file}' in '{metadata_dir}' or '{clean_dir}'."
    )


def load_training_assets(config: TrainingConfig) -> tuple[pd.DataFrame, Dict[str, int], List[str]]:
    clean_dir = os.environ.get("SM_CHANNEL_CLEAN", ".")
    metadata_dir = os.environ.get("SM_CHANNEL_METADATA", clean_dir)

    train_path = os.path.join(clean_dir, config.train_file)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    metadata_path = resolve_metadata_path(metadata_dir, clean_dir, config.metadata_file)

    LOGGER.info("Loading cleaned training data from %s", train_path)
    train_df = pd.read_csv(train_path)

    LOGGER.info("Loading metadata from %s", metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    vocab = metadata.get("aa_vocab")
    if not vocab:
        raise KeyError("Metadata is missing required key 'aa_vocab'.")

    aa_to_idx = {aa: idx for idx, aa in enumerate(vocab)}
    return train_df, aa_to_idx, vocab


def compute_allele_max_len(train_df: pd.DataFrame) -> Dict[str, int]:
    LOGGER.info("Computing per-allele maximum sequence length")
    return (
        train_df.assign(length=train_df["peptide"].astype(str).str.len())
        .groupby("allele")["length"]
        .max()
        .astype(int)
        .to_dict()
    )


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


def train_for_allele(
    allele: str,
    train_df: pd.DataFrame,
    aa_to_idx: Dict[str, int],
    max_len: int,
    config: TrainingConfig,
) -> tuple[object, Dict[str, object]]:
    rows = train_df[train_df["allele"] == allele].reset_index(drop=True)
    n_samples = len(rows)
    LOGGER.info(
        "Training allele=%s with %d samples and max_len=%d",
        allele,
        n_samples,
        max_len,
    )
    y_full = rows["hit"].astype(int).values

    if len(np.unique(y_full)) < 2:
        LOGGER.info("Only one class present for %s; training DummyClassifier", allele)
        clf = DummyClassifier(strategy="most_frequent", random_state=config.random_state)
        clf.fit(np.zeros((1, max_len * len(aa_to_idx))), [int(y_full[0])])
    else:
        classes = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes, y=y_full)
        class_weight = {cls: weight for cls, weight in zip(classes, weights)}

        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            random_state=config.random_state,
            max_iter=config.max_iter,
            validation_fraction=0.1,
            learning_rate="optimal",
            tol=1e-3,
            n_jobs=-1,
            early_stopping=False,
        )

        rows = rows.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)

        for epoch in range(config.epochs):
            LOGGER.info("  Epoch %d/%d", epoch + 1, config.epochs)
            for start in range(0, n_samples, config.chunk_size):
                end = min(start + config.chunk_size, n_samples)
                batch = rows.iloc[start:end]
                X_batch = one_hot_encode_padded(
                    batch["peptide"].tolist(), max_len, aa_to_idx
                )
                y_batch = batch["hit"].astype(int).values
                sample_weight = np.array([class_weight[label] for label in y_batch])

                clf.fit(X_batch, y_batch, sample_weight=sample_weight)

                del X_batch, y_batch, batch, sample_weight
                gc.collect()
                LOGGER.info(
                    "    Processed batch %d covering rows %d-%d",
                    start // config.chunk_size + 1,
                    start,
                    end,
                )

            rows = rows.sample(
                frac=1.0, random_state=config.random_state + epoch + 1
            ).reset_index(drop=True)

    metadata = {
        "allele": allele,
        "n_train": int(n_samples),
        "max_len": int(max_len),
        "model_type": type(clf).__name__,
        "chunk_size": config.chunk_size,
        "epochs": config.epochs,
    }
    return clf, metadata


def main(args: Iterable[str] | None = None) -> None:
    config = parse_args(args)

    train_df, aa_to_idx, vocab = load_training_assets(config)
    allele_max_len = compute_allele_max_len(train_df)

    model_dir = os.environ.get("SM_MODEL_DIR", "./model")
    models_dir = os.path.join(model_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    manifest = {
        "aa_vocab": vocab,
        "allele_max_len": allele_max_len,
        "models": [],
    }

    for idx, allele in enumerate(sorted(allele_max_len), start=1):
        LOGGER.info(
            "[%d/%d] Starting training for allele %s",
            idx,
            len(allele_max_len),
            allele,
        )
        clf, metadata = train_for_allele(
            allele, train_df, aa_to_idx, allele_max_len[allele], config
        )
        safe_allele = (
            allele.replace("*", "").replace(":", "_").replace("/", "-")
        )
        model_path = os.path.join(models_dir, f"model_{safe_allele}.joblib")
        joblib.dump(clf, model_path)
        LOGGER.info("Saved model for %s to %s", allele, model_path)

        metadata.update({"path": os.path.relpath(model_path, model_dir)})
        manifest["models"].append(metadata)
        gc.collect()

    manifest_path = os.path.join(model_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    LOGGER.info("Training manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
