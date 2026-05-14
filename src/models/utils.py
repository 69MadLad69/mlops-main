from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DATA_PATH = Path("data/raw/dataset.csv")
TARGET_COL = "target"
DROP_COLS = ("target", "target_name")

def load_data(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run `python -m src.data.create_dataset` first "
            f"or `dvc pull` to fetch from remote."
        )

    df = pd.read_csv(path)
    y = df[TARGET_COL]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }
    return metrics

def save_model_to_file(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved to {path}")


def load_model_from_file(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_from_mlflow(run_id: str, artifact_path: str = "model") -> Any:
    """
    >>> model = load_model_from_mlflow("a1b2c3...", "model")
    """
    import mlflow.sklearn

    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)


def predict(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(X, "values") and not isinstance(X, np.ndarray):
        pass
    return model.predict(X)


if __name__ == "__main__":
    X, y = load_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Columns: {X.columns.tolist()}")
