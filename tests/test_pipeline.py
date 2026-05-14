from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from src.models.pipeline import SUPPORTED_MODELS, create_pipeline
from src.models.utils import compute_metrics


@pytest.fixture
def iris_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=[f"f{i}" for i in range(iris.data.shape[1])])
    y = pd.Series(iris.target, name="target")
    return X, y


def test_pipeline_creation_all_models():
    for model_type in SUPPORTED_MODELS:
        pipe = create_pipeline(model_type=model_type)
        assert "scaler" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_pipeline_unknown_model_raises():
    with pytest.raises(ValueError):
        create_pipeline(model_type="xgboost")


def test_pipeline_fits_and_predicts(iris_data):
    X, y = iris_data
    pipe = create_pipeline("random_forest", {"n_estimators": 10, "max_depth": 3})
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(np.unique(y)))


def test_pipeline_with_custom_params(iris_data):
    X, y = iris_data
    pipe = create_pipeline("random_forest", {"n_estimators": 25, "max_depth": 7})
    assert pipe.named_steps["classifier"].n_estimators == 25
    assert pipe.named_steps["classifier"].max_depth == 7


def test_compute_metrics(iris_data):
    _, y = iris_data
    y_true = y.values
    y_pred = y.values  # ідеальний прогноз
    m = compute_metrics(y_true, y_pred, prefix="test_")
    assert m["test_accuracy"] == 1.0
    assert m["test_f1"] == 1.0
    assert m["test_precision"] == 1.0
    assert m["test_recall"] == 1.0


def test_no_data_leakage_in_pipeline(iris_data):
    from sklearn.model_selection import train_test_split

    X, y = iris_data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    pipe = create_pipeline("random_forest", {"n_estimators": 10})
    pipe.fit(X_tr, y_tr)

    scaler = pipe.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_tr.mean().values, rtol=1e-6)
