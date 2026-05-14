from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SUPPORTED_MODELS = ("random_forest", "decision_tree", "knn")

RANDOM_STATE = 42


def _build_estimator(model_type: str, params: Dict[str, Any]):
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)
    if model_type == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if model_type == "knn":
        return KNeighborsClassifier(**params)

    raise ValueError(
        f"Unknown model_type='{model_type}'. "
        f"Supported: {SUPPORTED_MODELS}"
    )


def create_pipeline(
    model_type: str = "random_forest",
    model_params: Dict[str, Any] | None = None,
) -> Pipeline:
    if model_params is None:
        model_params = {}

    estimator = _build_estimator(model_type, model_params)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )
    return pipeline


if __name__ == "__main__":
    for mtype in SUPPORTED_MODELS:
        pipe = create_pipeline(mtype)
        print(f"[{mtype}] steps: {list(pipe.named_steps.keys())}")
        print(f"           classifier: {pipe.named_steps['classifier']}\n")
