from __future__ import annotations

import argparse
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from src.models.pipeline import RANDOM_STATE, create_pipeline
from src.models.utils import compute_metrics, load_data, save_model_to_file


EXPERIMENT_NAME = "iris-classification"
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_OUTPUT_DIR = "models"


def train(
    model_type: str,
    model_params: Dict[str, Any],
    run_name: str | None = None,
    save_local: bool = True,
) -> Dict[str, float]:
    """
    Returns
    -------
    dict
        Зведення метрик (test_accuracy, test_f1, cv_mean, cv_std тощо).
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = create_pipeline(model_type=model_type, model_params=model_params)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_params({f"model__{k}": v for k, v in model_params.items()})

        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("stage", "experimentation")

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS, scoring="accuracy"
        )
        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = compute_metrics(y_train, train_pred, prefix="train_")
        test_metrics = compute_metrics(y_test, test_pred, prefix="test_")
        cv_metrics = {
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
        }

        all_metrics = {**train_metrics, **test_metrics, **cv_metrics}
        mlflow.log_metrics(all_metrics)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        if save_local:
            local_path = f"{MODEL_OUTPUT_DIR}/pipeline_{model_type}.pkl"
            save_model_to_file(pipeline, local_path)
            mlflow.log_artifact(local_path, artifact_path="local-pickle")

        run_id = run.info.run_id
        print(f"\n[MLflow run_id] {run_id}")
        print(f"[model_type   ] {model_type}")
        print(f"[params       ] {model_params}")
        print(f"[cv_accuracy  ] {cv_metrics['cv_mean_accuracy']:.4f} "
              f"(+/- {cv_metrics['cv_std_accuracy']:.4f})")
        print(f"[test_accuracy] {test_metrics['test_accuracy']:.4f}")
        print(f"[test_f1      ] {test_metrics['test_f1']:.4f}\n")

        return {"run_id": run_id, **all_metrics}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model with MLflow tracking")
    p.add_argument("--model", default="random_forest",
                   choices=["random_forest", "decision_tree", "knn"])
    p.add_argument("--run-name", default=None)

    # RF / DT
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--criterion", type=str, default=None)
    p.add_argument("--min-samples-split", type=int, default=None)

    # KNN
    p.add_argument("--n-neighbors", type=int, default=None)
    p.add_argument("--weights", type=str, default=None,
                   choices=["uniform", "distance", None])

    return p.parse_args()


def _args_to_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Перетворити CLI-аргументи на параметри моделі (тільки не-None)."""
    mapping = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "criterion": args.criterion,
        "min_samples_split": args.min_samples_split,
        "n_neighbors": args.n_neighbors,
        "weights": args.weights,
    }

    valid_keys = {
        "random_forest": {"n_estimators", "max_depth", "criterion", "min_samples_split"},
        "decision_tree": {"max_depth", "criterion", "min_samples_split"},
        "knn": {"n_neighbors", "weights"},
    }
    allowed = valid_keys[args.model]
    return {k: v for k, v in mapping.items() if v is not None and k in allowed}


if __name__ == "__main__":
    args = _parse_args()
    params = _args_to_params(args)
    train(model_type=args.model, model_params=params, run_name=args.run_name)
