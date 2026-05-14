from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.models.train import train

EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "model_type": "random_forest",
        "params": {"n_estimators": 50, "max_depth": 3},
        "run_name": "rf_small",
    },
    {
        "model_type": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 5},
        "run_name": "rf_medium",
    },
    {
        "model_type": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 4},
        "run_name": "rf_large",
    },

    {
        "model_type": "decision_tree",
        "params": {"max_depth": 3, "criterion": "gini"},
        "run_name": "dt_shallow_gini",
    },
    {
        "model_type": "decision_tree",
        "params": {"max_depth": 10, "criterion": "entropy"},
        "run_name": "dt_deep_entropy",
    },

    {
        "model_type": "knn",
        "params": {"n_neighbors": 3, "weights": "uniform"},
        "run_name": "knn_k3_uniform",
    },
    {
        "model_type": "knn",
        "params": {"n_neighbors": 7, "weights": "distance"},
        "run_name": "knn_k7_distance",
    },
]


def main() -> None:
    print("=" * 70)
    print(f"Running {len(EXPERIMENTS)} experiments...")
    print("=" * 70)

    results = []
    for i, exp in enumerate(EXPERIMENTS, start=1):
        print(f"\n--- Experiment {i}/{len(EXPERIMENTS)}: {exp['run_name']} ---")
        result = train(
            model_type=exp["model_type"],
            model_params=exp["params"],
            run_name=exp["run_name"],
            save_local=False,
        )
        results.append({
            "run_name": exp["run_name"],
            "model_type": exp["model_type"],
            "params": str(exp["params"]),
            "run_id": result["run_id"],
            "cv_mean": result["cv_mean_accuracy"],
            "cv_std": result["cv_std_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "test_f1": result["test_f1"],
        })

    df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by test_accuracy)")
    print("=" * 70)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(f"Run name      : {best['run_name']}")
    print(f"Model type    : {best['model_type']}")
    print(f"Params        : {best['params']}")
    print(f"Test accuracy : {best['test_accuracy']:.4f}")
    print(f"Test F1       : {best['test_f1']:.4f}")
    print(f"CV accuracy   : {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})")
    print(f"MLflow run_id : {best['run_id']}")
    print("\nUse MLflow UI to inspect: mlflow ui --port 5000")

    df.to_csv("models/experiments_summary.csv", index=False)
    print("\n[OK] Summary saved to models/experiments_summary.csv")


if __name__ == "__main__":
    main()
