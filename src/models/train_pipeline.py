from __future__ import annotations

import mlflow

from src.models.train import EXPERIMENT_NAME, train

PRODUCTION_CONFIG = {
    "model_type": "random_forest",
    "params": {"n_estimators": 100, "max_depth": 5},
    "run_name": "production_pipeline",
}


def main() -> None:
    print("Training production pipeline...")
    print(f"Config: {PRODUCTION_CONFIG}")

    result = train(
        model_type=PRODUCTION_CONFIG["model_type"],
        model_params=PRODUCTION_CONFIG["params"],
        run_name=PRODUCTION_CONFIG["run_name"],
        save_local=True,
    )

    with mlflow.start_run(run_id=result["run_id"]):
        mlflow.set_tag("candidate", "production")

    print("\nNext steps:")
    print("  dvc add models/pipeline_random_forest.pkl")
    print("  git add models/pipeline_random_forest.pkl.dvc")
    print("  git commit -m 'Add trained production pipeline'")
    print("  dvc push")


if __name__ == "__main__":
    main()
