from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

OUTPUT_PATH = Path("data/raw/dataset.csv")


def create_and_save_dataset(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    iris = load_iris()

    feature_names = [
        name.replace(" (cm)", "_cm").replace(" ", "_")
        for name in iris.feature_names
    ]

    df = pd.DataFrame(iris.data, columns=feature_names)
    df["target"] = iris.target
    df["target_name"] = df["target"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[OK] Dataset saved to {output_path}")
    print(f"     Shape: {df.shape}")
    print(f"     Classes: {df['target_name'].unique().tolist()}")
    print(f"     Class distribution:\n{df['target_name'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    create_and_save_dataset()
