#!/usr/bin/env bash
# setup.sh — швидке розгортання проєкту з нуля.
# Виконує всі кроки з README, розділ 3.

set -e

echo "==> 1. Створення venv"
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 2. Установлення залежностей"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> 3. Ініціалізація Git"
if [ ! -d .git ]; then
    git init
    git add .gitignore requirements.txt README.md src/ tests/ setup.sh
    git commit -m "Initial commit: project scaffold" || true
fi

echo "==> 4. Ініціалізація DVC"
if [ ! -d .dvc ]; then
    dvc init
    git add .dvc .dvcignore
    git commit -m "Initialize DVC" || true
fi

echo "==> 5. Локальний DVC remote"
mkdir -p /tmp/dvc-storage
if ! dvc remote list | grep -q myremote; then
    dvc remote add -d myremote /tmp/dvc-storage
    git add .dvc/config
    git commit -m "Configure DVC local remote" || true
fi

echo "==> 6. Створення датасету"
python -m src.data.create_dataset
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Add Iris dataset under DVC" || true
dvc push

echo "==> 7. Запуск тестів"
pytest tests/ -q

echo ""
echo "================================================================"
echo "Setup завершено."
echo ""
echo "Далі:"
echo "  • MLflow UI:           mlflow ui --port 5000"
echo "  • Усі експерименти:    python -m src.experiments.run_experiments"
echo "  • Production pipeline: python -m src.models.train_pipeline"
echo "================================================================"
