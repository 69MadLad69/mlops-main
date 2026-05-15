# MLOps Lab 1: Iris Classification (Variant 1)

**Дисципліна:** MLOps / Розробка ML-систем у веб-додатках
**Варіант:** 1 — Iris Dataset + Random Forest (основна модель) + Decision Tree, KNN (альтернативи)

---

**Автор:** Авраменко Олег Вячеславович
**Група:** ТР-51мп 
**Група:** 15.05.2026 

## 1. Опис проєкту

Навчальний проєкт демонструє повний MLOps-цикл на класичній задачі класифікації трьох видів ірисів:

- **версіонування коду** через Git;
- **версіонування даних і моделей** через DVC;
- **відстеження експериментів** через MLflow Tracking;
- **інкапсуляція обробки + моделі** через `sklearn.pipeline.Pipeline`;
- **повна відтворюваність** через фіксовані `random_state`, закріплені версії пакетів та DVC-чексуми.

Натреновано 7 моделей (3 × Random Forest, 2 × Decision Tree, 2 × KNN) із вибором найкращої за метрикою `test_accuracy` / `test_f1`.

---

## 2. Структура проєкту

```
mlops-lab1-iris/
├── data/
│   ├── raw/                       # сирий датасет (під DVC)
│   ├── processed/                 # оброблені дані (під DVC)
│   └── external/                  # дані з зовнішніх джерел
├── models/                        # збережені pickle-файли pipeline (під DVC)
├── notebooks/                     # дослідні Jupyter-нотбуки
├── src/
│   ├── data/
│   │   └── create_dataset.py      # завантаження Iris -> CSV
│   ├── features/                  # (заготовка під feature engineering)
│   ├── models/
│   │   ├── pipeline.py            # фабрика sklearn Pipeline для 3 моделей
│   │   ├── train.py               # CLI-тренування з MLflow
│   │   ├── train_pipeline.py      # тренування production-кандидата
│   │   └── utils.py               # load_data / load_model / metrics
│   └── experiments/
│       └── run_experiments.py     # batch-запуск усіх 7 експериментів
├── tests/
│   └── test_pipeline.py           # pytest-тести (6 шт.)
├── .gitignore
├── .dvcignore
├── requirements.txt
└── README.md
```

---

## 3. Швидкий старт

### 3.1 Передумови
- Python ≥ 3.10
- Git
- DVC (ставиться разом з `requirements.txt`)

### 3.2 Установлення

```bash
git clone <URL-репозиторію>

python -m venv .venv #py -3.11 -m venv .venv   #if you have newer version
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3.3 Ініціалізація Git + DVC (тільки при першому створенні проєкту)

```bash
git init
git add .gitignore requirements.txt README.md src/ tests/
git commit -m "Initial commit"

dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Локальний remote (для навчальних цілей). Для команди — Google Drive / S3.
mkdir -p /tmp/dvc-storage
dvc remote add -d myremote /tmp/dvc-storage
git add .dvc/config
git commit -m "Configure DVC local remote"
```

### 3.4 Отримання даних

Якщо клонували вже існуючий репозиторій:
```bash
dvc pull
```

Якщо створюєте з нуля:
```bash
python -m src.data.create_dataset
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc data/raw/.gitignore
mkdir D:\dvc-storage
dvc remote add -d myremote D:/dvc-storage
git commit -m "Add Iris dataset under DVC"
dvc push
```

### 3.5 Запуск MLflow UI

```bash
mlflow ui --port 5000
```
Далі відкрити `http://localhost:5000`.

### 3.6 Тренування

**Усі експерименти варіанту (7 запусків):**
```bash
python -m src.experiments.run_experiments
```

**Окремий запуск з кастомними параметрами:**
```bash
python -m src.models.train --model random_forest --n-estimators 100 --max-depth 5
python -m src.models.train --model decision_tree --max-depth 3 --criterion gini
python -m src.models.train --model knn --n-neighbors 7 --weights distance
```

**Тренування фінального production-pipeline + збереження для DVC:**
```bash
python -m src.models.train_pipeline

dvc add models/pipeline_random_forest.pkl
git add models/pipeline_random_forest.pkl.dvc
git commit -m "Add trained production pipeline"
dvc push
```

### 3.7 Тести

```bash
pytest tests/ -q
```

---

## 4. Опис датасету

| Параметр | Значення |
|---|---|
| Джерело | `sklearn.datasets.load_iris` |
| К-сть зразків | 150 |
| К-сть фіч | 4 (sepal_length, sepal_width, petal_length, petal_width — усі в см) |
| Цільова змінна | `target` (0=setosa, 1=versicolor, 2=virginica) |
| Баланс класів | 50 / 50 / 50 (ідеально збалансований) |
| Розподіл train/test | 80% / 20%, stratified, `random_state=42` |
| Cross-validation | 5-fold на train |

---

## 5. Результати експериментів

Усі 7 запусків залоговані в MLflow під експериментом **`iris-classification`**.
Сортування — за `test_accuracy` (спадання).

| # | Run name | Модель | Параметри | CV mean ± std | Test accuracy | Test F1 |
|---|---|---|---|---|---|---|
| 1 | `knn_k7_distance` | KNN | `n_neighbors=7, weights=distance` | 0.9583 ± 0.0373 | **1.0000** | **1.0000** |
| 2 | `rf_large` | Random Forest | `n_estimators=200, max_depth=10, min_samples_split=4` | 0.9500 ± 0.0167 | 0.9667 | 0.9666 |
| 3 | `rf_small` | Random Forest | `n_estimators=50, max_depth=3` | 0.9583 ± 0.0000 | 0.9667 | 0.9666 |
| 4 | `dt_shallow_gini` | Decision Tree | `max_depth=3, criterion=gini` | 0.9333 ± 0.0204 | 0.9667 | 0.9666 |
| 5 | `rf_medium` | Random Forest | `n_estimators=100, max_depth=5` | 0.9500 ± 0.0167 | 0.9333 | 0.9333 |
| 6 | `dt_deep_entropy` | Decision Tree | `max_depth=10, criterion=entropy` | 0.9250 ± 0.0486 | 0.9333 | 0.9333 |
| 7 | `knn_k3_uniform` | KNN | `n_neighbors=3, weights=uniform` | 0.9583 ± 0.0373 | 0.9333 | 0.9327 |

### 5.1 Найкраща модель

| Поле | Значення |
|---|---|
| **Run name** | `knn_k7_distance` |
| **Модель** | KNeighborsClassifier |
| **Параметри** | `n_neighbors=7, weights='distance'` |
| **Test accuracy** | **1.0000** |
| **Test F1 (weighted)** | **1.0000** |
| **CV mean accuracy** | 0.9583 ± 0.0373 |

**Зауваження.** Test accuracy = 1.0 досягнута лише на одній фіксованій test-вибірці (30 зразків). Cross-validation показує реалістичніше значення ~0.96. Іris — дуже простий датасет, тому всі моделі тримаються в діапазоні 0.93–1.00.

### 5.2 Аналіз
- **Random Forest** дає стабільні результати (CV std ≤ 0.017) — мала дисперсія, передбачуваний production-кандидат.
- **Decision Tree** з малою глибиною (3) працює нарівні з RF, але `max_depth=10` починає переучуватись (нижчий CV).
- **KNN** дуже чутливий до масштабу — без StandardScaler у Pipeline результати були б помітно гіршими.

---

## 6. Використання натренованої моделі

### 6.1 З локального pickle
```python
from src.models.utils import load_model_from_file, predict
import pandas as pd

model = load_model_from_file("models/pipeline_random_forest.pkl")
X_new = pd.DataFrame([{
    "sepal_length_cm": 5.1, "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4, "petal_width_cm": 0.2,
}])
print(predict(model, X_new))   # -> [0]  (setosa)
```

### 6.2 З MLflow Model Registry
```python
from src.models.utils import load_model_from_mlflow

model = load_model_from_mlflow(run_id="<RUN_ID_З_ТАБЛИЦІ>")
predictions = model.predict(X_new)
```

---

## 7. Відтворюваність

Усі фактори, що впливають на результат, зафіксовані:

| Що | Як зафіксовано |
|---|---|
| Версії пакетів | `requirements.txt` з точними версіями (`==`) |
| Версія датасету | DVC-хеш у `data/raw/dataset.csv.dvc` |
| Версія моделі | DVC-хеш у `models/pipeline_*.pkl.dvc` |
| Розбиття train/test | `random_state=42`, `stratify=y` |
| `random_state` моделей | 42 (у `pipeline.RANDOM_STATE`) |
| Гіперпараметри | логуються у MLflow `params` |
| Метрики | логуються у MLflow `metrics` |
| Код | Git-commit, на який посилається `mlflow.source.git.commit` |

Щоб відтворити будь-який експеримент на іншій машині:
```bash
git clone <repo>
git checkout <commit-hash>
pip install -r requirements.txt
dvc pull
python -m src.experiments.run_experiments
```

---

## 8. Стратегія Git-гілок

- `main` — стабільна гілка, в неї потрапляють лише перевірені результати.
- `develop` — поточна розробка.
- `feature/<name>` — окремі експерименти або фічі (`feature/add-knn`, `feature/hyperparam-tuning`).

Робочий цикл:
```bash
git checkout -b feature/new-experiment
# … код, експерименти …
git add . && git commit -m "Add new experiment"
git checkout develop && git merge feature/new-experiment
```

---

## 9. Troubleshooting

| Проблема | Рішення |
|---|---|
| `FileNotFoundError: data/raw/dataset.csv` | Виконати `dvc pull` або `python -m src.data.create_dataset` |
| `mlflow: command not found` | Активувати venv: `source .venv/bin/activate` |
| MLflow UI порожній | Запускати команди з кореня проєкту (де є `mlruns/`) |
| `dvc push` помилка з правами | Перевірити, чи існує директорія remote (`/tmp/dvc-storage`) і чи є запис до неї |
| Інші результати ніж у README | Перевірити: чи зафіксовано `random_state=42` у `pipeline.py`, чи зроблено `dvc pull` для даних |
| `WARNING: artifact_path is deprecated` від MLflow | Безпечне попередження сумісності, на результати не впливає |
| Тести `pytest` не знаходять модулі | Запускати з кореня проєкту, переконатись що є `src/__init__.py` |