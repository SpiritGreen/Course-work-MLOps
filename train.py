"""
Stage 2 — Model training via ClearML Agent.

Trains a TF-IDF + LogisticRegression pipeline on the IMDB Sentiment dataset.
Submits the task to the 'students' queue for execution by the ClearML Agent.

Usage:
    python train.py

The script stops at task.execute_remotely() and the agent re-runs it remotely.
To run locally without an agent (for debugging), comment out execute_remotely().
"""
from pathlib import Path

import joblib
import pandas as pd

from clearml import Dataset, Logger, OutputModel, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# === ClearML Task ===

task = Task.init(
    project_name="Sentiment",
    task_name="TF-IDF LogReg",
    output_uri=True,   # разрешение загрузки артефактов на File Server
)

# можно переопределить через UI
args = {
    "dataset_id": "893db0b2dfd54b7683ed764596f48b11",
    "max_features": 5000,   # сколько уникальных слов оставить в TF-IDF словаре
    "C": 1.0,               # регуляризация LogReg
    "test_size": 0.2,
    "random_state": 42,
}
task.connect(args)  # после этой строки агент перепишет args своими значениями

# === Отправка задачу агенту ===
task.execute_remotely(queue_name="students")

# === Загрузка данных ===

dataset = Dataset.get(dataset_id=args["dataset_id"])
local_path = Path(dataset.get_local_copy())

df = pd.read_csv(local_path / "imdb.csv")
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=args["test_size"],
    random_state=args["random_state"],
    stratify=y,   # сохранение пропорций классов в обеих выборках
)

# === Обучение ===

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=int(args["max_features"]))),
    ("logreg", LogisticRegression(C=args["C"], max_iter=1000)),
])
pipeline.fit(X_train, y_train)

# === Метрики ===

logger: Logger = task.get_logger()
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label="positive")

logger.report_scalar("accuracy", "test", value=float(acc), iteration=0)
logger.report_scalar("f1", "test", value=float(f1), iteration=0)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
logger.report_confusion_matrix(
    title="Confusion Matrix",
    series="test",
    matrix=cm,
    xlabels=["positive", "negative"],
    ylabels=["positive", "negative"],
)

# Classification report
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).T
logger.report_table("Classification Report", "test", table_plot=report_df)

print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

# === Сохранение модели ===

output_model = OutputModel(task=task, framework="scikit-learn")
joblib.dump(pipeline, "model.pkl", compress=True)
output_model.update_weights(weights_filename="model.pkl")

print("Done. Model saved as artifact.")
