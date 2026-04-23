"""
Stage 1 — Dataset upload.

Registers the IMDB Sentiment CSV file as a versioned ClearML Dataset.
Run once to create Version 1; each subsequent run creates a new version.

Usage:
    python upload_dataset.py

Prerequisites:
    - ClearML server running (docker compose up -d)
    - data/imdb.csv
    - ClearML credentials configured (~/.clearml.conf)
"""
import pathlib
from clearml import Dataset

DATASET_NAME = "IMDB Sentiment"
DATASET_PROJECT = "Sentiment"
DATA_PATH = pathlib.Path(__file__).parent / "data" / "imdb.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"File not found: {DATA_PATH}"
    )

# Регистрация нового датасета или его версии в ClearML
ds = Dataset.create(
    dataset_name=DATASET_NAME,
    dataset_project=DATASET_PROJECT,
)


ds.add_files(str(DATA_PATH))

print("Uploading dataset on ClearML File Server...")
ds.upload()

# Финализация датасета -> status = Finalized
ds.finalize()

print("Dataset uploaded successfully!")
print(f"Dataset ID: {ds.id}")
