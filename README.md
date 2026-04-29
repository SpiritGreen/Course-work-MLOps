# Sentiment Analysis MLOps Pipeline

Курсовой проект по MLOps. Полный жизненный цикл ML-модели через [ClearML](https://clear.ml/): версионирование данных, обучение через агента, Model Registry, inference endpoint, веб-интерфейс.

**Задача:** анализ тональности отзывов IMDB — `positive` / `negative`.
**Модель:** TF-IDF + LogisticRegression (scikit-learn).

---

## Структура проекта

```
Course work/
├── docker/
│   ├── docker-compose.yaml      # ClearML Server (6 сервисов)
│   └── clearml_config/
├── data/
│   └── imdb.csv                 # Датасет (50k отзывов)
├── upload_dataset.py            # Этап 1: загрузка датасета в ClearML
├── train.py                     # Этап 2: обучение через агента
├── register_model.py            # Этап 3: публикация в Model Registry
├── preprocess.py                # Этап 4: хук препроцессинга для serving
├── service_check.py             # Этап 4: smoke-тест inference endpoint
├── app.py                       # Этап 5: Streamlit UI
└── requirements.txt
```

---

## Требования

- Python 3.11
- Docker Desktop (Windows)

```bash
pip install -r requirements.txt
```

---

## Запуск

### 1. Поднять ClearML Server

```bash
cd docker
docker compose up -d
```

| Сервис   | URL                   |
|----------|-----------------------|
| Web UI   | http://localhost:8080 |
| API      | http://localhost:8008 |
| Files    | http://localhost:8081 |

Первичная настройка SDK (один раз):
```bash
clearml-init
```

### 2. Запустить ClearML Agent

```bash
clearml-agent daemon --queue students --foreground
```

Агент запускается в **venv-режиме** — задачи выполняются в виртуальном окружении на хосте. Надо держать терминал с агентом открытым на всё время обучения.

### 3. Этап 1 — Загрузить датасет

```bash
python upload_dataset.py
```

Датасет появится в ClearML UI → Datasets.

### 4. Этап 2 — Обучить модель

```bash
python train.py
```

Скрипт отправляет задачу в очередь `students`; агент обучает модель и логирует метрики (accuracy, F1, confusion matrix) в UI → Experiments.

Для запуска с другими гиперпараметрами — надо отредактировать в UI и нажать **Clone → Enqueue**.

### 5. Этап 3 — Зарегистрировать модель

```bash
python register_model.py
```

Выбирает лучший эксперимент по F1 и публикует модель в ClearML UI → Models.

### 6. Этап 4 — Запустить Inference Endpoint

Установить CLI:
```bash
pip install clearml-serving
```

Создать serving controller:
```bash
clearml-serving create --name "Sentiment Serving"
```

Зарегистрировать модель на endpoint:
```bash
clearml-serving --id <SERVICE_ID> model add \
  --engine sklearn \
  --endpoint sentiment \
  --preprocess preprocess.py \
  --model-id <MODEL_ID>
```

Запустить inference сервер (Windows, Docker Desktop):
```bash
docker run -d \
  --name clearml-serving-inference \
  --network clearml_frontend \
  -p 8890:8080 \
  --add-host=localhost:host-gateway \
  -e CLEARML_SERVING_TASK_ID="<SERVICE_ID>" \
  -e CLEARML_API_HOST="http://apiserver:8008" \
  -e CLEARML_FILES_HOST="http://localhost:8081" \
  -e CLEARML_EXTRA_PYTHON_PACKAGES="scikit-learn==1.8.0" \
  -e CLEARML_API_ACCESS_KEY="<key>" \
  -e CLEARML_API_SECRET_KEY="<secret>" \
  allegroai/clearml-serving-inference:latest
```

Проверка:
```bash
python service_check.py
```

### 7. Этап 5 — Запустить UI

```bash
streamlit run app.py
```

Открыть в браузере: http://localhost:8501

---

## Примечания по инфраструктуре (Windows)

- `--network clearml_frontend` — serving-контейнер должен быть в одной сети с ClearML-сервисами, иначе не сможет скачать артефакты.
- `--add-host=localhost:host-gateway` — artifact URL хранится как `http://localhost:8081/...`; без этого флага контейнер не может его разрезолвить.
- `CLEARML_EXTRA_PYTHON_PACKAGES` — версия sklearn в контейнере должна совпадать с версией, которой обучали модель.
