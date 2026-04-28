"""
Stage 3 — Model Registry.

Publishes the best experiment's model to ClearML Model Registry.
Automatically selects the task with the highest accuracy among completed
training runs. Set task_id to override and publish a specific experiment.

Usage:
    python register_model.py
"""
import re

from clearml import Model, Task

task = Task.init(project_name="Sentiment", task_name="Register Best Model")

args = {
    "task_id":       "",          # оставить пустым для автовыбора лучшего эксперимента по f1
    "metric":        "f1",
    "series":        "test",
    "min_metric":    0.80,        # quality gate: модель не публикуется ниже порога
    "model_version": "",          # оставить пустым для автоинкремента
}
task.connect(args)

logger = task.get_logger()


def _get_metric(t, metric, series):
    try:
        return t.get_reported_scalars()[metric][series]["y"][-1]
    except (KeyError, IndexError):
        return 0.0


def _next_version(project_name):
    all_models = Model.query_models(project_name=project_name)
    max_v = 0
    for m in all_models:
        for tag in (m.tags or []):
            match = re.fullmatch(r"v(\d+)", tag)
            if match:
                max_v = max(max_v, int(match.group(1)))
    return f"v{max_v + 1}"


if args["task_id"]:
    source_task = Task.get_task(task_id=args["task_id"])
    print(f"Using task specified in args: {source_task.id}")
else:
    completed = Task.get_tasks(
        project_name="Sentiment",
        task_name="TF-IDF LogReg",
        task_filter={"status": ["completed"]},
    )
    if not completed:
        raise RuntimeError("No completed 'TF-IDF LogReg' tasks found in project 'Sentiment'")

    source_task = max(completed, key=lambda t: _get_metric(t, args["metric"], args["series"]))
    print(f"Auto-selected best task: {source_task.id}")

best_score = _get_metric(source_task, args["metric"], args["series"])
print(f"  {args['metric']}={best_score:.4f}")

if best_score < args["min_metric"]:
    raise RuntimeError(
        f"Quality gate failed: {args['metric']}={best_score:.4f} < min={args['min_metric']}"
    )

logger.report_scalar(args["metric"], "selected_model", value=best_score, iteration=0)

output_models = source_task.get_models()["output"]
if not output_models:
    raise RuntimeError(f"No output models found in task {source_task.id}")

model = output_models[0]
print(f"Found model: id={model.id}  name={model.name}")

if model.published:
    print("Model is already published — skipping publish().")
else:
    model.publish()

prev_best = Model.query_models(project_name="Sentiment", tags=["best"])
for prev in prev_best:
    if prev.id != model.id:
        prev.tags = [t for t in (prev.tags or []) if t not in ("best", "production")]
        print(f"Demoted previous best model: {prev.id}")

VERSION = args["model_version"] or _next_version("Sentiment")
print(f"Model version: {VERSION}")

model.tags = ["best", "production", "tfidf-logreg", VERSION]

print("Model published successfully.")
print(f"Model ID : {model.id}")
print(f"Model URL: {model.url}")

task.close()
