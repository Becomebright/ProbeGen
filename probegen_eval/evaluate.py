"""Score a folder of predictions for one benchmark."""

from . import counting, depth, segmentation
from .data import TASKS


def evaluate(dataset, samples, data_root, predictions):
    task = {"segmentation": segmentation, "depth": depth, "counting": counting}[TASKS[dataset]]
    rows, missing, failed = [], [], []
    for sample in samples:
        path = predictions / sample.prediction
        if not path.is_file():
            missing.append(sample.id)
            continue
        try:
            rows.append({"id": sample.id, **task.score(dataset, sample, data_root, path)})
        except Exception as error:   # e.g. an unreadable prediction
            failed.append({"id": sample.id, "error": f"{type(error).__name__}: {error}"})
    return {"dataset": dataset, "selected": len(samples), "scored": len(rows),
            "metrics": task.aggregate(rows) if rows else {},
            "missing": missing, "failed": failed, "per_sample": rows}
