"""Generate evaluation tables and Matplotlib charts for the project report.

Run from the project root:
    python generate_report_metrics.py

Outputs:
    report/evaluation_results.txt
    report/charts/confusion_matrix.png
    report/charts/evaluation_metrics.png
    report/charts/movies_by_region.png
    report/charts/rating_distribution.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from evaluation import EvaluationMetrics
from model import KNNRecommender

REPORT_DIR = Path(__file__).resolve().parent / "report"
CHARTS_DIR = REPORT_DIR / "charts"

METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score")
METRIC_LABELS = ("Accuracy", "Precision", "Recall", "F1-Score")


def _can_stratify(y) -> bool:
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return len(counts) > 0 and int(counts.min()) >= 2


def five_fold_runs(X, y, seed: int) -> list[dict]:
    """Return per-fold Accuracy, Precision, Recall, and F1 (5 runs)."""
    runs = []
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_index, (train_index, test_index) in enumerate(folds.split(X), start=1):
        model = KNeighborsClassifier(n_neighbors=7, metric="cosine")
        model.fit(X[train_index], y.iloc[train_index])
        predictions = model.predict(X[test_index])
        y_test = y.iloc[test_index]
        runs.append(
            {
                "run": fold_index,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
                "f1_score": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
            }
        )
    return runs


def _format_percent(value: float) -> str:
    """Format a 0–1 score as a percentage (e.g. 0.986 → 98.6%, 0.9722 → 97.22%)."""
    pct = value * 100
    text = f"{pct:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def format_evaluation_report(runs: list[dict], holdout: dict) -> str:
    """Build console/report text matching the required submission layout."""
    lines = [
        "Evaluation Results",
        "Run\tAccuracy\tPrecision\tRecall\tF1-Score",
    ]
    for run in runs:
        lines.append(
            f"{run['run']}\t{run['accuracy']:.4f}\t{run['precision']:.4f}\t"
            f"{run['recall']:.4f}\t{run['f1_score']:.4f}"
        )

    lines.extend(["", "Mean ± Standard Deviation", "Metric\tMean ± Standard Deviation"])
    for key, label in zip(METRIC_KEYS, METRIC_LABELS):
        values = [float(run[key]) for run in runs]
        mean = float(np.mean(values))
        std = float(np.std(values))
        lines.append(f"{label}\t{mean:.4f} ± {std:.4f}")

    lines.extend(["", "Hold-Out Test Results"])
    for key, label in zip(METRIC_KEYS, METRIC_LABELS):
        lines.append(f"•\t{label} = {_format_percent(float(holdout[key]))}")

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    recommender = KNNRecommender()
    recommender.train()
    evaluator = EvaluationMetrics(recommender)
    summary = evaluator.evaluate()

    import matplotlib

    matplotlib.use("Agg")

    movies = recommender.movies
    X = recommender.tfidf_matrix
    region_categories = movies["region"].astype("category")
    y = region_categories.cat.codes
    max_eval_rows = min(2000, X.shape[0])
    X_eval = X[:max_eval_rows]
    y_eval = y.iloc[:max_eval_rows]
    eval_seed = summary.get("random_seed", evaluator.eval_seed)
    split_kwargs = {"test_size": 0.25, "random_state": eval_seed}
    if _can_stratify(y_eval):
        split_kwargs["stratify"] = y_eval
    _x_train, X_test, _y_train, y_test = train_test_split(X_eval, y_eval, **split_kwargs)
    classifier = KNeighborsClassifier(n_neighbors=7, metric="cosine")
    classifier.fit(_x_train, _y_train)
    predictions = classifier.predict(X_test)
    evaluator._save_charts(y_test, predictions)

    from utils import CHARTS_DIR as default_charts

    for name in (
        "confusion_matrix.png",
        "evaluation_metrics.png",
        "movies_by_region.png",
        "rating_distribution.png",
    ):
        source = default_charts / name
        if source.exists():
            target = CHARTS_DIR / name
            target.write_bytes(source.read_bytes())

    runs = five_fold_runs(X_eval, y_eval, eval_seed)
    holdout = {
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1_score": summary["f1_score"],
    }
    report_text = format_evaluation_report(runs, holdout)

    output_path = REPORT_DIR / "evaluation_results.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nCharts saved under: {CHARTS_DIR}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
