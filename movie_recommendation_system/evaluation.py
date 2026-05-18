"""Evaluation metrics and Matplotlib charts for the recommender."""

from __future__ import annotations

import secrets

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

from model import KNNRecommender
from utils import CHARTS_DIR, ensure_directories


def _can_stratify(y) -> bool:
    """Return True when every class has at least two samples for stratified splitting."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return len(counts) > 0 and int(counts.min()) >= 2


def _new_evaluation_seed() -> int:
    """Fresh seed each evaluation so global metrics change on every app/script run."""
    return secrets.randbelow(2**31 - 1) or 1


def recommendation_chart_data(selected: dict, recommendations: list[dict]) -> dict:
    """Build chart payloads that reflect the current recommendation set."""
    rec_titles = [str(movie.get("title", "")) for movie in recommendations]
    short_titles = []
    for index, title in enumerate(rec_titles, start=1):
        clean = title[:16] + ("…" if len(title) > 16 else "")
        short_titles.append(f"#{index} {clean}")
    similarities = [float(movie.get("similarity", 0)) for movie in recommendations]

    genre_counter: Counter[str] = Counter()
    for movie in [selected, *recommendations]:
        for genre in str(movie.get("genre", "")).split("|"):
            label = genre.strip()
            if label:
                genre_counter[label] += 1
    top_genres = genre_counter.most_common(8)

    region_counter = Counter(str(movie.get("region", "Unknown")) for movie in recommendations)
    selected_region = str(selected.get("region", "Unknown")).strip() or "Unknown"
    rating_labels = [str(selected.get("title", "Selected"))[:18], *short_titles[:6]]
    rating_values = [float(selected.get("rating", 0)), *[float(movie.get("rating", 0)) for movie in recommendations[:6]]]

    avg_similarity = round(sum(similarities) / len(similarities), 2) if similarities else 0.0
    avg_rating = round(
        sum(float(movie.get("rating", 0)) for movie in recommendations) / len(recommendations),
        2,
    ) if recommendations else 0.0
    top_region = selected_region

    return {
        "context_title": str(selected.get("title", "")),
        "summary": {
            "accuracy": avg_similarity,
            "precision": len(recommendations),
            "recall": avg_rating,
            "f1_score": float(selected.get("rating", 0)),
            "mean_accuracy": max(similarities) if similarities else 0.0,
            "std_accuracy": min(similarities) if similarities else 0.0,
            "total_movies": len(recommendations) + 1,
            "top_region": top_region,
        },
        "similarity": {
            "labels": short_titles,
            "values": similarities,
        },
        "genres": {
            "labels": [label for label, _count in top_genres],
            "values": [count for _label, count in top_genres],
        },
        "regions": {
            "labels": [selected_region, *list(region_counter.keys())[:7]],
            "values": [1, *list(region_counter.values())[:7]],
        },
        "ratings": {
            "labels": rating_labels,
            "values": rating_values,
        },
    }


class EvaluationMetrics:
    """Compute Accuracy, Precision, Recall, F1, confusion matrix, and charts."""

    def __init__(self, recommender: KNNRecommender):
        self.recommender = recommender
        self.results = {}
        self.eval_seed: int | None = None

    def evaluate(self) -> dict:
        ensure_directories()
        movies = self.recommender.movies
        X = self.recommender.tfidf_matrix
        region_categories = movies["region"].astype("category")
        y = region_categories.cat.codes
        self.class_labels = list(region_categories.cat.categories)
        self.eval_seed = _new_evaluation_seed()
        max_eval_rows = min(2000, X.shape[0])
        X = X[:max_eval_rows]
        y = y.iloc[:max_eval_rows]
        split_kwargs = {"test_size": 0.25, "random_state": self.eval_seed}
        if _can_stratify(y):
            split_kwargs["stratify"] = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        classifier = KNeighborsClassifier(n_neighbors=7, metric="cosine")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        self.results = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
            "random_seed": self.eval_seed,
        }
        self.results.update(self._cross_validation(X, y))
        self.results["chart_data"] = self._chart_data(y_test, predictions)
        return self.results

    def _chart_data(self, y_test, predictions) -> dict:
        """Build chart data for the browser canvas charts (tied to the current eval seed)."""
        labels = getattr(self, "class_labels", sorted(self.recommender.movies["region"].unique()))
        matrix = confusion_matrix(y_test, predictions, labels=list(range(len(labels))))
        rating_bins = [0, 0, 0, 0, 0]
        for rating in self.recommender.movies["rating"]:
            if rating < 5:
                rating_bins[0] += 1
            elif rating < 6:
                rating_bins[1] += 1
            elif rating < 7:
                rating_bins[2] += 1
            elif rating < 8:
                rating_bins[3] += 1
            else:
                rating_bins[4] += 1
        return {
            "metrics": {
                "labels": ["Accuracy", "Precision", "Recall", "F1"],
                "values": [self.results["accuracy"], self.results["precision"], self.results["recall"], self.results["f1_score"]],
            },
            "regions": {
                "labels": self.recommender.movies["region"].value_counts().index.tolist(),
                "values": self.recommender.movies["region"].value_counts().astype(int).tolist(),
            },
            "ratings": {
                "labels": ["<5", "5-5.9", "6-6.9", "7-7.9", "8+"],
                "values": rating_bins,
            },
            "confusion": {
                "labels": labels,
                "matrix": matrix.astype(int).tolist(),
            },
        }

    def _cross_validation(self, X, y) -> dict:
        scores = []
        folds = KFold(n_splits=5, shuffle=True, random_state=self.eval_seed)
        for train_index, test_index in folds.split(X):
            model = KNeighborsClassifier(n_neighbors=7, metric="cosine")
            model.fit(X[train_index], y.iloc[train_index])
            fold_prediction = model.predict(X[test_index])
            scores.append(accuracy_score(y.iloc[test_index], fold_prediction))
        return {
            "mean_accuracy": round(float(np.mean(scores)), 4),
            "std_accuracy": round(float(np.std(scores)), 4),
        }

    def _save_charts(self, y_test, predictions) -> None:
        labels = getattr(self, "class_labels", sorted(self.recommender.movies["region"].unique()))
        matrix = confusion_matrix(y_test, predictions, labels=list(range(len(labels))))
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax = plt.subplots(figsize=(9, 6.4), facecolor="white")
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title("KNN Region Classification Confusion Matrix", fontsize=15, weight="bold", pad=14)
        ax.set_xlabel("Predicted Region", fontsize=11)
        ax.set_ylabel("Actual Region", fontsize=11)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                ax.text(column_index, row_index, value, ha="center", va="center", color="#0f172a", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "confusion_matrix.png", dpi=160)
        fig.savefig(CHARTS_DIR / "confusion_matrix.svg")
        plt.close()

        fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor="white")
        values = [self.results["accuracy"], self.results["precision"], self.results["recall"], self.results["f1_score"]]
        bars = ax.bar(["Accuracy", "Precision", "Recall", "F1"], values, color=["#2563eb", "#14b8a6", "#f59e0b", "#ef4444"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("KNN Model Evaluation Scores", fontsize=15, weight="bold", pad=14)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=10, weight="bold")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "evaluation_metrics.png", dpi=160)
        fig.savefig(CHARTS_DIR / "evaluation_metrics.svg")
        plt.close()

        region_counts = self.recommender.movies["region"].value_counts()
        fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor="white")
        region_counts.plot(kind="bar", color="#2563eb", ax=ax)
        ax.set_title("Catalog Coverage by Region", fontsize=15, weight="bold", pad=14)
        ax.set_xlabel("Region")
        ax.set_ylabel("Movie Count")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "movies_by_region.png", dpi=160)
        fig.savefig(CHARTS_DIR / "movies_by_region.svg")
        plt.close()

        fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor="white")
        self.recommender.movies["rating"].plot(kind="hist", bins=18, color="#14b8a6", edgecolor="#0f172a", ax=ax)
        ax.set_title("Movie Rating Distribution", fontsize=15, weight="bold", pad=14)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "rating_distribution.png", dpi=160)
        fig.savefig(CHARTS_DIR / "rating_distribution.svg")
        plt.close()
