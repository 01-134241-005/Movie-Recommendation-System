"""Flask user interface routes."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request, send_from_directory

from evaluation import EvaluationMetrics, recommendation_chart_data
from recommender import MovieRecommendationService
from utils import BASE_DIR


class MovieUI:
    """Encapsulates all web routes for the movie website."""

    def __init__(self, app: Flask):
        self.app = app
        self.service = MovieRecommendationService()
        self.metrics = EvaluationMetrics(self.service.model).evaluate()
        self.register_routes()

    def register_routes(self) -> None:
        self.app.add_url_rule("/", "index", self.index, methods=["GET", "POST"])
        self.app.add_url_rule("/movie/<path:title>", "movie_details", self.movie_details, methods=["GET"])
        self.app.add_url_rule("/metrics", "project_metrics", self.project_metrics, methods=["GET"])
        self.app.add_url_rule("/api/chart-data", "chart_data", self.chart_data, methods=["GET"])
        self.app.add_url_rule("/api/poster", "poster_lookup", self.poster_lookup, methods=["GET"])
        self.app.add_url_rule("/architecture.svg", "architecture_file", self.architecture_file, methods=["GET"])
        self.app.add_url_rule("/api/autocomplete", "autocomplete", self.autocomplete, methods=["GET"])

    def index(self):
        filters = self._filters()
        query = request.values.get("movie_title", "").strip()
        options = self.service.filter_options()
        if query and request.values.get("recommend"):
            selected, recommendations = self.service.recommendations_for(query, filters)
            return render_template(
                "movie_details.html",
                selected=selected,
                recommendations=recommendations,
                options=options,
                filters=filters,
                metrics=self.metrics,
                featured=self.service.featured(),
            )
        movies = self.service.home_movies(filters)
        hero = self.service.featured()[0]
        return render_template(
            "index.html",
            movies=movies,
            hero=hero,
            featured=self.service.featured(),
            options=options,
            filters=filters,
            metrics=self.metrics,
        )

    def movie_details(self, title: str):
        filters = self._filters()
        selected, recommendations = self.service.recommendations_for(title, filters)
        return render_template(
            "movie_details.html",
            selected=selected,
            recommendations=recommendations,
            options=self.service.filter_options(),
            filters=filters,
            metrics=self.metrics,
            featured=self.service.featured(),
        )

    def project_metrics(self):
        """Show academic requirement proof without cluttering the movie UI."""
        return render_template(
            "metrics.html",
            metrics=self.metrics,
            options=self.service.filter_options(),
            total_movies=len(self.service.model.movies),
        )

    def autocomplete(self):
        query = request.args.get("q", "")
        return jsonify(self.service.suggestions(query))

    def chart_data(self):
        title = request.args.get("title", "").strip()
        if title:
            selected, recommendations = self.service.recommendations_for(title, self._filters(), fast=True)
            payload = recommendation_chart_data(selected, list(recommendations))
            payload["metrics"] = payload.pop("summary", {})
            return jsonify(payload)
        return jsonify(self.metrics["chart_data"])

    def poster_lookup(self):
        title = request.args.get("title", "").strip()
        year = request.args.get("year", "")
        current = request.args.get("current", "")
        if not title:
            return jsonify({"poster": ""})
        movie_id = request.args.get("movie_id", "")
        poster = self.service.poster_for(title, year, current)
        if poster and "placehold.co" in poster:
            poster = self.service.posters.resolve_online(title, year, current, movie_id)
            self.service.posters.save_cache()
        return jsonify({"poster": poster})

    def architecture_file(self):
        return send_from_directory(BASE_DIR, "architecture.svg")

    def _filters(self) -> dict:
        return {
            "genre": request.values.get("genre", ""),
            "region": request.values.get("region", ""),
            "year": request.values.get("year", ""),
            "language": request.values.get("language", ""),
            "rating": request.values.get("rating", ""),
        }
