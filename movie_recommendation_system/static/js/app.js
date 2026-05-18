window.addEventListener("load", () => {
    const loader = document.getElementById("loader");
    if (loader) loader.classList.add("hidden");
    enhancePosters();
});

function titlePosterUrl(title, year = "") {
    const label = `${title || "Movie"}${year ? `\n${year}` : ""}`;
    return `https://placehold.co/420x630/111827/e5e7eb/png?text=${encodeURIComponent(label)}`;
}

function isPlaceholderPoster(url = "") {
    return !url || url.includes("placehold.co") || url.includes("picsum.photos");
}

function bindPosterFallback(image) {
    image.addEventListener("error", () => {
        image.src = titlePosterUrl(image.dataset.title, image.dataset.year);
    }, { once: true });
}

function enhancePosters() {
    document.querySelectorAll("img[data-title]").forEach((image) => {
        bindPosterFallback(image);
        if (!isPlaceholderPoster(image.src)) return;
        const params = new URLSearchParams({
            title: image.dataset.title || "",
            year: image.dataset.year || "",
            current: image.src,
            movie_id: image.dataset.movieId || "",
        });
        fetch(`/api/poster?${params.toString()}`)
            .then((response) => response.json())
            .then((data) => {
                if (data.poster && !isPlaceholderPoster(data.poster)) {
                    image.src = data.poster;
                }
            })
            .catch(() => {});
    });
}

function suggestionTemplate(movie) {
    const poster = isPlaceholderPoster(movie.poster) ? titlePosterUrl(movie.title, movie.release_year) : movie.poster;
    return `
        <div class="suggestion-item" data-title="${movie.title}">
            <img src="${poster}" alt="" loading="lazy" data-title="${movie.title}" data-year="${movie.release_year}">
            <div>
                <strong>${movie.title}</strong>
                <p class="small">${movie.release_year} | ${movie.genre} | ${Number(movie.rating).toFixed(1)}/10</p>
            </div>
        </div>`;
}

function setupAutocomplete(input, suggestionsBox, onSelect) {
    if (!input || !suggestionsBox) return;
    let timer = null;
    let requestId = 0;

    input.addEventListener("input", () => {
        clearTimeout(timer);
        const query = input.value.trim();
        if (query.length < 2) {
            suggestionsBox.innerHTML = "";
            suggestionsBox.classList.remove("open");
            return;
        }
        suggestionsBox.classList.add("open");
        suggestionsBox.innerHTML = '<div class="suggestion-loading">Searching…</div>';
        const currentRequest = ++requestId;
        timer = setTimeout(async () => {
            const response = await fetch(`/api/autocomplete?q=${encodeURIComponent(query)}`);
            if (currentRequest !== requestId) return;
            const data = await response.json();
            suggestionsBox.innerHTML = data.length
                ? data.map(suggestionTemplate).join("")
                : '<div class="suggestion-loading">No matches found</div>';
            enhancePosters();
        }, 50);
    });

    suggestionsBox.addEventListener("click", (event) => {
        const item = event.target.closest(".suggestion-item");
        if (!item) return;
        input.value = item.dataset.title;
        suggestionsBox.innerHTML = "";
        suggestionsBox.classList.remove("open");
        if (onSelect) onSelect(item.dataset.title);
    });

    document.addEventListener("click", (event) => {
        const container = input.closest(".search-box") || input.closest(".metrics-search");
        if (container && !container.contains(event.target)) {
            suggestionsBox.classList.remove("open");
        }
    });
}

setupAutocomplete(
    document.getElementById("movieSearch"),
    document.getElementById("suggestions"),
    null
);

document.querySelectorAll("form.search-panel .filters select").forEach((select) => {
    select.addEventListener("change", () => {
        const form = select.closest("form");
        if (!form || form.method.toLowerCase() !== "get") return;
        if (typeof form.requestSubmit === "function") {
            form.requestSubmit();
        } else {
            form.submit();
        }
    });
});

const chartMovieSearch = document.getElementById("chartMovieSearch");
const chartSuggestions = document.getElementById("chartSuggestions");
let chartRequestId = 0;
const chartDataCache = new Map();

async function applyChartContext(title) {
    const requestId = ++chartRequestId;
    document.body.dataset.chartTitle = title || "";
    const cards = document.getElementById("metricCards");

    if (!title) {
        chartDataCache.clear();
        const globalData = await loadChartData("");
        if (requestId !== chartRequestId) return;
        await renderMetricCharts(globalData);
        updateMetricCards(null);
        return;
    }

    if (cards) cards.classList.add("loading");
    let data = chartDataCache.get(title);
    if (!data) {
        data = await loadChartData(title);
        chartDataCache.set(title, data);
    }
    if (requestId !== chartRequestId) return;
    updateMetricCards(data);
    await renderMetricCharts(data);
    if (cards) cards.classList.remove("loading");
}

setupAutocomplete(chartMovieSearch, chartSuggestions, (title) => {
    applyChartContext(title);
});

document.getElementById("resetCharts")?.addEventListener("click", () => {
    if (chartMovieSearch) chartMovieSearch.value = "";
    applyChartContext("");
});

function updateMetricCards(data) {
    const cards = document.getElementById("metricCards");
    if (!cards) return;
    const metrics = data?.metrics || data?.summary || window.GLOBAL_METRICS;
    if (!metrics) return;
    const recLabels = {
        accuracy: "Avg Similarity",
        precision: "Recommendations",
        recall: "Avg Rating",
        f1_score: "Selected Rating",
        mean_accuracy: "Best Match",
        std_accuracy: "Lowest Match",
        total_movies: "Result Count",
        top_region: "Film Region",
    };
    const labels = data ? recLabels : (window.GLOBAL_LABELS || recLabels);
    cards.querySelectorAll(".metric-card").forEach((card) => {
        const key = card.dataset.metric;
        if (labels[key]) card.querySelector("span").textContent = labels[key];
        const value = metrics[key];
        if (value !== undefined) card.querySelector("strong").textContent = value;
    });
    if (!data) {
        const totalCard = cards.querySelector('[data-metric="total_movies"] strong');
        if (totalCard && window.GLOBAL_TOTAL !== undefined) totalCard.textContent = window.GLOBAL_TOTAL;
        const modelCard = cards.querySelector('[data-metric="top_region"] strong');
        if (modelCard) modelCard.textContent = "KNN";
    }
}

const trailerBackground = document.getElementById("trailerBackground");
document.querySelectorAll(".featured-film").forEach((film) => {
    film.addEventListener("mouseenter", () => {
        if (!trailerBackground || !film.dataset.trailer) return;
        const id = film.dataset.trailer;
        trailerBackground.src = `https://www.youtube.com/embed/${id}?autoplay=1&mute=1&controls=0&loop=1&playlist=${id}&modestbranding=1&showinfo=0`;
    });
});

function prepCanvas(canvas) {
    const wrap = canvas.parentElement;
    const width = wrap ? wrap.clientWidth : canvas.clientWidth || 300;
    const height = 260;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { context, width, height };
}

function drawBarChart(canvas, labels, values, colors) {
    if (!canvas || !labels.length || !values.length) return;
    const count = Math.min(labels.length, values.length, 10);
    labels = labels.slice(0, count);
    values = values.slice(0, count);
    const { context, width: chartWidth, height: chartHeight } = prepCanvas(canvas);
    context.clearRect(0, 0, chartWidth, chartHeight);
    const maxValue = Math.max(...values, 1);
    const barGap = Math.min(12, chartWidth / (count * 5));
    const left = 40;
    const bottom = chartHeight - 56;
    const availableWidth = chartWidth - left - 16;
    const barWidth = Math.max(14, (availableWidth - barGap * (count - 1)) / count);
    context.strokeStyle = "#dbe3ef";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left, 16);
    context.lineTo(left, bottom);
    context.lineTo(chartWidth - 8, bottom);
    context.stroke();
    values.forEach((value, index) => {
        const barHeight = (value / maxValue) * (chartHeight - 100);
        const x = left + index * (barWidth + barGap);
        const y = bottom - barHeight;
        context.fillStyle = colors[index % colors.length];
        context.fillRect(x, y, barWidth, barHeight);
        context.fillStyle = "#0f172a";
        context.font = "700 11px Segoe UI";
        context.textAlign = "center";
        context.fillText(value < 1 ? value.toFixed(2) : Math.round(value * 10) / 10, x + barWidth / 2, y - 6);
        context.fillStyle = "#475569";
        context.font = "10px Segoe UI";
        const label = String(labels[index]);
        context.save();
        context.translate(x + barWidth / 2, bottom + 8);
        context.rotate(-0.7);
        context.textAlign = "right";
        context.fillText(label.length > 18 ? `${label.slice(0, 16)}…` : label, 0, 0);
        context.restore();
    });
}

function drawMatrix(canvas, labels, matrix) {
    if (!canvas || !labels.length) return;
    const { context, width: chartWidth, height: chartHeight } = prepCanvas(canvas);
    context.clearRect(0, 0, chartWidth, chartHeight);
    const size = Math.min(chartWidth - 120, chartHeight - 60);
    const cell = size / labels.length;
    const left = 86;
    const top = 24;
    const maxValue = Math.max(...matrix.flat(), 1);
    matrix.forEach((row, rowIndex) => {
        row.forEach((value, columnIndex) => {
            const intensity = value / maxValue;
            context.fillStyle = `rgba(37, 99, 235, ${0.12 + intensity * 0.78})`;
            context.fillRect(left + columnIndex * cell, top + rowIndex * cell, cell - 2, cell - 2);
            context.fillStyle = intensity > .55 ? "#ffffff" : "#0f172a";
            context.font = "700 11px Segoe UI";
            context.textAlign = "center";
            context.fillText(value, left + columnIndex * cell + cell / 2, top + rowIndex * cell + cell / 2 + 4);
        });
    });
    context.fillStyle = "#475569";
    context.font = "11px Segoe UI";
    labels.forEach((label, index) => {
        context.textAlign = "right";
        context.fillText(label, left - 8, top + index * cell + cell / 2 + 4);
        context.save();
        context.translate(left + index * cell + cell / 2, top + size + 16);
        context.rotate(-0.55);
        context.textAlign = "right";
        context.fillText(label, 0, 0);
        context.restore();
    });
}

async function loadChartData(chartTitle = "") {
    const url = chartTitle
        ? `/api/chart-data?title=${encodeURIComponent(chartTitle)}`
        : "/api/chart-data";
    const response = await fetch(url);
    return response.json();
}

async function renderMetricCharts(cachedData) {
    const metricCanvas = document.getElementById("metricsChart");
    if (!metricCanvas) return;
    const chartTitle = document.body.dataset.chartTitle || "";
    const data = cachedData ?? await loadChartData(chartTitle);
    const panels = document.querySelectorAll(".chart-panel h3");
    if (chartTitle && data.similarity) {
        if (panels[0]) panels[0].textContent = "Similarity Scores";
        if (panels[1]) panels[1].textContent = "Shared Genres";
        if (panels[2]) panels[2].textContent = "Regions in Results";
        if (panels[3]) panels[3].textContent = "Rating Comparison";
        drawBarChart(metricCanvas, data.similarity.labels, data.similarity.values, ["#2563eb"]);
        drawBarChart(document.getElementById("confusionChart"), data.genres.labels, data.genres.values, ["#14b8a6"]);
        drawBarChart(document.getElementById("regionChart"), data.regions.labels, data.regions.values, ["#f59e0b"]);
        drawBarChart(document.getElementById("ratingChart"), data.ratings.labels, data.ratings.values, ["#ef4444", "#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]);
        return;
    }
    if (panels[0]) panels[0].textContent = "KNN Scores";
    if (panels[1]) panels[1].textContent = "Confusion Matrix";
    if (panels[2]) panels[2].textContent = "Movies by Region";
    if (panels[3]) panels[3].textContent = "Rating Distribution";
    drawBarChart(metricCanvas, data.metrics.labels, data.metrics.values, ["#2563eb", "#14b8a6", "#f59e0b", "#ef4444"]);
    drawMatrix(document.getElementById("confusionChart"), data.confusion.labels, data.confusion.matrix);
    drawBarChart(document.getElementById("regionChart"), data.regions.labels, data.regions.values, ["#2563eb"]);
    drawBarChart(document.getElementById("ratingChart"), data.ratings.labels, data.ratings.values, ["#14b8a6"]);
}

async function renderRecommendationCharts() {
    const similarityCanvas = document.getElementById("similarityChart");
    if (!similarityCanvas) return;
    const chartTitle = document.body.dataset.chartTitle || "";
    const data = chartDataCache.get(chartTitle) || await loadChartData(chartTitle);
    drawBarChart(similarityCanvas, data.similarity.labels, data.similarity.values, ["#2563eb"]);
    drawBarChart(document.getElementById("genreChart"), data.genres.labels, data.genres.values, ["#14b8a6"]);
    drawBarChart(document.getElementById("recRegionChart"), data.regions.labels, data.regions.values, ["#f59e0b"]);
    drawBarChart(document.getElementById("recRatingChart"), data.ratings.labels, data.ratings.values, ["#ef4444", "#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]);
}

renderMetricCharts();
renderRecommendationCharts();

let resizeTimer = null;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        const title = document.body.dataset.chartTitle || "";
        const data = title ? chartDataCache.get(title) : null;
        renderMetricCharts(data || undefined);
        renderRecommendationCharts();
    }, 250);
});
