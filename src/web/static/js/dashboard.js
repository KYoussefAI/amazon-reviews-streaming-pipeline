const config = window.APP_CONFIG || {};
const REQUIRED_PRODUCT_ID = config.requiredProductId || "B001E4KFG0";
const SOURCE_NAME = config.sourceName || "spark_structured_streaming";

let refreshTimer = null;

const colorMap = {
    positive: "var(--positive)",
    negative: "var(--negative)",
    neutral: "var(--neutral)",
    unknown: "var(--unknown)",
    one_vs_rest_linear_svc: "var(--accent)",
    logistic_regression: "var(--positive)",
    naive_bayes: "var(--neutral)",
    majority_vote_ensemble: "var(--purple)",
};

function formatNumber(value) {
    if (value === null || value === undefined) return "N/A";
    return Number(value).toLocaleString();
}

function formatPercent(value) {
    if (value === null || value === undefined) return "N/A";
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function getActiveProductId() {
    const typed = document.getElementById("productTextFilter").value.trim();
    const selected = document.getElementById("productSelectFilter").value;

    if (typed) return typed;
    return selected || "All";
}

function getFilters() {
    return {
        sentiment: document.getElementById("sentimentFilter").value,
        score: document.getElementById("scoreFilter").value,
        batch_id: document.getElementById("batchFilter").value,
        product_id: getActiveProductId(),
        model_type: document.getElementById("modelTypeFilter").value,
        source_split: document.getElementById("sourceSplitFilter").value,
        start_date: document.getElementById("startDateFilter").value,
        end_date: document.getElementById("endDateFilter").value,
        limit: document.getElementById("limitFilter").value,
        source: SOURCE_NAME,
    };
}

function buildQueryString() {
    const params = new URLSearchParams(getFilters());
    return params.toString();
}

async function fetchJson(url) {
    const response = await fetch(url);

    if (!response.ok) {
        const text = await response.text();
        throw new Error(`${response.status}: ${text}`);
    }

    return response.json();
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function badge(label) {
    const safe = label || "unknown";
    return `<span class="badge badge-${safe}">${safe}</span>`;
}

function renderBars(containerId, rows, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!rows || rows.length === 0) {
        container.innerHTML = `<div class="empty-state">No data available</div>`;
        return;
    }

    const labelKey = options.labelKey || "label";
    const valueKey = options.valueKey || "count";
    const maxValue = Math.max(...rows.map(row => Number(row[valueKey]) || 0), 1);

    container.innerHTML = rows.map(row => {
        const label = row[labelKey] ?? "unknown";
        const value = Number(row[valueKey]) || 0;
        const width = Math.max((value / maxValue) * 100, value > 0 ? 1 : 0);
        const color = colorMap[label] || options.color || "var(--accent)";

        return `
            <div class="bar-row">
                <div title="${label}">${label}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${width}%; background:${color};"></div>
                </div>
                <div class="bar-value">${formatNumber(value)}</div>
            </div>
        `;
    }).join("");
}

function renderConfidenceBars(containerId, rows, valueKey = "count") {
    renderBars(containerId, rows, {
        labelKey: "label",
        valueKey,
        color: "var(--accent)",
    });
}

function renderScoreBars(rows) {
    const normalized = [1, 2, 3, 4, 5].map(score => {
        const found = rows.find(row => Number(row.score) === score);
        return {
            label: String(score),
            count: found ? found.count : 0,
        };
    });

    renderBars("scoreBars", normalized, {
        labelKey: "label",
        valueKey: "count",
        color: "var(--accent)",
    });
}

function renderScoreBySentiment(rows) {
    const container = document.getElementById("scoreSentimentBars");
    if (!container) return;

    const grouped = {};

    [1, 2, 3, 4, 5].forEach(score => {
        grouped[score] = {
            score,
            positive: 0,
            negative: 0,
            neutral: 0,
            unknown: 0,
            total: 0,
        };
    });

    rows.forEach(row => {
        const score = Number(row.score);
        const label = row.predicted_label || "unknown";

        if (!grouped[score]) return;

        grouped[score][label] = (grouped[score][label] || 0) + row.count;
        grouped[score].total += row.count;
    });

    container.innerHTML = Object.values(grouped).map(row => {
        const total = row.total || 1;

        return `
            <div class="stacked-row">
                <div>Score ${row.score}</div>
                <div class="stacked-track">
                    <div class="stack-positive" style="width:${(row.positive / total) * 100}%"></div>
                    <div class="stack-negative" style="width:${(row.negative / total) * 100}%"></div>
                    <div class="stack-neutral" style="width:${(row.neutral / total) * 100}%"></div>
                    <div class="stack-unknown" style="width:${(row.unknown / total) * 100}%"></div>
                </div>
                <div class="bar-value">${formatNumber(row.total)}</div>
            </div>
        `;
    }).join("");
}

function renderDateChart(containerId, rows, valueKey = "total") {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!rows || rows.length === 0) {
        container.innerHTML = `<div class="empty-state">No date data available</div>`;
        return;
    }

    const maxValue = Math.max(...rows.map(row => Number(row[valueKey]) || 0), 1);

    container.innerHTML = rows.map(row => {
        const value = Number(row[valueKey]) || 0;
        const height = Math.max((value / maxValue) * 245, value > 0 ? 2 : 0);
        const title = row.review_date
            ? `${row.review_date}: ${value}`
            : `Batch ${row.batch_id}: ${value}`;

        return `<div class="date-bar" title="${title}" style="height:${height}px;"></div>`;
    }).join("");
}

function renderBatchSummary(rows) {
    const container = document.getElementById("batchSummary");
    if (!container) return;

    if (!rows || rows.length === 0) {
        container.innerHTML = `<div class="empty-state">No batch data available</div>`;
        return;
    }

    const records = rows.map(row => row.records || 0);
    const total = records.reduce((acc, value) => acc + value, 0);
    const avg = total / rows.length;
    const min = Math.min(...records);
    const max = Math.max(...records);
    const latest = rows[rows.length - 1];

    const summary = [
        ["Total micro-batches shown", rows.length],
        ["Total records shown", total],
        ["Average records per batch", avg.toFixed(2)],
        ["Minimum records in batch", min],
        ["Maximum records in batch", max],
        ["Latest batch ID", latest.batch_id],
    ];

    container.innerHTML = summary.map(([label, value]) => `
        <div class="summary-row">
            <span>${label}</span>
            <strong>${formatNumber(value)}</strong>
        </div>
    `).join("");
}

function renderConfusionMatrix(rows) {
    const container = document.getElementById("confusionMatrix");
    if (!container) return;

    const labels = ["positive", "negative", "neutral"];
    const lookup = {};

    (rows || []).forEach(row => {
        lookup[`${row.true_label}|${row.predicted_label}`] = row.count;
    });

    let html = `<div class="matrix-grid">`;
    html += `<div class="matrix-cell header">True \\ Pred</div>`;
    labels.forEach(label => html += `<div class="matrix-cell header">${label}</div>`);

    labels.forEach(trueLabel => {
        html += `<div class="matrix-cell header">${trueLabel}</div>`;

        labels.forEach(predLabel => {
            const count = lookup[`${trueLabel}|${predLabel}`] || 0;
            const className = trueLabel === predLabel ? "good" : "bad";
            html += `<div class="matrix-cell ${className}"><strong>${formatNumber(count)}</strong></div>`;
        });
    });

    html += `</div>`;
    container.innerHTML = html;
}

function renderProductAnalysis(data) {
    const container = document.getElementById("productAnalysis");

    if (!data || data.total_predictions === 0 || !data.latest) {
        container.innerHTML = `<div class="empty-state">No prediction found yet for this ProductId.</div>`;
        return;
    }

    const latest = data.latest;
    const single = data.total_predictions === 1;

    const cards = single
        ? [
            ["Product Predictions", data.total_predictions],
            ["Review Score", latest.score],
            ["True Label", latest.true_label],
            ["Predicted Label", latest.predicted_label],
            ["Confidence", latest.confidence_display],
            ["Review Date", latest.review_date],
            ["Model Type", latest.model_type],
            ["Batch ID", latest.batch_id],
        ]
        : [
            ["Product Predictions", data.total_predictions],
            ["Latest Score", latest.score],
            ["Latest Predicted Label", latest.predicted_label],
            ["Average Confidence", data.average_confidence_display],
            ["Streamed Accuracy", data.streamed_accuracy_display],
            ["Latest Review Date", latest.review_date],
            ["Model Type", latest.model_type],
            ["Latest Batch ID", latest.batch_id],
        ];

    container.innerHTML = `
        <p class="hint">
            ${single
                ? "This ProductId currently has one streamed prediction. The cards below describe that exact event."
                : "This ProductId has multiple streamed predictions. The cards combine latest event values and aggregate product metrics."
            }
        </p>

        <div class="product-cards">
            ${cards.map(([label, value]) => `
                <div class="product-card">
                    <span>${label}</span>
                    <strong>${value ?? "N/A"}</strong>
                </div>
            `).join("")}
        </div>

        <div class="grid-2">
            <article>
                <h3>Product Sentiment Distribution</h3>
                <div id="productSentimentBars" class="bar-chart"></div>
            </article>

            <article>
                <h3>Product Score Distribution</h3>
                <div id="productScoreBars" class="bar-chart"></div>
            </article>
        </div>
    `;

    renderBars("productSentimentBars", data.sentiment_distribution || []);
    renderBars(
        "productScoreBars",
        (data.score_distribution || []).map(row => ({
            label: String(row.score),
            count: row.count,
        })),
        { color: "var(--accent)" }
    );
}

function renderLatestTable(rows) {
    const body = document.getElementById("latestTableBody");
    if (!body) return;

    if (!rows || rows.length === 0) {
        body.innerHTML = `<tr><td colspan="11" class="empty-state">No prediction documents found.</td></tr>`;
        return;
    }

    body.innerHTML = rows.map(row => `
        <tr>
            <td>${row.processed_at || ""}</td>
            <td>${row.product_id || ""}</td>
            <td>${row.review_date || ""}</td>
            <td>${row.score ?? ""}</td>
            <td>${badge(row.true_label)}</td>
            <td>${badge(row.predicted_label)}</td>
            <td>${row.confidence_display || "N/A"}</td>
            <td>${row.model_type || "unknown"}</td>
            <td>${row.source_split || ""}</td>
            <td>${row.batch_id ?? ""}</td>
            <td>${row.text_preview || ""}</td>
        </tr>
    `).join("");
}

function renderRiskTable(id, rows, type) {
    const body = document.getElementById(id);
    if (!body) return;

    if (!rows || rows.length === 0) {
        body.innerHTML = `<tr><td colspan="6" class="empty-state">No samples found.</td></tr>`;
        return;
    }

    body.innerHTML = rows.map(row => {
        if (type === "low") {
            return `
                <tr>
                    <td>${row.processed_at || ""}</td>
                    <td>${row.score ?? ""}</td>
                    <td>${badge(row.predicted_label)}</td>
                    <td>${row.confidence_display || "N/A"}</td>
                    <td>${row.product_id || ""}</td>
                    <td>${row.text_preview || ""}</td>
                </tr>
            `;
        }

        return `
            <tr>
                <td>${row.processed_at || ""}</td>
                <td>${row.score ?? ""}</td>
                <td>${badge(row.true_label)}</td>
                <td>${badge(row.predicted_label)}</td>
                <td>${row.product_id || ""}</td>
                <td>${row.text_preview || ""}</td>
            </tr>
        `;
    }).join("");
}

async function loadOptions() {
    const data = await fetchJson("/api/options");

    setText("requiredProductIdLabel", data.required_product_id);
    setText("requiredProductCode", data.required_product_id);
    setText("sourceNameLabel", data.source_name);

    const productSelect = document.getElementById("productSelectFilter");
    productSelect.innerHTML = `<option value="All">All</option>` + (data.products || [])
        .map(product => `<option value="${product}">${product}</option>`)
        .join("");

    const batchSelect = document.getElementById("batchFilter");
    batchSelect.innerHTML = `<option value="All">All</option>` + (data.batches || [])
        .map(batch => `<option value="${batch}">${batch}</option>`)
        .join("");

    const modelSelect = document.getElementById("modelTypeFilter");
    modelSelect.innerHTML = `<option value="All">All</option>` + (data.model_types || [])
        .map(model => `<option value="${model}">${model}</option>`)
        .join("");

    const splitSelect = document.getElementById("sourceSplitFilter");
    splitSelect.innerHTML = `<option value="All">All</option>` + (data.source_splits || [])
        .map(split => `<option value="${split}">${split}</option>`)
        .join("");
}

async function refreshDashboard() {
    const queryString = buildQueryString();

    try {
        const [health, summary, charts, risk, latest, product] = await Promise.all([
            fetchJson("/api/health"),
            fetchJson(`/api/summary?${queryString}`),
            fetchJson(`/api/charts?${queryString}`),
            fetchJson(`/api/risk?${queryString}`),
            fetchJson(`/api/latest?${queryString}`),
            fetchJson(`/api/product/${REQUIRED_PRODUCT_ID}`),
        ]);

        setText("mongodbStatus", health.connected ? "Connected" : "Disconnected");
        setText("totalPredictions", formatNumber(summary.total_predictions));
        setText("latestBatchId", summary.latest_batch_id ?? "N/A");
        setText("streamedAccuracy", summary.streamed_accuracy_display);
        setText("averageConfidence", summary.average_confidence_display);
        setText("positivePredictions", formatNumber(summary.positive_predictions));
        setText("negativePredictions", formatNumber(summary.negative_predictions));
        setText("neutralPredictions", formatNumber(summary.neutral_predictions));
        setText("confidenceCoverage", formatPercent(summary.confidence_coverage));
        setText("lowConfidenceCount", formatNumber(summary.low_confidence_count));
        setText("labeledSamples", formatNumber(summary.streamed_labeled_total));
        setText("latestProcessedAt", summary.latest_processed_at || "N/A");

        renderBars("sentimentBars", charts.sentiment_distribution || []);
        renderBars("modelTypeBars", charts.model_type_distribution || [], { color: "var(--accent)" });
        renderScoreBars(charts.score_distribution || []);
        renderScoreBySentiment(charts.score_by_sentiment || []);
        renderDateChart("dateChart", charts.predictions_by_date || [], "total");
        renderBars("sourceBars", charts.source_split_distribution || [], { color: "var(--accent)" });
        renderDateChart("batchChart", charts.batch_records || [], "records");
        renderBatchSummary(charts.batch_records || []);

        renderConfidenceBars("confidenceBars", charts.confidence_distribution || []);
        renderBars(
            "confidenceByLabelBars",
            (charts.confidence_by_label || []).map(row => ({
                label: row.label,
                count: row.average_confidence === null ? 0 : row.average_confidence * 100,
            })),
            { color: "var(--accent-2)" }
        );
        renderBars(
            "confidenceByModelBars",
            (charts.confidence_by_model || []).map(row => ({
                label: row.label,
                count: row.average_confidence === null ? 0 : row.average_confidence * 100,
            })),
            { color: "var(--purple)" }
        );
        renderConfusionMatrix(charts.confusion_matrix || []);

        renderProductAnalysis(product);
        renderLatestTable(latest);
        renderRiskTable("suspiciousTableBody", risk.suspicious_samples || [], "suspicious");
        renderRiskTable("lowConfidenceTableBody", risk.low_confidence_samples || [], "low");

        const reportParams = new URLSearchParams(getFilters());
        document.getElementById("pdfReportLink").href = `/api/report.pdf?${reportParams.toString()}`;
        setText("lastRefresh", `Last refresh: ${new Date().toLocaleString()}`);
    } catch (error) {
        console.error(error);
        setText("mongodbStatus", "Error");
        setText("lastRefresh", `Error: ${error.message}`);
    }
}

function setupAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }

    const enabled = document.getElementById("autoRefresh").checked;
    const seconds = Number(document.getElementById("refreshSeconds").value) || 5;

    if (enabled) {
        refreshTimer = setInterval(refreshDashboard, seconds * 1000);
    }
}

function setupEventListeners() {
    [
        "sentimentFilter",
        "scoreFilter",
        "batchFilter",
        "modelTypeFilter",
        "sourceSplitFilter",
        "productTextFilter",
        "productSelectFilter",
        "startDateFilter",
        "endDateFilter",
        "limitFilter",
    ].forEach(id => {
        const element = document.getElementById(id);
        element.addEventListener("change", refreshDashboard);
        element.addEventListener("input", refreshDashboard);
    });

    document.getElementById("refreshButton").addEventListener("click", refreshDashboard);

    document.getElementById("loadRequiredProductButton").addEventListener("click", () => {
        document.getElementById("productTextFilter").value = REQUIRED_PRODUCT_ID;
        refreshDashboard();
    });

    document.getElementById("autoRefresh").addEventListener("change", setupAutoRefresh);
    document.getElementById("refreshSeconds").addEventListener("change", setupAutoRefresh);
}

async function main() {
    await loadOptions();
    setupEventListeners();
    setupAutoRefresh();
    await refreshDashboard();
}

main();
