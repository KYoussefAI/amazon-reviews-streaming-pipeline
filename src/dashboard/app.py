from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from streamlit_autorefresh import st_autorefresh

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "amazon_reviews_db"
COLLECTION_NAME = "sentiment_predictions"
SOURCE_NAME = "spark_structured_streaming"

DEFAULT_LIMIT = 1000
LOW_CONFIDENCE_THRESHOLD = 0.60

SENTIMENT_COLORS = {
    "positive": "#22C55E",
    "negative": "#EF4444",
    "neutral": "#F59E0B",
}

BACKGROUND_COLOR = "#0B1120"
CARD_COLOR = "#111827"
BORDER_COLOR = "#263244"
TEXT_COLOR = "#E5E7EB"
MUTED_TEXT_COLOR = "#9CA3AF"


st.set_page_config(
    page_title="Amazon Reviews Sentiment Command Center",
    layout="wide"
)


st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
        }}

        .main-title {{
            font-size: 2.35rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
            color: {TEXT_COLOR};
        }}

        .subtitle {{
            font-size: 1rem;
            color: {MUTED_TEXT_COLOR};
            margin-bottom: 1.2rem;
        }}

        .section-title {{
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 0.8rem;
            color: {TEXT_COLOR};
        }}

        .info-box {{
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid {BORDER_COLOR};
            background-color: {CARD_COLOR};
            margin-bottom: 1rem;
            color: {TEXT_COLOR};
        }}

        .status-card {{
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid {BORDER_COLOR};
            background-color: {CARD_COLOR};
            margin-bottom: 0.75rem;
        }}

        .status-label {{
            font-size: 0.85rem;
            color: {MUTED_TEXT_COLOR};
            margin-bottom: 0.25rem;
        }}

        .status-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {TEXT_COLOR};
        }}

        .small-text {{
            font-size: 0.9rem;
            color: {MUTED_TEXT_COLOR};
        }}

        div[data-testid="stMetric"] {{
            background-color: {CARD_COLOR};
            border: 1px solid {BORDER_COLOR};
            padding: 1rem;
            border-radius: 0.75rem;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {MUTED_TEXT_COLOR};
        }}

        div[data-testid="stMetricValue"] {{
            color: {TEXT_COLOR};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        hr {{
            border-color: {BORDER_COLOR};
        }}
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_mongo_client():
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=3000
    )
    return client


def get_collection():
    client = get_mongo_client()
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    return collection


def check_mongodb_connection():
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True, "Connected"
    except ServerSelectionTimeoutError as error:
        return False, str(error)
    except Exception as error:
        return False, str(error)


def build_query(sentiment_filter, score_filter, batch_filter):
    query = {
        "source": SOURCE_NAME
    }

    if sentiment_filter != "All":
        query["predicted_label"] = sentiment_filter

    if score_filter != "All":
        query["score"] = int(score_filter)

    if batch_filter != "All":
        query["batch_id"] = int(batch_filter)

    return query


def load_predictions(query, limit):
    collection = get_collection()

    projection = {
        "_id": 0,
        "text_preview": 1,
        "text": 1,
        "score": 1,
        "prediction": 1,
        "predicted_label": 1,
        "probability": 1,
        "batch_id": 1,
        "processed_at": 1,
        "source": 1,
    }

    cursor = collection.find(
        query,
        projection
    ).sort(
        "processed_at",
        -1
    ).limit(limit)

    return pd.DataFrame(list(cursor))


def load_all_for_analytics():
    collection = get_collection()

    projection = {
        "_id": 0,
        "score": 1,
        "predicted_label": 1,
        "probability": 1,
        "batch_id": 1,
        "processed_at": 1,
        "source": 1,
    }

    cursor = collection.find(
        {
            "source": SOURCE_NAME
        },
        projection
    )

    return pd.DataFrame(list(cursor))


def get_available_batches():
    collection = get_collection()

    batches = collection.distinct(
        "batch_id",
        {
            "source": SOURCE_NAME
        }
    )

    batches = sorted([
        batch for batch in batches
        if batch is not None
    ])

    return batches


def extract_confidence(probability):
    if not isinstance(probability, list) or len(probability) == 0:
        return None

    return float(max(probability))


def format_probability(probability):
    if not isinstance(probability, list) or len(probability) == 0:
        return ""

    return " | ".join([
        f"{float(value):.4f}"
        for value in probability
    ])


def prepare_dataframe(df):
    if df.empty:
        return df

    df = df.copy()

    if "probability" in df.columns:
        df["confidence"] = df["probability"].apply(extract_confidence)
        df["probability_values"] = df["probability"].apply(format_probability)

    if "processed_at" in df.columns:
        df["processed_at"] = pd.to_datetime(
            df["processed_at"],
            errors="coerce"
        )

    return df


def calculate_sentiment_summary(df):
    if df.empty or "predicted_label" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])

    summary = df["predicted_label"].value_counts().reset_index()
    summary.columns = ["sentiment", "count"]

    total = summary["count"].sum()

    summary["percentage"] = (
        summary["count"] / total * 100
    ).round(2)

    return summary


def calculate_score_distribution(df):
    if df.empty or "score" not in df.columns:
        return pd.DataFrame(columns=["score", "count"])

    score_df = df["score"].value_counts().sort_index().reset_index()
    score_df.columns = ["score", "count"]

    return score_df


def calculate_confidence_by_sentiment(df):
    if (
        df.empty
        or "predicted_label" not in df.columns
        or "confidence" not in df.columns
    ):
        return pd.DataFrame(columns=["predicted_label", "average_confidence"])

    confidence_df = df.groupby("predicted_label")["confidence"] \
        .mean() \
        .reset_index()

    confidence_df["average_confidence"] = (
        confidence_df["confidence"] * 100
    ).round(2)

    confidence_df = confidence_df.drop(columns=["confidence"])

    return confidence_df


def calculate_batch_summary(df):
    if df.empty or "batch_id" not in df.columns:
        return pd.DataFrame(columns=["batch_id", "records"])

    batch_df = df.groupby("batch_id") \
        .size() \
        .reset_index(name="records") \
        .sort_values("batch_id", ascending=False)

    return batch_df


def calculate_batch_kpis(batch_df):
    if batch_df.empty:
        return {
            "total_batches": 0,
            "total_records": 0,
            "avg_records_per_batch": 0,
            "min_records_per_batch": 0,
            "max_records_per_batch": 0,
            "latest_batch_id": "N/A",
        }

    return {
        "total_batches": len(batch_df),
        "total_records": int(batch_df["records"].sum()),
        "avg_records_per_batch": float(batch_df["records"].mean()),
        "min_records_per_batch": int(batch_df["records"].min()),
        "max_records_per_batch": int(batch_df["records"].max()),
        "latest_batch_id": int(batch_df["batch_id"].max()),
    }


def calculate_confidence_distribution(df):
    if df.empty or "confidence" not in df.columns:
        return pd.DataFrame(columns=["confidence_range", "count"])

    confidence_bins = pd.cut(
        df["confidence"],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True
    )

    confidence_distribution = confidence_bins.value_counts() \
        .sort_index() \
        .reset_index()

    confidence_distribution.columns = [
        "confidence_range",
        "count"
    ]

    confidence_distribution["confidence_range"] = (
        confidence_distribution["confidence_range"].astype(str)
    )

    return confidence_distribution


def calculate_suspicious_predictions(df):
    if df.empty:
        return pd.DataFrame()

    if "score" not in df.columns or "predicted_label" not in df.columns:
        return pd.DataFrame()

    suspicious_df = df[
        (
            (df["score"] == 5) &
            (df["predicted_label"] == "negative")
        )
        |
        (
            (df["score"] == 1) &
            (df["predicted_label"] == "positive")
        )
        |
        (
            (df["score"].isin([4, 5])) &
            (df["predicted_label"] == "neutral")
        )
    ].copy()

    return suspicious_df


def build_table(dataframe, columns=None, max_rows=20):
    if dataframe.empty:
        return [["No data available"]]

    if columns is not None:
        existing_columns = [
            column for column in columns
            if column in dataframe.columns
        ]
        dataframe = dataframe[existing_columns]

    dataframe = dataframe.head(max_rows).copy()

    for column in dataframe.columns:
        dataframe[column] = dataframe[column].astype(str)

    return [list(dataframe.columns)] + dataframe.values.tolist()


def add_pdf_table(elements, dataframe, title, styles, columns=None, max_rows=20):
    elements.append(Paragraph(title, styles["SectionHeading"]))
    elements.append(Spacer(1, 0.2 * cm))

    table_data = build_table(
        dataframe=dataframe,
        columns=columns,
        max_rows=max_rows
    )

    table = Table(
        table_data,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
            ("TOPPADDING", (0, 0), (-1, 0), 7),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))


def generate_pdf_report(analytics_df, predictions_df, active_filters):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm
    )

    base_styles = getSampleStyleSheet()

    styles = {
        "Title": ParagraphStyle(
            "CustomTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=12,
        ),
        "SectionHeading": ParagraphStyle(
            "CustomSectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "CustomBody",
            parent=base_styles["Normal"],
            fontSize=9,
            leading=13,
            spaceAfter=8,
        ),
    }

    elements = []

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_predictions = len(analytics_df)
    sentiment_summary = calculate_sentiment_summary(analytics_df)
    score_distribution = calculate_score_distribution(analytics_df)
    confidence_by_sentiment = calculate_confidence_by_sentiment(analytics_df)
    batch_summary = calculate_batch_summary(analytics_df)
    batch_kpis = calculate_batch_kpis(batch_summary)

    average_confidence = None
    median_confidence = None
    min_confidence = None
    max_confidence = None
    low_confidence_count = 0

    if not analytics_df.empty and "confidence" in analytics_df.columns:
        average_confidence = analytics_df["confidence"].mean()
        median_confidence = analytics_df["confidence"].median()
        min_confidence = analytics_df["confidence"].min()
        max_confidence = analytics_df["confidence"].max()
        low_confidence_count = len(
            analytics_df[
                analytics_df["confidence"] < LOW_CONFIDENCE_THRESHOLD
            ]
        )

    elements.append(Paragraph("Amazon Reviews Sentiment Analysis Report", styles["Title"]))
    elements.append(
        Paragraph(
            f"Generated at: {generated_at}",
            styles["Subtitle"]
        )
    )

    elements.append(Paragraph("1. Pipeline Overview", styles["SectionHeading"]))
    elements.append(
        Paragraph(
            "This report summarizes the analytics generated from the real-time sentiment pipeline. "
            "The dashboard reads Spark ML predictions stored in MongoDB.",
            styles["Body"]
        )
    )
    elements.append(
        Paragraph(
            "Pipeline: Producer -> Kafka -> Spark Structured Streaming -> Spark ML Prediction -> MongoDB -> Dashboard",
            styles["Body"]
        )
    )
    elements.append(
        Paragraph(
            f"MongoDB source: {DATABASE_NAME}.{COLLECTION_NAME}",
            styles["Body"]
        )
    )
    elements.append(
        Paragraph(
            f"Source filter: {SOURCE_NAME}",
            styles["Body"]
        )
    )

    elements.append(Paragraph("2. Active Dashboard Filters", styles["SectionHeading"]))
    filters_table = Table(
        [
            ["Filter", "Value"],
            ["Predicted sentiment", active_filters["sentiment_filter"]],
            ["Amazon score", active_filters["score_filter"]],
            ["Batch ID", active_filters["batch_filter"]],
            ["Displayed latest rows", str(active_filters["limit"])],
        ],
        hAlign="LEFT"
    )
    filters_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
        ])
    )
    elements.append(filters_table)
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph("3. Executive Summary", styles["SectionHeading"]))

    executive_rows = [
        ["Metric", "Value"],
        ["Total predictions", f"{total_predictions:,}"],
        ["Average confidence", "N/A" if average_confidence is None else f"{average_confidence * 100:.2f}%"],
        ["Median confidence", "N/A" if median_confidence is None else f"{median_confidence * 100:.2f}%"],
        ["Minimum confidence", "N/A" if min_confidence is None else f"{min_confidence * 100:.2f}%"],
        ["Maximum confidence", "N/A" if max_confidence is None else f"{max_confidence * 100:.2f}%"],
        [
            f"Low-confidence predictions below {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%",
            f"{low_confidence_count:,}"
        ],
    ]

    executive_table = Table(
        executive_rows,
        hAlign="LEFT"
    )
    executive_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
        ])
    )

    elements.append(executive_table)
    elements.append(Spacer(1, 0.5 * cm))

    add_pdf_table(
        elements=elements,
        dataframe=sentiment_summary,
        title="4. Sentiment Distribution",
        styles=styles,
        max_rows=10
    )

    add_pdf_table(
        elements=elements,
        dataframe=score_distribution,
        title="5. Amazon Score Distribution",
        styles=styles,
        max_rows=10
    )

    add_pdf_table(
        elements=elements,
        dataframe=confidence_by_sentiment,
        title="6. Average Confidence by Sentiment",
        styles=styles,
        max_rows=10
    )

    if (
        not analytics_df.empty
        and "score" in analytics_df.columns
        and "predicted_label" in analytics_df.columns
    ):
        score_sentiment_df = pd.crosstab(
            analytics_df["score"],
            analytics_df["predicted_label"]
        ).reset_index()

        add_pdf_table(
            elements=elements,
            dataframe=score_sentiment_df,
            title="7. Score by Predicted Sentiment",
            styles=styles,
            max_rows=20
        )

    batch_kpi_df = pd.DataFrame([
        {
            "metric": "Total micro-batches",
            "value": f"{batch_kpis['total_batches']:,}"
        },
        {
            "metric": "Total records",
            "value": f"{batch_kpis['total_records']:,}"
        },
        {
            "metric": "Average records per micro-batch",
            "value": f"{batch_kpis['avg_records_per_batch']:.2f}"
        },
        {
            "metric": "Minimum records in one micro-batch",
            "value": f"{batch_kpis['min_records_per_batch']:,}"
        },
        {
            "metric": "Maximum records in one micro-batch",
            "value": f"{batch_kpis['max_records_per_batch']:,}"
        },
        {
            "metric": "Latest batch ID",
            "value": str(batch_kpis["latest_batch_id"])
        },
    ])

    add_pdf_table(
        elements=elements,
        dataframe=batch_kpi_df,
        title="8. Streaming Batch Summary",
        styles=styles,
        max_rows=10
    )

    add_pdf_table(
        elements=elements,
        dataframe=batch_summary,
        title="8.1 Latest Micro-Batches",
        styles=styles,
        max_rows=10
    )

    if not analytics_df.empty and "confidence" in analytics_df.columns:
        low_confidence_df = analytics_df[
            analytics_df["confidence"] < LOW_CONFIDENCE_THRESHOLD
        ].copy()

        if not low_confidence_df.empty:
            low_confidence_df["confidence"] = (
                low_confidence_df["confidence"] * 100
            ).round(2)

        add_pdf_table(
            elements=elements,
            dataframe=low_confidence_df,
            title="9. Low-Confidence Prediction Samples",
            styles=styles,
            columns=[
                "processed_at",
                "score",
                "predicted_label",
                "confidence",
                "batch_id"
            ],
            max_rows=20
        )

    latest_df = predictions_df.copy()

    if not latest_df.empty and "confidence" in latest_df.columns:
        latest_df["confidence"] = (
            latest_df["confidence"] * 100
        ).round(2)

    if not latest_df.empty and "text_preview" in latest_df.columns:
        latest_df["text_preview"] = latest_df["text_preview"].astype(str).str.slice(0, 90)

    add_pdf_table(
        elements=elements,
        dataframe=latest_df,
        title="10. Latest Prediction Samples",
        styles=styles,
        columns=[
            "processed_at",
            "text_preview",
            "score",
            "predicted_label",
            "confidence",
            "batch_id"
        ],
        max_rows=20
    )

    elements.append(PageBreak())
    elements.append(Paragraph("11. Engineering Notes", styles["SectionHeading"]))
    elements.append(
        Paragraph(
            "This report is generated from MongoDB, not directly from Kafka or Spark. "
            "That means it represents predictions that were successfully persisted by the storage layer.",
            styles["Body"]
        )
    )
    elements.append(
        Paragraph(
            "Confidence values come from the Spark ML probability vector. "
            "They represent model confidence for each individual prediction, not global model accuracy.",
            styles["Body"]
        )
    )
    elements.append(
        Paragraph(
            "Low-confidence records are useful for future model improvement because they show cases where the model is less certain.",
            styles["Body"]
        )
    )

    doc.build(elements)

    buffer.seek(0)
    return buffer


def create_plotly_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR)
        ),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR
        )
    )

    return fig


def render_header():
    st.markdown(
        '<div class="main-title">Amazon Reviews Sentiment Command Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Real-time operations and sentiment monitoring dashboard powered by Kafka, Spark Structured Streaming, Spark ML, and MongoDB.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <strong>Pipeline:</strong>
        Producer -> Kafka -> Spark Structured Streaming -> Spark ML Prediction -> MongoDB -> Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    st.sidebar.header("Dashboard Controls")

    sentiment_filter = st.sidebar.selectbox(
        "Predicted sentiment",
        ["All", "positive", "negative", "neutral"]
    )

    score_filter = st.sidebar.selectbox(
        "Amazon score",
        ["All", "1", "2", "3", "4", "5"]
    )

    batches = get_available_batches()
    batch_options = ["All"] + [str(batch) for batch in batches]

    batch_filter = st.sidebar.selectbox(
        "Batch ID",
        batch_options
    )

    limit = st.sidebar.slider(
        "Latest records to display",
        min_value=50,
        max_value=5000,
        value=DEFAULT_LIMIT,
        step=50
    )

    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh dashboard",
        value=True
    )

    refresh_interval = st.sidebar.slider(
        "Refresh interval in seconds",
        min_value=2,
        max_value=30,
        value=5,
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.write("MongoDB source filter")
    st.sidebar.code(SOURCE_NAME)

    return sentiment_filter, score_filter, batch_filter, limit, auto_refresh, refresh_interval


def render_status_row(is_connected, connection_message, analytics_df, batch_kpis):
    total_predictions = len(analytics_df)

    average_confidence = None
    if not analytics_df.empty and "confidence" in analytics_df.columns:
        average_confidence = analytics_df["confidence"].mean()

    latest_batch_id = batch_kpis["latest_batch_id"]

    status_value = connection_message if is_connected else "Disconnected"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">MongoDB Status</div>
                <div class="status-value">{status_value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.metric(
            "Total Predictions",
            f"{total_predictions:,}"
        )

    with col3:
        st.metric(
            "Latest Batch ID",
            latest_batch_id
        )

    with col4:
        if average_confidence is not None:
            st.metric(
                "Average Confidence",
                f"{average_confidence * 100:.2f}%"
            )
        else:
            st.metric(
                "Average Confidence",
                "N/A"
            )


def render_prediction_kpis(analytics_df):
    sentiment_summary = calculate_sentiment_summary(analytics_df)

    counts = {}
    if not sentiment_summary.empty:
        counts = dict(
            zip(
                sentiment_summary["sentiment"],
                sentiment_summary["count"]
            )
        )

    positive_count = counts.get("positive", 0)
    negative_count = counts.get("negative", 0)
    neutral_count = counts.get("neutral", 0)

    low_confidence_count = 0
    if not analytics_df.empty and "confidence" in analytics_df.columns:
        low_confidence_count = len(
            analytics_df[
                analytics_df["confidence"] < LOW_CONFIDENCE_THRESHOLD
            ]
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Positive Predictions", f"{positive_count:,}")
    col2.metric("Negative Predictions", f"{negative_count:,}")
    col3.metric("Neutral Predictions", f"{neutral_count:,}")
    col4.metric("Low Confidence", f"{low_confidence_count:,}")


def render_sentiment_pie(analytics_df):
    sentiment_summary = calculate_sentiment_summary(analytics_df)

    if sentiment_summary.empty:
        st.warning("No sentiment data available.")
        return

    fig = px.pie(
        sentiment_summary,
        names="sentiment",
        values="count",
        hole=0.42,
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )

    fig = create_plotly_layout(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def render_confidence_distribution(analytics_df):
    confidence_distribution = calculate_confidence_distribution(analytics_df)

    if confidence_distribution.empty:
        st.warning("No confidence data available.")
        return

    fig = px.bar(
        confidence_distribution,
        x="confidence_range",
        y="count",
        title="Confidence Distribution",
        text="count",
        color="count",
        color_continuous_scale="Blues"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>Confidence range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
    )

    fig = create_plotly_layout(fig)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def render_batch_line_chart(batch_summary):
    if batch_summary.empty:
        st.warning("No batch data available.")
        return

    chart_df = batch_summary.sort_values("batch_id")

    fig = px.line(
        chart_df,
        x="batch_id",
        y="records",
        title="Records per Micro-Batch",
        markers=True
    )

    fig.update_traces(
        line=dict(color="#06B6D4", width=2),
        marker=dict(size=5),
        hovertemplate="<b>Batch:</b> %{x}<br><b>Records:</b> %{y}<extra></extra>"
    )

    fig = create_plotly_layout(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def render_batch_kpi_panel(batch_kpis):
    st.write("Batch Summary")

    batch_kpi_df = pd.DataFrame([
        {"metric": "Total micro-batches", "value": f"{batch_kpis['total_batches']:,}"},
        {"metric": "Total records", "value": f"{batch_kpis['total_records']:,}"},
        {"metric": "Average records per batch", "value": f"{batch_kpis['avg_records_per_batch']:.2f}"},
        {"metric": "Minimum records in batch", "value": f"{batch_kpis['min_records_per_batch']:,}"},
        {"metric": "Maximum records in batch", "value": f"{batch_kpis['max_records_per_batch']:,}"},
        {"metric": "Latest batch ID", "value": str(batch_kpis["latest_batch_id"])},
    ])

    st.dataframe(
        batch_kpi_df,
        use_container_width=True,
        hide_index=True
    )


def render_score_distribution(analytics_df):
    score_distribution = calculate_score_distribution(analytics_df)

    if score_distribution.empty:
        st.warning("No score data available.")
        return

    fig = px.bar(
        score_distribution,
        x="score",
        y="count",
        title="Amazon Score Distribution",
        text="count",
        color="score",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>Score:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
    )

    fig = create_plotly_layout(fig)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def render_score_sentiment_grouped_bar(analytics_df):
    if (
        analytics_df.empty
        or "score" not in analytics_df.columns
        or "predicted_label" not in analytics_df.columns
    ):
        st.warning("No score/sentiment data available.")
        return

    grouped_df = analytics_df.groupby(
        ["score", "predicted_label"]
    ).size().reset_index(name="count")

    fig = px.bar(
        grouped_df,
        x="score",
        y="count",
        color="predicted_label",
        barmode="group",
        title="Score by Predicted Sentiment",
        color_discrete_map=SENTIMENT_COLORS
    )

    fig.update_traces(
        hovertemplate="<b>Score:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
    )

    fig = create_plotly_layout(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def render_confidence_by_sentiment(analytics_df):
    confidence_by_sentiment = calculate_confidence_by_sentiment(analytics_df)

    if confidence_by_sentiment.empty:
        st.warning("No confidence by sentiment data available.")
        return

    fig = px.bar(
        confidence_by_sentiment,
        x="average_confidence",
        y="predicted_label",
        orientation="h",
        title="Average Confidence by Sentiment",
        color="predicted_label",
        color_discrete_map=SENTIMENT_COLORS,
        text="average_confidence"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Average confidence: %{x:.2f}%<extra></extra>"
    )

    fig = create_plotly_layout(fig, height=320)
    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def render_low_confidence_table(analytics_df):
    if analytics_df.empty or "confidence" not in analytics_df.columns:
        st.warning("No confidence data available.")
        return

    low_confidence_df = analytics_df[
        analytics_df["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ].copy()

    st.write(
        f"Predictions below {LOW_CONFIDENCE_THRESHOLD * 100:.0f}% confidence"
    )

    if low_confidence_df.empty:
        st.info("No low-confidence predictions found.")
        return

    display_columns = [
        "processed_at",
        "score",
        "predicted_label",
        "confidence",
        "batch_id"
    ]

    existing_columns = [
        column for column in display_columns
        if column in low_confidence_df.columns
    ]

    table_df = low_confidence_df[existing_columns].head(50).copy()

    if "confidence" in table_df.columns:
        table_df["confidence"] = (
            table_df["confidence"] * 100
        ).round(2)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )


def render_suspicious_predictions_table(predictions_df):
    suspicious_df = calculate_suspicious_predictions(predictions_df)

    st.write("Suspicious prediction samples")

    if suspicious_df.empty:
        st.info("No suspicious prediction samples found in the selected latest records.")
        return

    display_columns = [
        "processed_at",
        "text_preview",
        "score",
        "predicted_label",
        "confidence",
        "batch_id"
    ]

    existing_columns = [
        column for column in display_columns
        if column in suspicious_df.columns
    ]

    table_df = suspicious_df[existing_columns].head(50).copy()

    if "confidence" in table_df.columns:
        table_df["confidence"] = (
            table_df["confidence"] * 100
        ).round(2)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )


def render_latest_predictions(predictions_df):
    if predictions_df.empty:
        st.info("No documents match the selected filters.")
        return

    display_columns = [
        "processed_at",
        "text_preview",
        "score",
        "predicted_label",
        "confidence",
        "probability_values",
        "batch_id"
    ]

    existing_columns = [
        column for column in display_columns
        if column in predictions_df.columns
    ]

    table_df = predictions_df[existing_columns].copy()

    if "confidence" in table_df.columns:
        table_df["confidence"] = (
            table_df["confidence"] * 100
        ).round(2)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )


def render_report_download(analytics_df, predictions_df, active_filters):
    st.write(
        "Generate a PDF report from the current MongoDB dashboard data and active filters."
    )

    if "pdf_report_buffer" not in st.session_state:
        st.session_state["pdf_report_buffer"] = None

    if "pdf_report_filename" not in st.session_state:
        st.session_state["pdf_report_filename"] = None

    prepare_report = st.button(
        "Prepare PDF report",
        use_container_width=True
    )

    if prepare_report:
        pdf_buffer = generate_pdf_report(
            analytics_df=analytics_df,
            predictions_df=predictions_df,
            active_filters=active_filters
        )

        file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"amazon_reviews_sentiment_report_{file_timestamp}.pdf"

        st.session_state["pdf_report_buffer"] = pdf_buffer
        st.session_state["pdf_report_filename"] = file_name

        st.success("PDF report is ready. Click the download button below.")

    if st.session_state["pdf_report_buffer"] is not None:
        st.download_button(
            label="Download PDF report",
            data=st.session_state["pdf_report_buffer"],
            file_name=st.session_state["pdf_report_filename"],
            mime="application/pdf",
            use_container_width=True
        )


def main():
    render_header()

    is_connected, connection_message = check_mongodb_connection()

    if not is_connected:
        st.error(
            f"MongoDB connection failed: {connection_message}"
        )
        st.stop()

    sentiment_filter, score_filter, batch_filter, limit, auto_refresh, refresh_interval = render_sidebar()

    if auto_refresh:
        st_autorefresh(
            interval=refresh_interval * 1000,
            key="dashboard_auto_refresh"
        )

    query = build_query(
        sentiment_filter=sentiment_filter,
        score_filter=score_filter,
        batch_filter=batch_filter
    )

    analytics_df = load_all_for_analytics()
    analytics_df = prepare_dataframe(analytics_df)

    predictions_df = load_predictions(
        query=query,
        limit=limit
    )
    predictions_df = prepare_dataframe(predictions_df)

    batch_summary = calculate_batch_summary(analytics_df)
    batch_kpis = calculate_batch_kpis(batch_summary)

    active_filters = {
        "sentiment_filter": sentiment_filter,
        "score_filter": score_filter,
        "batch_filter": batch_filter,
        "limit": limit,
    }

    st.markdown(
        '<div class="section-title">Pipeline Status</div>',
        unsafe_allow_html=True
    )
    render_status_row(
        is_connected=is_connected,
        connection_message=connection_message,
        analytics_df=analytics_df,
        batch_kpis=batch_kpis
    )

    st.markdown(
        '<div class="section-title">Prediction Summary</div>',
        unsafe_allow_html=True
    )
    render_prediction_kpis(analytics_df)

    st.markdown(
        '<div class="section-title">Main Analytics</div>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)

    with col1:
        render_sentiment_pie(analytics_df)

    with col2:
        render_confidence_distribution(analytics_df)

    st.markdown(
        '<div class="section-title">Streaming Operations</div>',
        unsafe_allow_html=True
    )
    col3, col4 = st.columns([2, 1])

    with col3:
        render_batch_line_chart(batch_summary)

    with col4:
        render_batch_kpi_panel(batch_kpis)

    st.markdown(
        '<div class="section-title">Score Analytics</div>',
        unsafe_allow_html=True
    )
    col5, col6 = st.columns(2)

    with col5:
        render_score_distribution(analytics_df)

    with col6:
        render_score_sentiment_grouped_bar(analytics_df)

    st.markdown(
        '<div class="section-title">Model Confidence by Class</div>',
        unsafe_allow_html=True
    )
    render_confidence_by_sentiment(analytics_df)

    st.markdown(
        '<div class="section-title">Model Risk Monitoring</div>',
        unsafe_allow_html=True
    )
    col7, col8 = st.columns(2)

    with col7:
        render_low_confidence_table(analytics_df)

    with col8:
        render_suspicious_predictions_table(predictions_df)

    st.markdown(
        '<div class="section-title">Latest Streaming Events</div>',
        unsafe_allow_html=True
    )
    render_latest_predictions(predictions_df)

    st.markdown(
        '<div class="section-title">Export Report</div>',
        unsafe_allow_html=True
    )
    render_report_download(
        analytics_df=analytics_df,
        predictions_df=predictions_df,
        active_filters=active_filters
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="small-text">
        Last dashboard refresh: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()