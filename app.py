import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    layout="wide"
)

BASE = Path(".")

# =========================
# LOAD FILES
# =========================
@st.cache_data
def load_data():
    files = {
        "metrics": "evaluation_metrics.csv",
        "insights": "business_insights.csv",
        "processed": "data_scaled_with_kmeans.csv",
        "segments": "final_customer_segments.csv",
        "dbscan": "dbscan_labels.csv",
        "gmm_probs": "gmm_probabilities.csv"
    }

    data = {}

    for key, file in files.items():
        try:
            data[key] = pd.read_csv(BASE / file)
        except:
            data[key] = None

    return data


@st.cache_resource
def load_models():
    models = {}

    try:
        models["kmeans"] = joblib.load(BASE / "kmeans_model.pkl")
    except:
        models["kmeans"] = None

    try:
        models["gmm"] = joblib.load(BASE / "gmm_model.pkl")
    except:
        models["gmm"] = None

    try:
        models["scaler"] = joblib.load(BASE / "scaler.pkl")
    except:
        models["scaler"] = None

    try:
        models["pca"] = joblib.load(BASE / "pca_model.pkl")
    except:
        models["pca"] = None

    return models


data = load_data()
models = load_models()

# =========================
# HEADER
# =========================
st.title("Credit Card Customer Segmentation Dashboard")
st.markdown(
    """
Applied Machine Learning Project using:

- KMeans Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)
- Deep Clustering using DEC
"""
)

# =========================
# TOP METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Models Used", "4")

with col2:
    if data["processed"] is not None:
        st.metric("Customers", len(data["processed"]))
    else:
        st.metric("Customers", "N/A")

with col3:
    st.metric("Best Deep Model", "DEC")

with col4:
    if data["metrics"] is not None:
        best_score = data["metrics"]["Silhouette"].max()
        st.metric("Best Silhouette", f"{best_score:.3f}")
    else:
        st.metric("Best Silhouette", "N/A")

st.divider()

# =========================
# PROJECT OVERVIEW
# =========================
left, right = st.columns([2, 1])

with left:
    st.subheader("Project Methodology")

    methodology = pd.DataFrame({
        "Stage": [
            "Preprocessing",
            "Feature Engineering",
            "Dimensionality Reduction",
            "Traditional Clustering",
            "Probabilistic Clustering",
            "Deep Clustering"
        ],
        "Description": [
            "Missing values, scaling, log transformation",
            "Behavioral financial feature creation",
            "PCA compression for efficient clustering",
            "KMeans and DBSCAN clustering",
            "Gaussian Mixture customer segmentation",
            "Autoencoder + DEC latent clustering"
        ]
    })

    st.dataframe(methodology, use_container_width=True)

with right:
    st.subheader("Dataset Overview")

    if data["processed"] is not None:
        st.write("Dataset Shape:")
        st.write(data["processed"].shape)

        st.write("Preview:")
        st.dataframe(data["processed"].head(), use_container_width=True)
    else:
        st.warning("Dataset file not found")

st.divider()

# =========================
# MODEL COMPARISON
# =========================
st.subheader("Model Performance Comparison")

if data["metrics"] is not None:
    st.dataframe(data["metrics"], use_container_width=True)

    fig = plt.figure(figsize=(10, 5))
    sns.barplot(
        data=data["metrics"],
        x="Model",
        y="Silhouette"
    )
    plt.xticks(rotation=30)
    plt.title("Silhouette Score Comparison")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("evaluation_metrics.csv not found")

st.divider()

# =========================
# DBSCAN ANALYSIS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("DBSCAN Analysis")

    if data["dbscan"] is not None:
        total = len(data["dbscan"])
        noise = (data["dbscan"]["DBSCAN_Cluster"] == -1).sum()
        clusters = data["dbscan"]["DBSCAN_Cluster"].nunique() - 1

        st.metric("Detected Clusters", clusters)
        st.metric("Noise Points", noise)
        st.metric("Noise Percentage", f"{(noise / total) * 100:.2f}%")

        st.info(
            "DBSCAN was useful for anomaly detection, but struggled due to overlapping density structures in customer behavior."
        )
    else:
        st.warning("dbscan_labels.csv not found")

with col2:
    st.subheader("GMM Confidence")

    if data["gmm_probs"] is not None:
        fig = plt.figure(figsize=(8, 4))

        numeric_probs = data["gmm_probs"].select_dtypes(include=["number"])

        sns.histplot(
            numeric_probs.max(axis=1),
            bins=30
        )

        plt.title("GMM Assignment Confidence")
        plt.xlabel("Maximum Cluster Probability")
        plt.tight_layout()

        st.pyplot(fig)

    else:
        st.info("gmm_probabilities.csv not found")

st.divider()

# =========================
# BUSINESS INSIGHTS
# =========================
st.subheader("Business Customer Segmentation Insights")

if data["insights"] is not None:
    st.dataframe(data["insights"], use_container_width=True)
else:
    st.warning("business_insights.csv not found")

st.divider()

# =========================
# FINAL DEC SEGMENTS
# =========================
st.subheader("Final Deep Clustering Segmentation (DEC)")

if data["segments"] is not None:
    st.dataframe(data["segments"].head(20), use_container_width=True)

    if "Customer_Type" in data["segments"].columns:
        counts = data["segments"]["Customer_Type"].value_counts()

        fig = plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.title("Customer Segment Distribution")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.warning("final_customer_segments.csv not found")

st.divider()

# =========================
# FINAL RECOMMENDATION
# =========================
st.subheader("Final Recommendation")

recommendation = pd.DataFrame({
    "Category": [
        "Best Traditional Model",
        "Best Density Model",
        "Best Probabilistic Model",
        "Best Deep Learning Model",
        "Recommended Final Approach"
    ],
    "Model": [
        "KMeans",
        "DBSCAN",
        "GMM",
        "DEC",
        "DEC"
    ]
})

st.dataframe(recommendation, use_container_width=True)

st.success(
    """
Deep Clustering using DEC achieved the strongest customer separation performance.
This makes it the most effective segmentation approach for this project.
"""
)