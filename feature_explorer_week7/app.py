import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import umap.umap_ as umap


st.set_page_config(
    page_title="Feature Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Feature Selection & Dimensionality Reduction")
st.caption("Week 7 — Intermediate to Advanced")

st.write(
    "Explore filter, wrapper, embedded, and dimensionality-reduction "
    "methods using the Breast Cancer dataset."
)


@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


X, y = load_data()

st.sidebar.header("Dataset Information")
st.sidebar.metric("Rows", X.shape[0])
st.sidebar.metric("Features", X.shape[1])
st.sidebar.metric("Classes", y.nunique())

with st.expander("🔍 View Dataset"):
    st.dataframe(X.head(10), use_container_width=True)
    st.write("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

method = st.sidebar.selectbox(
    "Select Method",
    [
        "Correlation",
        "Mutual Information",
        "RFE",
        "LASSO",
        "PCA",
        "t-SNE",
        "UMAP"
    ]
)


if method == "Correlation":
    st.header("1️⃣ Correlation Feature Selection")

    correlation = X.corr()

    threshold = st.slider(
        "Correlation Threshold",
        0.10, 1.00, 0.80, 0.05
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(correlation, aspect="auto")
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(X.columns)))
    ax.set_yticks(range(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(X.columns, fontsize=6)
    fig.tight_layout()
    st.pyplot(fig)

    selected_features = set()

    for i in range(len(correlation.columns)):
        for j in range(i):
            if abs(correlation.iloc[i, j]) > threshold:
                selected_features.add(correlation.columns[i])

    selected_features = sorted(selected_features)

    st.subheader("Highly Correlated Features")
    if selected_features:
        st.write(selected_features)
    else:
        st.info("No features found above the selected threshold.")


elif method == "Mutual Information":
    st.header("2️⃣ Mutual Information Feature Selection")

    k = st.slider(
        "Number of Features",
        1, X.shape[1], 10
    )

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=k
    )
    selector.fit(X_train, y_train)

    scores = pd.DataFrame({
        "Feature": X.columns,
        "Mutual Information": selector.scores_
    }).sort_values(
        "Mutual Information",
        ascending=False
    )

    st.subheader("Feature Scores")
    st.dataframe(scores, use_container_width=True)

    st.subheader("Top Selected Features")
    st.write(scores.head(k)["Feature"].tolist())

    fig, ax = plt.subplots(figsize=(10, 6))
    top = scores.head(k).sort_values("Mutual Information")
    ax.barh(top["Feature"], top["Mutual Information"])
    ax.set_xlabel("Mutual Information")
    ax.set_title("Top Features")
    fig.tight_layout()
    st.pyplot(fig)


elif method == "RFE":
    st.header("3️⃣ Recursive Feature Elimination (RFE)")

    k = st.slider(
        "Number of Features to Select",
        1, X.shape[1], 10
    )

    model = LogisticRegression(max_iter=5000)

    rfe = RFE(
        estimator=model,
        n_features_to_select=k
    )
    rfe.fit(X_train_scaled, y_train)

    selected_features = X.columns[rfe.support_]

    st.subheader("Selected Features")
    st.write(selected_features.tolist())

    ranking = pd.DataFrame({
        "Feature": X.columns,
        "Ranking": rfe.ranking_,
        "Selected": rfe.support_
    }).sort_values("Ranking")

    st.subheader("RFE Ranking")
    st.dataframe(ranking, use_container_width=True)

    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)

    model.fit(X_train_rfe, y_train)
    predictions = model.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, predictions)

    st.metric("Logistic Regression Accuracy", f"{accuracy:.2%}")


elif method == "LASSO":
    st.header("4️⃣ LASSO Feature Selection")

    alpha = st.slider(
        "LASSO Alpha",
        0.001, 1.000, 0.010, 0.001
    )

    lasso = Lasso(
        alpha=alpha,
        max_iter=10000
    )
    lasso.fit(X_train_scaled, y_train)

    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lasso.coef_
    })

    coefficients["Absolute Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute Coefficient",
        ascending=False
    )

    st.subheader("LASSO Coefficients")
    st.dataframe(coefficients, use_container_width=True)

    selected = coefficients[
        coefficients["Coefficient"] != 0
    ]

    st.subheader(
        f"Selected Features ({len(selected)})"
    )
    st.write(selected["Feature"].tolist())

    fig, ax = plt.subplots(figsize=(10, 7))
    top = coefficients.head(15).sort_values("Coefficient")
    ax.barh(top["Feature"], top["Coefficient"])
    ax.set_xlabel("Coefficient")
    ax.set_title("Top LASSO Coefficients")
    fig.tight_layout()
    st.pyplot(fig)


elif method == "PCA":
    st.header("5️⃣ Principal Component Analysis (PCA)")

    n_components = st.slider(
        "Number of Components",
        2, X.shape[1], 2
    )

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train_scaled)

    explained = pca.explained_variance_ratio_

    st.subheader("Explained Variance")
    variance_df = pd.DataFrame({
        "Component": [
            f"PC{i+1}" for i in range(len(explained))
        ],
        "Explained Variance": explained,
        "Cumulative Variance": np.cumsum(explained)
    })
    st.dataframe(variance_df, use_container_width=True)

    st.metric(
        "Total Explained Variance",
        f"{explained.sum():.2%}"
    )

    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_train_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_train,
        alpha=0.7
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA 2D Visualization")
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cumulative = np.cumsum(explained)
    ax.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker="o"
    )
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    ax.grid()
    fig.tight_layout()
    st.pyplot(fig)


elif method == "t-SNE":
    st.header("6️⃣ t-SNE Dimensionality Reduction")

    perplexity = st.slider(
        "Perplexity",
        5, 50, 30
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )

    X_tsne = tsne.fit_transform(X_train_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_train,
        alpha=0.7
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE 2D Visualization")
    fig.tight_layout()
    st.pyplot(fig)


elif method == "UMAP":
    st.header("7️⃣ UMAP Dimensionality Reduction")

    neighbors = st.slider(
        "Number of Neighbors",
        5, 50, 15
    )

    min_dist = st.slider(
        "Minimum Distance",
        0.0, 1.0, 0.10, 0.05
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=42
    )

    X_umap = reducer.fit_transform(X_train_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=y_train,
        alpha=0.7
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("UMAP 2D Visualization")
    fig.tight_layout()
    st.pyplot(fig)


st.sidebar.markdown("---")
st.sidebar.info(
    "Week 7 Project\n"
    "Feature Selection & Dimensionality Reduction"
)

st.success("Feature Explorer is ready!")
