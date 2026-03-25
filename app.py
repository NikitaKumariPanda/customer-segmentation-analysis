import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

st.title("Customer Segmentation Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("C:\\Users\\nikit\\Downloads\\new (1).csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(df.describe())

    # Handle missing values
    df = df.dropna()

    # Date split
    if "Dt_Customer" in df.columns:
        parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
        df["day"] = parts[0].astype(int)
        df["month"] = parts[1].astype(int)
        df["year"] = parts[2].astype(int)
        df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1, inplace=True, errors="ignore")

    # Separate types
    floats, objects = [], []
    for col in df.columns:
        if df[col].dtype == object:
            objects.append(col)
        elif df[col].dtype == float:
            floats.append(col)

    st.subheader("Categorical Columns")
    st.write(objects)

    st.subheader("Numerical Columns")
    st.write(floats)

    # Countplots
    st.subheader("Categorical Distributions")
    fig, axes = plt.subplots(len(objects), 1, figsize=(10, 5*len(objects)))
    if len(objects) == 1:
        axes = [axes]
    for ax, col in zip(axes, objects):
        sb.countplot(x=df[col], ax=ax)
    st.pyplot(fig)

    # Label Encoding
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sb.heatmap(df.corr() > 0.8, annot=True, cbar=False, ax=ax)
    st.pyplot(fig)

    # Scaling
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    # TSNE
    st.subheader("t-SNE Visualization")
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(tsne_data[:, 0], tsne_data[:, 1])
    st.pyplot(fig)

    # Elbow Method
    st.subheader("Elbow Method for KMeans")
    error = []
    for n_clusters in range(1, 21):
        model = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=500, random_state=22)
        model.fit(df)
        error.append(model.inertia_)
    fig, ax = plt.subplots(figsize=(10, 5))
    sb.lineplot(x=range(1, 21), y=error, ax=ax)
    sb.scatterplot(x=range(1, 21), y=error, ax=ax)
    st.pyplot(fig)

    # Final Clustering
    st.subheader("Clustered Segments")
    model = KMeans(init='k-means++', n_clusters=5, max_iter=500, random_state=22)
    segments = model.fit_predict(df)
    df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})
    fig, ax = plt.subplots(figsize=(7, 7))
    sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne, ax=ax)
    st.pyplot(fig)