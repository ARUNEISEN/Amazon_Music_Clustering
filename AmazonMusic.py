# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open(r"C:\Users\arune\OneDrive\Desktop\AmazonMusicClustering\data\standardscaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

st.title("Amazon Music Clustering (K-Means)")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("Upload your Amazon Music CSV dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write("Dataset preview:", df.head())

    # --- Feature Selection ---
    numeric_features = st.multiselect(
        "Select features for clustering",
        options=df.select_dtypes(include=np.number).columns.tolist(),
        default=['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_ms']
    )

    if numeric_features:
        X = df[numeric_features].copy()
        
        # --- Data Normalization ---
        
        X_scaled = scaler.fit_transform(X)
        
        # --- Select Number of Clusters ---
        k = st.slider("Select number of clusters (k)", min_value=2, max_value=20, value=5, step=1)
        
        # --- Run K-Means ---
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = cluster_labels
        st.success(f"K-Means clustering completed with {k} clusters!")
        
        # --- Cluster Summary ---
        st.subheader("Cluster Summary")
        cluster_summary = df.groupby('Cluster')[numeric_features].mean().round(2)
        st.dataframe(cluster_summary)
        
        st.subheader("Cluster Sizes")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        # --- 2D PCA Visualization ---
        st.subheader("2D PCA Visualization of Clusters")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='PCA1', y='PCA2',
            hue='Cluster',
            palette='tab10',
            data=df,
            ax=ax,
            alpha=0.7
        )
        plt.title("K-Means Clustering Visualization (2D PCA Projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(title="Cluster")
        st.pyplot(fig)

        print(df.columns.tolist())


        # Let user pick columns
        track_col = st.selectbox("Select the track name column", options=df.columns)
        artist_col = st.selectbox("Select the artist name column", options=df.columns)
        

        top_tracks = df.groupby('Cluster').head(5)[[track_col, artist_col, 'Cluster']]
        st.dataframe(top_tracks)

        
        # --- Show Top Tracks per Cluster ---
        st.subheader("Top Tracks per Cluster")
        top_tracks = df.groupby('Cluster').head(5)[['id_songs', 'id_artists', 'Cluster']]
        st.dataframe(top_tracks)

        # --- Export Clustered Dataset ---
        st.subheader("Export Clustered Dataset")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV with Cluster Labels",
            data=csv,
            file_name='amazon_music_clusters.csv',
            mime='text/csv'
        )
