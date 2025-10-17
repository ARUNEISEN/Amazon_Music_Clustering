# Amazon Music Clustering

## Project Overview
With millions of songs available on platforms like Amazon Music, manually categorizing tracks into genres or moods is impractical. This project leverages **unsupervised machine learning techniques to automatically group songs based on their audio characteristics such as tempo, energy, danceability, acousticness, and more. The resulting clusters can represent musical genres, moods, or listening patterns without any prior labels.


---

## Skills & Tools
- **Data Handling:** Pandas, NumPy  
- **Data Preprocessing:** Feature selection, normalization (StandardScaler/MinMaxScaler)  
- **Clustering Techniques:** K-Means, DBSCAN, Hierarchical Clustering  
- **Evaluation Metrics:** Silhouette Score, Davies-Bouldin Index
- **Dimensionality Reduction:** PCA, t-SNE  
- **Visualization:** Matplotlib, Seaborn  
- **Programming Language:** Python  
- **Libraries:** scikit-learn  

---

## Dataset
- **File Name:** `single_genre_artists.csv`  
- **Numeric Features:** popularity_songs', 'duration_ms', 'explicit', 'danceability', 'energy','key', 'loudness', 'mode', 'speechiness', 'acousticness',  'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature','followers', 'popularity_artists'  
- **Text Fields:** id_songs', 'name_song', 'id_artists', 'release_date', 'genres', 'name_artists'
- **sound_features:** 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'duration_ms'


**Dataset Description:** Audio characteristics describe a song’s rhythm, mood, instrumentation, and intensity, which are essential for clustering songs based on similarity.

---

## Project Approach

### 1. Data Exploration & Preprocessing
- Load the dataset into Pandas DataFrame  
- Check for missing values and duplicates  
- Drop unnecessary columns (`track_name`, `artist_name`, `track_id`)  
- Visualize feature distributions  
- Normalize features using `StandardScaler` or `MinMaxScaler`

### 2. Feature Selection
Features used for clustering:  
`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`  

### 3. Dimensionality Reduction (Optional)
- **PCA:** Reduce dimensions for visualization while preserving variance  
- **t-SNE:** Capture complex feature relationships  

### 4. Clustering Techniques
- **K-Means:** Determine optimal clusters via Elbow Method and Silhouette Score  
- **DBSCAN:** Identify arbitrary-shaped clusters and noise points; tune `eps` and `min_samples`  
- **Hierarchical Clustering:** Create dendrograms without specifying the number of clusters  

### 5. Cluster Evaluation & Interpretation
- **Silhouette Score:** Measures intra-cluster cohesion vs inter-cluster separation  
- **Davies-Bouldin Index:** Lower values indicate better separation  
- Analyze feature means per cluster to interpret the musical characteristics of each group  

### 6. Visualization
- 2D scatter plots using PCA/t-SNE with color-coded clusters  
- Bar charts showing average feature values per cluster  
- Heatmaps comparing features across clusters  
- Distribution plots of features like energy, danceability, and tempo  

### 7. Final Analysis & Export
- Add cluster labels to the original dataset  
- Sort or group by clusters to find top tracks per cluster  
- Export final dataset with cluster labels to CSV  
- Summarize cluster characteristics in a report  

---

## Evaluation Metrics
| Metric | Description |
|--------|-------------|
| Silhouette Score | Measures cohesion and separation of clusters |
| Davies-Bouldin Index | Lower values indicate better cluster separation |
| Cluster Visualization | PCA/t-SNE plots for interpretability |
| Cluster Size Balance | Ensures clusters are evenly distributed |
| Feature Interpretability | Highlights dominant audio features per cluster |

---

## Project Deliverables
1. **Source Code** (`.ipynb` or `.py`) for preprocessing, clustering, and visualization  
2. **CSV Output:** Final dataset with cluster labels  
3. **Final Report / Presentation:** Including problem statement, approach, visualizations, and cluster analysis  
4. Streamlit app showcasing interactive cluster exploration  

---

## Technical Tags
Python, Pandas, NumPy, scikit-learn, KMeans, DBSCAN, Hierarchical Clustering, PCA, t-SNE, EDA, Clustering, Music Analytics, Recommendation, Unsupervised Learning  


