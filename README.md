# 🧠 Clickstream Customer Conversion Project

## 📊 Overview
This project analyzes clickstream data to identify user behavior patterns and segment users into distinct clusters.  
It applies unsupervised machine learning (KMeans, GMM, DBSCAN, Agglomerative) and automates feature engineering, clustering, profiling, and visualization using MLflow and Streamlit.
---

## 🧩 Project Pipeline

### 1️⃣ Data Preprocessing (preprocess_clickstream_v2.py)
- Removes outliers and handles missing values  
- Aggregates raw click-level data into session-level summaries  
- Encodes categorical columns and scales numeric features  
- Saves:
  - train_sessions.csv, test_sessions.csv (unscaled)
  - train_features.csv, test_features.csv (scaled)
  - imputer.pkl, scaler.pkl for reproducibility

Run:
bash
python preprocess_clickstream_v2.py --train data/train_data.csv --test data/test_data.csv --outdir outputs_preprocessing_v2


---

2️⃣ Feature Engineering (feature_engineering_v2.py)

Adds derived features:

exploration_ratio, value_per_model, price_variability, page_depth

Frequency-encoded categorical features (country_mode, page1_mode)


Logs everything to MLflow.


Run:

python feature_engineering_v2.py --train outputs_preprocessing_v2/train_features.csv --test outputs_preprocessing_v2/test_features.csv --orig_train outputs_preprocessing_v2/train_sessions.csv --orig_test outputs_preprocessing_v2/test_sessions.csv --outdir outputs_feature_engineering_v2


---

3️⃣ Clustering (clustering_clickstream_v2.py)

Compares multiple algorithms:

KMeans

Gaussian Mixture Model

DBSCAN

Agglomerative Clustering


Evaluates via:

Silhouette Score

Davies–Bouldin Index

Calinski–Harabasz Score


Selects and saves the best-performing model.


Run:

python clustering_clickstream_v2.py --train outputs_feature_engineering_v2/train_features_engineered.csv --outdir outputs_clustering_v2 --min_k 2 --max_k 8

Output:

🏆 Best model: Agglomerative (k=2), Silhouette=0.785
✅ Clustering comparison complete. Results logged to MLflow.


---

4️⃣ Cluster Profiling (cluster_profiling_v3.py)

Summarizes and visualizes clusters

Creates automatic insight text reports

Saves:

cluster_profiles.csv

cluster_feature_distributions.png

cluster_summary_heatmap.png

cluster_summary_text.txt



Run:

python cluster_profiling_v3.py --clustered outputs_clustering_v2/session_clusters_train.csv --outdir outputs_profiling_v3


---

5️⃣ Interactive Dashboard (app_clickstream_dashboard_v3.py)

Built using Streamlit

Includes:

Cluster overview and distribution

Feature explorer (numeric + categorical)

Automated insights

Session-level lookup and download options



Run:

streamlit run app_clickstream_dashboard_v3.py


---

📂 Project Structure

├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│
├── outputs_preprocessing_v2/
│   ├── train_features.csv
│   ├── test_features.csv
│   ├── train_sessions.csv
│   ├── test_sessions.csv
│
├── outputs_feature_engineering_v2/
│   ├── train_features_engineered.csv
│   ├── test_features_engineered.csv
│
├── outputs_clustering_v2/
│   ├── session_clusters_train.csv
│   ├── clustering_results.csv
│
├── outputs_profiling_v3/
│   ├── cluster_profiles.csv
│   ├── cluster_summary_text.txt
│
├── scripts/
│   ├── preprocess_clickstream_v2.py
│   ├── feature_engineering_v2.py
│   ├── clustering_clickstream_v2.py
│   ├── cluster_profiling_v2.py
│   ├── app_clickstream_dashboard_v2.py
│
├── README.md
├── .gitignore
└── requirements.txt


---

🧰 Requirements

Create a virtual environment and install dependencies:

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt


---

📊 MLflow Tracking

Run the MLflow UI to view experiment logs:

mlflow ui

Visit http://localhost:5000 to see:

Parameters and metrics for all experiments

Model comparisons

Logged plots and artifacts



---

💡 Insights

Cluster 0: Explorers — long sessions, low bounce rate, broad engagement

Cluster 1: Quick Buyers — shorter sessions, higher bounce, faster conversions



---

🧾 Author

Sreelakshmi Vaidhyanathan

Machine Learning Engineer | Data Science Enthusiast

📧 sree.vaidhy.98@gmail.com

---