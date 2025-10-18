"""
clustering_clickstream_v2.py
Advanced clustering comparison for Clickstream project.

- Uses engineered features from feature_engineering_v2.py
- Compares KMeans, GMM, DBSCAN, Agglomerative clustering
- Evaluates via Silhouette, Davies–Bouldin, Calinski–Harabasz indices
- Logs all results to MLflow
"""

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def evaluate_clusters(X, labels):
    """Compute evaluation metrics for clustering."""
    if len(set(labels)) <= 1:
        return np.nan, np.nan, np.nan
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, db, ch

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main(args):
    ensure_dir(args.outdir)

    mlflow.set_experiment("Clickstream_Clustering_v2")
    with mlflow.start_run(run_name="multi_model_clustering"):

        # Load engineered features
        train_df = pd.read_csv(args.train)
        X = train_df.drop(columns=["session_id"])
        print("Loaded features:", X.shape)

        # Store results
        results = []

        # ----------------------------------------------------------
        # KMEANS
        # ----------------------------------------------------------
        for k in range(args.min_k, args.max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            sil, db, ch = evaluate_clusters(X, labels)
            results.append(["KMeans", k, sil, db, ch])
            mlflow.log_metric(f"kmeans_sil_k{k}", sil)
            mlflow.log_metric(f"kmeans_db_k{k}", db)
            mlflow.log_metric(f"kmeans_ch_k{k}", ch)

        # ----------------------------------------------------------
        # GMM (Gaussian Mixture Model)
        # ----------------------------------------------------------
        for k in range(args.min_k, args.max_k + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(X)
            sil, db, ch = evaluate_clusters(X, labels)
            results.append(["GMM", k, sil, db, ch])
            mlflow.log_metric(f"gmm_sil_k{k}", sil)
            mlflow.log_metric(f"gmm_db_k{k}", db)
            mlflow.log_metric(f"gmm_ch_k{k}", ch)

        # ----------------------------------------------------------
        # DBSCAN
        # ----------------------------------------------------------
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        sil, db, ch = evaluate_clusters(X, labels)
        results.append(["DBSCAN", "-", sil, db, ch])
        mlflow.log_metric("dbscan_sil", sil)
        mlflow.log_metric("dbscan_db", db)
        mlflow.log_metric("dbscan_ch", ch)

        # ----------------------------------------------------------
        # Agglomerative Clustering
        # ----------------------------------------------------------
        for k in range(args.min_k, args.max_k + 1):
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(X)
            sil, db, ch = evaluate_clusters(X, labels)
            results.append(["Agglomerative", k, sil, db, ch])
            mlflow.log_metric(f"agg_sil_k{k}", sil)
            mlflow.log_metric(f"agg_db_k{k}", db)
            mlflow.log_metric(f"agg_ch_k{k}", ch)

        # ----------------------------------------------------------
        # Summarize results
        # ----------------------------------------------------------
        results_df = pd.DataFrame(results, columns=["Model", "k", "Silhouette", "DaviesBouldin", "CalinskiHarabasz"])
        results_file = os.path.join(args.outdir, "clustering_results.csv")
        results_df.to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)

        # Select best model by Silhouette score
        best_row = results_df.loc[results_df["Silhouette"].idxmax()]
        best_model_type = best_row["Model"]
        best_k = int(best_row["k"]) if best_row["k"] != "-" else None
        mlflow.log_param("best_model", best_model_type)
        mlflow.log_param("best_k", best_k)
        print(f"🏆 Best model: {best_model_type} (k={best_k}), Silhouette={best_row['Silhouette']:.3f}")

        # Fit and save best model
        if best_model_type == "KMeans":
            best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
            labels = best_model.labels_
        elif best_model_type == "GMM":
            best_model = GaussianMixture(n_components=best_k, random_state=42).fit(X)
            labels = best_model.predict(X)
        elif best_model_type == "Agglomerative":
            best_model = AgglomerativeClustering(n_clusters=best_k)
            labels = best_model.fit_predict(X)
        else:
            best_model = dbscan
            labels = dbscan.labels_

        # Save model
        model_file = os.path.join(args.outdir, f"{best_model_type}_model.pkl")
        joblib.dump(best_model, model_file)
        mlflow.log_artifact(model_file)

        # Save cluster assignments
        train_df["cluster"] = labels
        cluster_out = os.path.join(args.outdir, "session_clusters_train.csv")
        train_df.to_csv(cluster_out, index=False)
        mlflow.log_artifact(cluster_out)

        # ----------------------------------------------------------
        # Visualization (PCA 2D projection)
        # ----------------------------------------------------------
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette="viridis", s=10)
        plt.title(f"PCA Projection ({best_model_type})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="Cluster", bbox_to_anchor=(1,1))
        plt.tight_layout()
        plot_file = os.path.join(args.outdir, "pca_clusters.png")
        plt.savefig(plot_file)
        mlflow.log_artifact(plot_file)
        plt.close()

        print("✅ Clustering comparison complete. Results logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="outputs_feature_engineering_v2/train_features_engineered.csv")
    parser.add_argument("--outdir", type=str, default="outputs_clustering_v2")
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=8)
    args = parser.parse_args()
    main(args)