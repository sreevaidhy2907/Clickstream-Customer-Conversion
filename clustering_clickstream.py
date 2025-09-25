# clustering_clickstream.py
"""
Clustering pipeline for Clickstream dataset with MLflow tracking.
- Loads preprocessed train_features.csv and test_features.csv
- Runs KMeans clustering (multiple k values)
- Logs silhouette & Davies–Bouldin scores
- Saves cluster profiles, plots, and session assignments
- Saves kmeans_model.pkl into outputs
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import mlflow
import joblib

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# ------------------------------------------------
# Cluster profiling
# ------------------------------------------------
def profile_clusters(session_df, labels, outdir):
    s = session_df.copy()
    s['cluster'] = labels

    numeric_cols = ['clicks','sum_price','mean_price','max_price',
                    'pages_nunique','distinct_models','avg_price_per_click','bounce']
    profile = s.groupby('cluster')[numeric_cols].mean().round(2)
    counts = s['cluster'].value_counts().sort_index().rename('size')
    profile = profile.join(counts)

    top_cats = s.groupby('cluster')['page1_mode'].agg(lambda x: x.value_counts().head(3).to_dict())
    top_countries = s.groupby('cluster')['country_mode'].agg(lambda x: x.value_counts().head(3).to_dict())
    profile['top_categories'] = top_cats
    profile['top_countries'] = top_countries

    profile_file = os.path.join(outdir, "cluster_profiles.csv")
    profile.to_csv(profile_file)
    print("Cluster profiles saved:", profile_file)
    return profile_file

# ------------------------------------------------
# Main
# ------------------------------------------------
def main(args):
    ensure_dir(args.outdir)

    # --- Load data ---
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Feature columns
    feature_cols = ['clicks','sum_price','mean_price','max_price',
                    'pages_nunique','distinct_models','avg_price_per_click','last_page','bounce']

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # --- MLflow tracking ---
    mlflow.set_experiment("Clickstream_Clustering")
    with mlflow.start_run(run_name="kmeans_clustering"):

        mlflow.log_param("train_file", args.train)
        mlflow.log_param("test_file", args.test)
        mlflow.log_param("min_k", args.min_k)
        mlflow.log_param("max_k", args.max_k)

        sil_scores, db_scores = {}, {}

        # try multiple k values
        for k in range(args.min_k, args.max_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_train)
            sil = silhouette_score(X_train, labels)
            db = davies_bouldin_score(X_train, labels)
            sil_scores[k] = sil
            db_scores[k] = db
            mlflow.log_metric(f"silhouette_k{k}", sil)
            mlflow.log_metric(f"davies_bouldin_k{k}", db)

        # best k
        best_k = max(sil_scores, key=lambda k: sil_scores[k])
        mlflow.log_param("best_k", best_k)

        # fit best model
        best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_train)
        train_labels = best_km.labels_
        test_labels = best_km.predict(X_test)

        # save session assignments
        train_df['cluster'] = train_labels
        test_df['cluster'] = test_labels
        train_out = os.path.join(args.outdir, "session_clusters_train.csv")
        test_out = os.path.join(args.outdir, "session_clusters_test.csv")
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        mlflow.log_artifact(train_out)
        mlflow.log_artifact(test_out)

        # log metrics for best model
        best_sil = silhouette_score(X_train, train_labels)
        best_db = davies_bouldin_score(X_train, train_labels)
        mlflow.log_metric("best_silhouette", best_sil)
        mlflow.log_metric("best_davies_bouldin", best_db)

        # save model
        model_file = os.path.join(args.outdir, "kmeans_model.pkl")
        joblib.dump(best_km, model_file)
        mlflow.log_artifact(model_file)

        # visualization: cluster sizes
        cluster_sizes = pd.Series(train_labels).value_counts().sort_index()
        plt.figure(figsize=(6,4))
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette="viridis")
        plt.title("Cluster sizes (Train)")
        plt.xlabel("Cluster")
        plt.ylabel("Num sessions")
        sizes_file = os.path.join(args.outdir, "cluster_sizes.png")
        plt.savefig(sizes_file)
        mlflow.log_artifact(sizes_file)
        plt.close()

        # profiling
        profile_file = profile_clusters(train_df, train_labels, args.outdir)
        mlflow.log_artifact(profile_file)

        print(f"✅ Clustering complete. Best k={best_k}, silhouette={best_sil:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="outputs_preprocessing/train_features.csv")
    parser.add_argument("--test", type=str, default="outputs_preprocessing/test_features.csv")
    parser.add_argument("--outdir", type=str, default="outputs_clustering")
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=8)
    args = parser.parse_args()
    main(args)