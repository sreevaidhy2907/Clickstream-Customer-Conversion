"""
cluster_profiling_v2.py
-----------------------------------------------------
Generates cluster profiles, visualizations, and an automated
text-based interpretation summary. Logs everything to MLflow.

Input:
    outputs_clustering_v2/session_clusters_train.csv
Output:
    outputs_profiling_v2/
        ├── cluster_profiles.csv
        ├── cluster_feature_distributions.png
        ├── cluster_summary_heatmap.png
        ├── cluster_summary_text.txt
"""

import pandas as pd
import numpy as np
import argparse
import os
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_text_summary(mean_vals: pd.DataFrame) -> str:
    """Automatically generate an insight summary based on cluster means."""
    text = []
    text.append("📊 **Automated Cluster Insight Summary**\n")

    n_clusters = mean_vals.shape[0]
    text.append(f"\nTotal clusters detected: {n_clusters}\n")

    # Analyze feature behavior per cluster
    for cluster in mean_vals.index:
        text.append(f"\n---\n🟩 **Cluster {cluster} Summary:**\n")
        row = mean_vals.loc[cluster]

        # Heuristics for interpretation
        if "clicks" in row.index:
            if row["clicks"] > mean_vals["clicks"].mean():
                text.append("- Users in this cluster perform **many interactions** (high clicks).")
            else:
                text.append("- Users in this cluster are **quick visitors** (low clicks).")

        if "pages_nunique" in row.index:
            if row["pages_nunique"] > mean_vals["pages_nunique"].mean():
                text.append("- They explore **more unique pages** than average.")
            else:
                text.append("- They view **fewer pages**, possibly focused shoppers.")

        if "avg_price_per_click" in row.index:
            if row["avg_price_per_click"] > mean_vals["avg_price_per_click"].mean():
                text.append("- Their **average price per interaction** is high — potential buyers.")
            else:
                text.append("- Their **price per interaction** is low — likely window shoppers.")

        if "bounce" in row.index:
            if row["bounce"] > 0.4:
                text.append("- They show **higher bounce rates**, leaving early.")
            else:
                text.append("- They show **lower bounce rates**, indicating deeper engagement.")

        text.append(f"- Average session value: {row.get('sum_price', 0):.2f}")

    return "\n".join(text)

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main(args):
    ensure_dir(args.outdir)

    mlflow.set_experiment("Clickstream_Cluster_Profiling_v2")
    with mlflow.start_run(run_name="cluster_profiling_v2"):

        # Load clustered sessions
        df = pd.read_csv(args.clustered)
        print("Loaded clustered data:", df.shape)

        if "cluster" not in df.columns:
            raise ValueError("Input file must contain a 'cluster' column.")

        # Numeric feature summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "cluster"]

        cluster_profile = df.groupby("cluster")[numeric_cols].agg(["mean", "median"])
        cluster_profile.columns = ["_".join(col).strip("_") for col in cluster_profile.columns]
        cluster_profile.reset_index(inplace=True)

        profile_out = os.path.join(args.outdir, "cluster_profiles.csv")
        cluster_profile.to_csv(profile_out, index=False)
        mlflow.log_artifact(profile_out)

        # Plot: Feature distributions
        plt.figure(figsize=(14, 8))
        melted = df.melt(id_vars="cluster", value_vars=numeric_cols, var_name="Feature", value_name="Value")
        sns.boxplot(x="Feature", y="Value", hue="cluster", data=melted, palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Feature Distributions by Cluster")
        plt.tight_layout()
        boxplot_out = os.path.join(args.outdir, "cluster_feature_distributions.png")
        plt.savefig(boxplot_out)
        mlflow.log_artifact(boxplot_out)
        plt.close()

        # Plot: Heatmap
        mean_vals = df.groupby("cluster")[numeric_cols].mean()
        plt.figure(figsize=(10, 6))
        sns.heatmap(mean_vals, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Cluster Feature Means Heatmap")
        plt.tight_layout()
        heatmap_out = os.path.join(args.outdir, "cluster_summary_heatmap.png")
        plt.savefig(heatmap_out)
        mlflow.log_artifact(heatmap_out)
        plt.close()

        # Generate text summary
        summary_text = generate_text_summary(mean_vals)
        text_out = os.path.join(args.outdir, "cluster_summary_text.txt")
        with open(text_out, "w", encoding="utf-8") as f:
            f.write(summary_text)
        mlflow.log_artifact(text_out)

        print(summary_text)
        print("\n✅ Profiling complete. Insights logged to MLflow.")
        print(f"Outputs saved in: {args.outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", type=str, default="outputs_clustering_v2/session_clusters_train.csv")
    parser.add_argument("--outdir", type=str, default="outputs_profiling_v2")
    args = parser.parse_args()
    main(args)