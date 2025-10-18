"""
Enhanced preprocessing pipeline (v2) for Clickstream project.

Now also saves:
- train_sessions.csv and test_sessions.csv (aggregated unscaled data)
- train_features.csv and test_features.csv (scaled numeric features)

This ensures categorical columns like 'country_mode' and 'page1_mode'
are preserved for feature engineering in the next step.
"""

import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# ---------------------------------------------------------
# Aggregation
# ---------------------------------------------------------
def aggregate_sessions(df):
    agg = df.groupby('session_id').agg(
        clicks=('order', 'count'),
        sum_price=('price', 'sum'),
        mean_price=('price', 'mean'),
        max_price=('price', 'max'),
        pages_nunique=('page', 'nunique'),
        last_page=('page', 'last'),
        distinct_models=('page2_clothing_model', 'nunique'),
        country_mode=('country', lambda s: s.mode().iat[0] if len(s) > 0 else np.nan),
        page1_mode=('page1_main_category', lambda s: s.mode().iat[0] if len(s) > 0 else np.nan),
    ).reset_index()
    agg['avg_price_per_click'] = agg['sum_price'] / agg['clicks']
    agg['bounce'] = (agg['clicks'] == 1).astype(int)
    return agg

# ---------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------
def remove_outliers(df):
    before = len(df)
    df = df[(df['price'] > 0) & (df['price'] < 500)]  # remove invalid prices
    df = df[df['order'] <= 100]                       # remove extreme click sessions
    after = len(df)
    removed = before - after
    return df, removed

# ---------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------
def compare_distributions(train_df, test_df, numeric_cols, outdir):
    comp = []
    for col in numeric_cols:
        train_mean, test_mean = train_df[col].mean(), test_df[col].mean()
        train_std, test_std = train_df[col].std(), test_df[col].std()
        diff = abs(train_mean - test_mean)
        pct_diff = diff / (abs(train_mean) + 1e-9) * 100
        comp.append([col, train_mean, test_mean, train_std, test_std, pct_diff])
    comp_df = pd.DataFrame(comp, columns=['feature','train_mean','test_mean','train_std','test_std','%diff'])
    comp_file = os.path.join(outdir, "train_test_feature_comparison.csv")
    comp_df.to_csv(comp_file, index=False)
    return comp_file, comp_df

# ---------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------
def main(args):
    ensure_dir(args.outdir)

    mlflow.set_experiment("Clickstream_Preprocessing_v2")
    with mlflow.start_run(run_name="preprocessing_v2"):

        # Load data
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.test)
        mlflow.log_param("train_shape", train.shape)
        mlflow.log_param("test_shape", test.shape)

        # Handle outliers
        train, removed_train = remove_outliers(train)
        test, removed_test = remove_outliers(test)
        mlflow.log_metric("removed_outliers_train", removed_train)
        mlflow.log_metric("removed_outliers_test", removed_test)

        # Aggregate sessions
        train_sess = aggregate_sessions(train)
        test_sess = aggregate_sessions(test)

        # Save aggregated (unscaled) sessions
        train_sess_out = os.path.join(args.outdir, "train_sessions.csv")
        test_sess_out = os.path.join(args.outdir, "test_sessions.csv")
        train_sess.to_csv(train_sess_out, index=False)
        test_sess.to_csv(test_sess_out, index=False)
        mlflow.log_artifact(train_sess_out)
        mlflow.log_artifact(test_sess_out)

        # Imputer
        imputer = {
            "num": train_sess.median(numeric_only=True).to_dict(),
            "cat": {c: train_sess[c].mode().iat[0] for c in ['country_mode', 'page1_mode']}
        }
        joblib.dump(imputer, os.path.join(args.outdir, "imputer.pkl"))
        mlflow.log_artifact(os.path.join(args.outdir, "imputer.pkl"))

        # Scaling numeric features
        feature_cols = ['clicks','sum_price','mean_price','max_price',
                        'pages_nunique','distinct_models','avg_price_per_click','last_page','bounce']

        scaler = StandardScaler().fit(train_sess[feature_cols])
        joblib.dump(scaler, os.path.join(args.outdir, "scaler.pkl"))
        mlflow.log_artifact(os.path.join(args.outdir, "scaler.pkl"))

        train_scaled = scaler.transform(train_sess[feature_cols])
        test_scaled = scaler.transform(test_sess[feature_cols])

        train_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols)
        test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols)
        train_scaled_df['session_id'] = train_sess['session_id']
        test_scaled_df['session_id'] = test_sess['session_id']

        # Save scaled features
        train_out = os.path.join(args.outdir, "train_features.csv")
        test_out = os.path.join(args.outdir, "test_features.csv")
        train_scaled_df.to_csv(train_out, index=False)
        test_scaled_df.to_csv(test_out, index=False)
        mlflow.log_artifact(train_out)
        mlflow.log_artifact(test_out)

        # Distribution comparison
        comp_file, comp_df = compare_distributions(train_sess, test_sess, feature_cols, args.outdir)
        mlflow.log_artifact(comp_file)

        # Drift visualization
        plt.figure(figsize=(8,5))
        sns.barplot(x='feature', y='%diff', data=comp_df, palette='viridis')
        plt.xticks(rotation=45)
        plt.title("Train vs Test Feature Mean Difference (%)")
        plt.tight_layout()
        drift_plot = os.path.join(args.outdir, "train_test_drift.png")
        plt.savefig(drift_plot)
        mlflow.log_artifact(drift_plot)
        plt.close()

        print("✅ Preprocessing complete.")
        print(f"Unscaled sessions: {train_sess.shape}, Scaled features: {train_scaled_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/train_data.csv")
    parser.add_argument("--test", type=str, default="data/test_data.csv")
    parser.add_argument("--outdir", type=str, default="outputs_preprocessing_v2")
    args = parser.parse_args()
    main(args)