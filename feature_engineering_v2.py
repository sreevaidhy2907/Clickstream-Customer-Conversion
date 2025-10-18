"""
feature_engineering_v2.py
Enhanced feature engineering for Clickstream project.

Adds ratio-based features and categorical frequency encodings.
"""

import pandas as pd
import numpy as np
import argparse
import os
import mlflow
import joblib

# ------------------------------------------------
# Utility
# ------------------------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def add_derived_features(df):
    # Avoid division by zero
    df['exploration_ratio'] = np.where(df['clicks'] > 0, df['distinct_models'] / df['clicks'], 0)
    df['value_per_model'] = np.where(df['distinct_models'] > 0, df['sum_price'] / df['distinct_models'], 0)
    df['price_variability'] = df['max_price'] - df['mean_price']
    df['page_depth'] = df['last_page']
    return df

def frequency_encode(train_df, test_df, col):
    freq = train_df[col].value_counts(normalize=True)
    train_df[f"{col}_freq_enc"] = train_df[col].map(freq)
    test_df[f"{col}_freq_enc"] = test_df[col].map(freq).fillna(0)
    return train_df, test_df

# ------------------------------------------------
# Main
# ------------------------------------------------
def main(args):
    ensure_dir(args.outdir)

    mlflow.set_experiment("Clickstream_Feature_Engineering_v2")
    with mlflow.start_run(run_name="feature_engineering_v2"):

        # Load preprocessed outputs
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.test)

        # Load original unscaled session data for feature reconstruction
        orig_train = pd.read_csv(args.orig_train)
        orig_test = pd.read_csv(args.orig_test)

        # Merge to ensure consistency (match by session_id)
        train = pd.merge(train, orig_train, on="session_id", suffixes=("_scaled", ""))
        test = pd.merge(test, orig_test, on="session_id", suffixes=("_scaled", ""))

        # Add derived features
        train = add_derived_features(train)
        test = add_derived_features(test)

        # Frequency encoding for categorical columns
        for col in ['country_mode', 'page1_mode']:
            train, test = frequency_encode(train, test, col)

        # Final selected feature columns
        feature_cols = [
            'clicks','sum_price','mean_price','max_price','pages_nunique',
            'distinct_models','avg_price_per_click','bounce',
            'exploration_ratio','value_per_model','price_variability',
            'page_depth','country_mode_freq_enc','page1_mode_freq_enc'
        ]

        # Save engineered features
        train_out = os.path.join(args.outdir, "train_features_engineered.csv")
        test_out = os.path.join(args.outdir, "test_features_engineered.csv")
        train[feature_cols + ['session_id']].to_csv(train_out, index=False)
        test[feature_cols + ['session_id']].to_csv(test_out, index=False)

        mlflow.log_artifact(train_out)
        mlflow.log_artifact(test_out)
        mlflow.log_param("new_features", len(feature_cols))
        mlflow.log_text("\n".join(feature_cols), "feature_list.txt")

        print("✅ Feature engineering complete.")
        print(f"Engineered features saved to {args.outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="outputs_preprocessing_v2/train_features.csv")
    parser.add_argument("--test", type=str, default="outputs_preprocessing_v2/test_features.csv")
    parser.add_argument("--orig_train", type=str, default="outputs_preprocessing_v2/train_features.csv")  # or session data if separated
    parser.add_argument("--orig_test", type=str, default="outputs_preprocessing_v2/test_features.csv")
    parser.add_argument("--outdir", type=str, default="outputs_feature_engineering_v2")
    args = parser.parse_args()
    main(args)