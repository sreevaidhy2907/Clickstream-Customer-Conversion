# preprocess_clickstream.py
"""
Preprocessing pipeline for Clickstream dataset with MLflow logging.
- Loads train_data.csv and test_data.csv
- Reports and imputes missing values
- Aggregates data into session-level features
- Scales numerical features using StandardScaler
- Saves train_features.csv and test_features.csv
- Saves imputer.pkl (medians + modes) and scaler.pkl
- Logs everything to MLflow
"""

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def report_missing(df, name, outdir):
    """Generate missing value report and save CSV."""
    report = df.isnull().sum()
    report = report[report > 0].sort_values(ascending=False)
    report_file = os.path.join(outdir, f"missing_report_{name}.csv")
    report.to_csv(report_file)
    return report, report_file

# ------------------------------------------------
# Session aggregation
# ------------------------------------------------
def aggregate_sessions(df):
    """Aggregate raw click data into session-level features."""
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

# ------------------------------------------------
# Main
# ------------------------------------------------
def main(args):
    ensure_dir(args.outdir)

    mlflow.set_experiment("Clickstream_Preprocessing")
    with mlflow.start_run(run_name="preprocessing"):

        # -------- Load Train/Test --------
        train_df = pd.read_csv(args.train)
        test_df = pd.read_csv(args.test)
        mlflow.log_param("train_shape_raw", train_df.shape)
        mlflow.log_param("test_shape_raw", test_df.shape)

        # -------- Missing Values --------
        train_missing, train_missing_file = report_missing(train_df, "train", args.outdir)
        test_missing, test_missing_file = report_missing(test_df, "test", args.outdir)
        mlflow.log_artifact(train_missing_file)
        mlflow.log_artifact(test_missing_file)

        # -------- Imputation --------
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        cat_cols = train_df.select_dtypes(exclude=[np.number]).columns

        imputer_dict = {"num": {}, "cat": {}}

        for col in num_cols:
            median_val = train_df[col].median()
            imputer_dict["num"][col] = median_val
            train_df[col].fillna(median_val, inplace=True)
            test_df[col].fillna(median_val, inplace=True)

        for col in cat_cols:
            mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "unknown"
            imputer_dict["cat"][col] = mode_val
            train_df[col].fillna(mode_val, inplace=True)
            test_df[col].fillna(mode_val, inplace=True)

        imputer_file = os.path.join(args.outdir, "imputer.pkl")
        joblib.dump(imputer_dict, imputer_file)
        mlflow.log_artifact(imputer_file)
        mlflow.log_param("imputation_strategy", "numeric=median, categorical=mode")

        # -------- Aggregate to sessions --------
        train_sessions = aggregate_sessions(train_df)
        test_sessions = aggregate_sessions(test_df)

        # -------- Scaling --------
        feature_cols = ['clicks','sum_price','mean_price','max_price',
                        'pages_nunique','distinct_models','avg_price_per_click','last_page','bounce']
        scaler = StandardScaler().fit(train_sessions[feature_cols])
        train_sessions_scaled = train_sessions.copy()
        test_sessions_scaled = test_sessions.copy()
        train_sessions_scaled[feature_cols] = scaler.transform(train_sessions[feature_cols])
        test_sessions_scaled[feature_cols] = scaler.transform(test_sessions[feature_cols])

        scaler_file = os.path.join(args.outdir, "scaler.pkl")
        joblib.dump(scaler, scaler_file)
        mlflow.log_artifact(scaler_file)

        # -------- Save --------
        train_outfile = os.path.join(args.outdir, "train_features.csv")
        test_outfile = os.path.join(args.outdir, "test_features.csv")
        train_sessions_scaled.to_csv(train_outfile, index=False)
        test_sessions_scaled.to_csv(test_outfile, index=False)

        mlflow.log_artifact(train_outfile)
        mlflow.log_artifact(test_outfile)
        mlflow.log_param("train_sessions", train_sessions.shape[0])
        mlflow.log_param("test_sessions", test_sessions.shape[0])
        mlflow.log_metric("train_mean_clicks", train_sessions['clicks'].mean())
        mlflow.log_metric("test_mean_clicks", test_sessions['clicks'].mean())

        print("✅ Preprocessing complete.")
        print("Train sessions:", train_sessions.shape, "→ saved to", train_outfile)
        print("Test sessions:", test_sessions.shape, "→ saved to", test_outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train_data.csv")
    parser.add_argument("--test", type=str, default="test_data.csv")
    parser.add_argument("--outdir", type=str, default="outputs_preprocessing")
    args = parser.parse_args()
    main(args)