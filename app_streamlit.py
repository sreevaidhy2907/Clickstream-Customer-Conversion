# app_streamlit.py
"""
Streamlit app for Clickstream clustering
- Upload raw clickstream CSV
- Apply preprocessing (imputer + scaler)
- Apply trained KMeans model
- Show cluster assignments and profiles
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Session aggregation (same as in preprocessing)
# ------------------------------------------------
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

# ------------------------------------------------
# Load model + preprocessing artifacts
# ------------------------------------------------
def load_artifacts(model_dir="outputs_clustering", prep_dir="outputs_preprocessing"):
    model_file = os.path.join(model_dir, "kmeans_model.pkl")
    scaler_file = os.path.join(prep_dir, "scaler.pkl")
    imputer_file = os.path.join(prep_dir, "imputer.pkl")

    if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(imputer_file)):
        st.error("Required artifacts not found. Please run preprocessing and clustering first.")
        return None, None, None

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    imputer = joblib.load(imputer_file)
    return model, scaler, imputer

# ------------------------------------------------
# Main Streamlit App
# ------------------------------------------------
def main():
    st.title("🛒 Clickstream Clustering Dashboard")

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a clickstream CSV", type=["csv"])

    model, scaler, imputer = load_artifacts()
    if not model:
        return

    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview", df.head())

        # Impute missing values using saved imputer
        for col, val in imputer["num"].items():
            if col in df.columns:
                df[col].fillna(val, inplace=True)
        for col, val in imputer["cat"].items():
            if col in df.columns:
                df[col].fillna(val, inplace=True)

        # Aggregate into sessions
        session_df = aggregate_sessions(df)

        # Feature columns
        feature_cols = ['clicks','sum_price','mean_price','max_price',
                        'pages_nunique','distinct_models','avg_price_per_click','last_page','bounce']

        # Scale
        X_scaled = scaler.transform(session_df[feature_cols])

        # Predict clusters
        clusters = model.predict(X_scaled)
        session_df['cluster'] = clusters

        st.write("### Clustered Sessions", session_df.head())

        # Cluster sizes
        cluster_sizes = session_df['cluster'].value_counts().sort_index()
        st.write("### Cluster Sizes", cluster_sizes)

        fig, ax = plt.subplots()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette="viridis", ax=ax)
        ax.set_title("Cluster Sizes")
        st.pyplot(fig)

        # Cluster profiles
        numeric_cols = ['clicks','sum_price','mean_price','max_price',
                        'pages_nunique','distinct_models','avg_price_per_click','bounce']
        profile = session_df.groupby('cluster')[numeric_cols].mean().round(2)
        profile['size'] = cluster_sizes
        st.write("### Cluster Profiles", profile)

    else:
        st.info("👆 Upload a clickstream CSV file to begin.")

if __name__ == "__main__":
    main()