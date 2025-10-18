"""
app_clickstream_dashboard.py
------------------------------------------------
Streamlit dashboard for Clickstream Customer Conversion Project
with interactive filters, feature explorer, and session lookup.

New in v3:
✅ Search for a session_id
✅ View its cluster + stats
✅ Compare with cluster average
✅ Download session / cluster reports
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os

# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------
def file_download_link(filepath, label="📥 Download file"):
    """Generate a clickable download link for Streamlit."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(filepath)}">{label}</a>'
    return href

def to_csv_download_link(df, filename="data.csv", label="📥 Download data as CSV"):
    """Generate download link for a dataframe."""
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

# ---------------------------------------------------
# Main App
# ---------------------------------------------------
def main():
    st.set_page_config(page_title="Clickstream Dashboard", layout="wide")
    st.title("🧠 Clickstream Customer Conversion Dashboard")
    st.markdown("Analyze user sessions, explore clusters, and view behavioral insights interactively.")

    # Paths
    cluster_file = "outputs_clustering_v2/session_clusters_train.csv"
    profile_file = "outputs_profiling_v2/cluster_profiles.csv"
    summary_file = "outputs_profiling_v2/cluster_summary_text.txt"

    if not os.path.exists(cluster_file):
        st.error("❌ Clustered session file not found. Please run clustering first.")
        return
    if not os.path.exists(profile_file):
        st.error("❌ Cluster profile file not found. Please run cluster_profiling_v2.py first.")
        return

    df = pd.read_csv(cluster_file)
    profile = pd.read_csv(profile_file)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select view:", ["Overview", "Cluster Profiles", "Feature Explorer", "Session Lookup", "Insights"])

    # Cluster filter
    all_clusters = sorted(df["cluster"].unique())
    selected_clusters = st.sidebar.multiselect("Select clusters to display:", all_clusters, default=all_clusters)
    df_filtered = df[df["cluster"].isin(selected_clusters)]

    # ---------------------------------------------------
    # Overview
    # ---------------------------------------------------
    if page == "Overview":
        st.subheader("Dataset Overview")
        st.write(f"Total sessions: {len(df)}")
        st.write(f"Clusters displayed: {len(selected_clusters)}")
        st.dataframe(df_filtered.head())

        cluster_counts = df_filtered["cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        st.markdown("**Cluster size distribution (filtered):**")
        st.write(cluster_counts)

    # ---------------------------------------------------
    # Cluster Profiles
    # ---------------------------------------------------
    elif page == "Cluster Profiles":
        st.subheader("Cluster-Level Statistics")
        st.dataframe(profile)

        numeric_cols = [col for col in df_filtered.select_dtypes(include="number").columns if col != "cluster"]
        mean_vals = df_filtered.groupby("cluster")[numeric_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mean_vals, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.markdown("Heatmap of mean feature values across clusters.")

        st.markdown(to_csv_download_link(profile, "cluster_profiles.csv", "📥 Download cluster profiles"), unsafe_allow_html=True)

    # ---------------------------------------------------
    # Feature Explorer
    # ---------------------------------------------------
    elif page == "Feature Explorer":
        st.subheader("Feature Distribution Explorer")
        feature_type = st.radio("Select feature type:", ["Numeric", "Categorical"])

        if feature_type == "Numeric":
            numeric_cols = [col for col in df_filtered.select_dtypes(include="number").columns if col != "cluster"]
            selected_feature = st.selectbox("Select numeric feature:", numeric_cols)

            col1, col2 = st.columns([2, 2])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x="cluster", y=selected_feature, data=df_filtered, palette="viridis", ax=ax)
                plt.title(f"{selected_feature} by Cluster")
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                for c in selected_clusters:
                    subset = df_filtered[df_filtered["cluster"] == c]
                    sns.kdeplot(subset[selected_feature], fill=True, alpha=0.4, label=f"Cluster {c}", ax=ax)
                plt.title(f"Distribution of {selected_feature}")
                plt.legend()
                st.pyplot(fig)

        else:
            categorical_cols = [col for col in df_filtered.select_dtypes(exclude="number").columns if col != "cluster"]
            if not categorical_cols:
                st.warning("No categorical features found.")
            else:
                selected_feature = st.selectbox("Select categorical feature:", categorical_cols)
                freq_table = df_filtered.groupby(["cluster", selected_feature]).size().unstack(fill_value=0)

                st.write("**Category Counts by Cluster**")
                st.dataframe(freq_table)

                fig, ax = plt.subplots(figsize=(8, 5))
                freq_table.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                plt.title(f"{selected_feature} distribution across clusters")
                st.pyplot(fig)

    # ---------------------------------------------------
    # Session Lookup
    # ---------------------------------------------------
    elif page == "Session Lookup":
        st.subheader("🔍 Session-Level Lookup")

        session_ids = df["session_id"].unique().tolist()
        session_input = st.text_input("Enter a session_id:", "")

        if session_input:
            try:
                session_id = int(session_input)
                session_data = df[df["session_id"] == session_id]
                if session_data.empty:
                    st.warning("No session found with that ID.")
                else:
                    st.markdown("### Session Details")
                    st.write(session_data.T)

                    cluster_id = int(session_data["cluster"].values[0])
                    st.info(f"🧩 This session belongs to **Cluster {cluster_id}**")

                    cluster_avg = df[df["cluster"] == cluster_id].mean(numeric_only=True)
                    st.markdown("### Cluster Average Comparison")
                    st.write(cluster_avg)

                    st.markdown(to_csv_download_link(session_data, f"session_{session_id}.csv", "📥 Download this session"), unsafe_allow_html=True)
            except ValueError:
                st.error("Invalid session_id. Please enter a numeric ID.")

    # ---------------------------------------------------
    # Insights
    # ---------------------------------------------------
    elif page == "Insights":
        st.subheader("🧭 Automated Cluster Insights")

        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_text = f.read()
            st.markdown(summary_text.replace("\n", "  \n"))
            st.markdown(file_download_link(summary_file, "📥 Download full insight report"), unsafe_allow_html=True)
        else:
            st.warning("Insight summary not found. Run cluster_profiling_v3.py to generate it.")

    st.markdown("---")
    st.caption("Developed as part of the Clickstream Customer Conversion ML Project")

if __name__ == "__main__":
    main()