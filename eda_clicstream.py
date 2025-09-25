# eda_clickstream.py
"""
Clickstream Customer Conversion - Exploratory Data Analysis (EDA)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/train_data.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ===============================
# 1. Distribution of page
# ===============================
page_counts = df['page'].value_counts().sort_index()
plt.figure(figsize=(8,4))
sns.barplot(x=page_counts.index, y=page_counts.values, palette="viridis")
plt.title("Distribution of page values")
plt.xlabel("Page ID")
plt.ylabel("Frequency")
plt.savefig("eda/eda_page_distribution.png")
plt.close()
print("\nPage distribution:\n", page_counts.head(10))

# ===============================
# 2. Last page per session
# ===============================
last_pages = df.groupby('session_id')['page'].last().value_counts().sort_index()
plt.figure(figsize=(8,4))
sns.barplot(x=last_pages.index, y=last_pages.values, palette="magma")
plt.title("Distribution of last page per session")
plt.xlabel("Last Page ID")
plt.ylabel("Number of Sessions")
plt.savefig("eda/eda_last_page_per_session.png")
plt.close()
print("\nLast page distribution:\n", last_pages.head(10))

# ===============================
# 3. Page vs Price
# ===============================
page_price = df.groupby('page')['price'].mean()
plt.figure(figsize=(8,4))
sns.barplot(x=page_price.index, y=page_price.values, palette="coolwarm")
plt.title("Average Price by Page")
plt.xlabel("Page ID")
plt.ylabel("Average Price")
plt.savefig("eda/eda_avg_price_by_page.png")
plt.close()
print("\nAverage price by page:\n", page_price.head(10))

# ===============================
# 4. Session length distribution
# ===============================
session_length = df.groupby('session_id')['order'].count()
plt.figure(figsize=(6,4))
sns.histplot(session_length, bins=50, kde=True)
plt.title("Distribution of session length (number of clicks)")
plt.xlabel("Clicks per session")
plt.ylabel("Frequency")
plt.savefig("eda/eda_session_length.png")
plt.close()
print("\nSession length stats:\n", session_length.describe())

# ===============================
# 5. Country distribution
# ===============================
country_counts = df['country'].value_counts().head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=country_counts.index, y=country_counts.values, palette="Set2")
plt.title("Top 10 Countries by Clicks")
plt.xlabel("Country")
plt.ylabel("Clicks")
plt.savefig("eda/eda_country_distribution.png")
plt.close()
print("\nTop 10 countries:\n", country_counts)

# ===============================
# 6. Correlation heatmap
# ===============================
plt.figure(figsize=(8,6))
sns.heatmap(df[['order','price','price_2','page']].corr(), annot=True, cmap="Blues")
plt.title("Correlation heatmap of numeric features")
plt.savefig("eda/eda_correlation_heatmap.png")
plt.close()

print("\nEDA complete. Plots saved as PNG files.")