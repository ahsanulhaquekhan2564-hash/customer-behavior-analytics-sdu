import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

# Load the exam data
df = pd.read_csv("exam_cleaned.csv")

# PCA 1: Financial & Loyalty Index (The "Combo")
# We use 4 highly related columns to create a "Customer Power" score
subset_val = df[['total_spent', 'future_3m_spend', 'recency_days', 'orders_last_3m']].dropna()

pcs = PCA(n_components=2)
# Using scale ensures units like 'orders' (small) and 'spent' (large) are comparable
pcs.fit(preprocessing.scale(subset_val))

pcsSummary_df = pd.DataFrame({
    'Eigenvalue' : pcs.explained_variance_,
    'Proportion of Variance' : pcs.explained_variance_ratio_,
    'Cumulative proportion' : np.cumsum(pcs.explained_variance_ratio_)
})

print("PCA (n=2) Summary: Comprehensive Value Index")
print(pcsSummary_df)

# PCA 2: Full Behavioral Analysis (30 Columns) 
# # We select numeric columns and then use .iloc[:, 1:] to exclude 'customer_id' (index 0)
subset_30 = df.select_dtypes(include=[np.number]).dropna(axis=0)
pca_input = subset_30.iloc[:, 1:]

# If the dataset is larger, we limit to 30 columns total as requested
if len(pca_input.columns) > 30:
    pca_input = pca_input.iloc[:, :30]

n_cols = len(pca_input.columns)
pcs_full = PCA(n_components=n_cols)

# Standardization is applied here as per the best practices we discussed
pcs_full.fit(preprocessing.scale(pca_input))

pcsFullSummary_df = pd.DataFrame({
    'Eigenvalue' : pcs_full.explained_variance_,
    'Proportion of Variance' : pcs_full.explained_variance_ratio_,
    'Cumulative proportion' : np.cumsum(pcs_full.explained_variance_ratio_)
})

pcsFullSummary_df.index = ['PC{}'.format(i) for i in range(1, len(pcsFullSummary_df.index)+1)]

# Create the Components (Loadings) table
pcsFullComponents_df = pd.DataFrame(
    pcs_full.components_.transpose(), 
    columns=pcsFullSummary_df.index, 
    index=pca_input.columns
)

print(f"\n--- PCA (n={n_cols}) Summary (Standardized) ---")
print(pcsFullSummary_df.round(4))

print(f"\n--- PCA (n={n_cols}) Components (Loadings) ---")
print(pcsFullComponents_df.round(4))