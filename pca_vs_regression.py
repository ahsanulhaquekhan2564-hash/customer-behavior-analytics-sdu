import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data and select features
df = pd.read_csv('exam_cleaned.csv')
# Exclude non-numeric and target columns
features = df.select_dtypes(include=[np.number]).drop(columns=['customer_id', 'future_3m_spend', 'churn_90d'])
target = df['future_3m_spend']

# 2. Split and Scale (Crucial for PCA)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regression on Original Components 
reg_orig = LinearRegression()
reg_orig.fit(X_train_scaled, y_train)
pred_orig = reg_orig.predict(X_test_scaled)
r2_orig = r2_score(y_test, pred_orig)

# Regression on PCA Components (PCR)SS
# We'll use 10 components based on your previous output summary
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

reg_pca = LinearRegression()
reg_pca.fit(X_train_pca, y_train)
pred_pca = reg_pca.predict(X_test_pca)
r2_pca = r2_score(y_test, pred_pca)

# 3. Compare Results
print(f"R^2 Score (Original Features): {r2_orig:.4f}")
print(f"R^2 Score (PCA Components):    {r2_pca:.4f}")
