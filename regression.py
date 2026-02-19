import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dmba import regressionSummary

# Load the dataset
df = pd.read_csv("exam_cleaned.csv")

# 1. Summary Statistics (as per faculty template)
summary_df = pd.DataFrame({
    'mean' : df.select_dtypes(include='number').mean(),
    'std' : df.select_dtypes(include='number').std(),
    'min' : df.select_dtypes(include='number').min(),
    'max' : df.select_dtypes(include='number').max(),
    'median' : df.select_dtypes(include='number').median(),
    'missing' : df.isnull().sum()
})

# 2. Iterative Imputer (Handling Missing Values)
iter_imputer = IterativeImputer(random_state=42)
df_numeric = df.select_dtypes(include='number')
df_imputed = iter_imputer.fit_transform(df_numeric)
df_imputed_df = pd.DataFrame(df_imputed, columns=df_numeric.columns)

# 3. Normalization or Standardization of the Data
df_imputed_norm_df = df_imputed_df.apply(preprocessing.scale, axis=0)

# 4. Defining the Eight Predictors
predictors = ['total_spent', 'recency_days', 'satisfaction_score', 'email_open_rate', 'loyalty_points', 'tenure_months', 
              'discount_rate_mean', 'avg_delivery_days']
outcome = ['future_3m_spend']

X = df_imputed_norm_df[predictors]
Y = df_imputed_norm_df[outcome].values.ravel()

# 5. Partition Data
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. Linear Regression Model
model = LinearRegression()
model.fit(train_X, train_Y)

# 7. Print Coefficients
print("\n--- Model Coefficients (The Weights) ---")
coefficients = zip(X.columns, model.coef_)
for predictor, coef in coefficients:
    print(f"{predictor}: {round(coef, 4)}")

# 8. Training Evaluation
print("\n--- Training Data Summary ---")
regressionSummary(train_Y, model.predict(train_X))

# 9. Testing Evaluation
print("\n--- Test Data Summary ---")
model_pred = model.predict(test_X)
regressionSummary(test_Y, model_pred)

# 10. Sample Results
result = pd.DataFrame({'Predicted': model_pred, 'Actual': test_Y, 'Residual': test_Y - model_pred})
print("\n--- Sample Predictions ---")
print(result.head(20))