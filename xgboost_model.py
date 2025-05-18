#XGBOOST MODEL

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("/kaggle/input/System-Threat-Forecaster/train.csv")
test_data = pd.read_csv("/kaggle/input/System-Threat-Forecaster/test.csv")
sample_submission = pd.read_csv('/kaggle/input/System-Threat-Forecaster/sample_submission.csv')

# Define target column
target = 'target'

# Identify initial numerical and categorical columns
initial_num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target, errors='ignore')
cat_cols = df.select_dtypes(include=['object']).columns

# Impute missing values
imputer_num = SimpleImputer(strategy='mean')
df[initial_num_cols] = imputer_num.fit_transform(df[initial_num_cols])
test_data[initial_num_cols] = imputer_num.transform(test_data[initial_num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
test_data[cat_cols] = imputer_cat.transform(test_data[cat_cols])

# Process date columns: convert to month
for date_col in ['DateAS', 'DateOS']:
    df[date_col] = pd.to_datetime(df[date_col]).dt.month
    test_data[date_col] = pd.to_datetime(test_data[date_col]).dt.month

# Remove constant (low variance) features
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_cols)
test_data = test_data.drop(columns=constant_cols, errors='ignore')

# Handle categorical features
# Split categorical columns into two groups based on cardinality
categorical_cols = df.select_dtypes(include=['object']).columns
low_cardinality = [col for col in categorical_cols if df[col].nunique() <= 10]
high_cardinality = [col for col in categorical_cols if df[col].nunique() > 10]

# One-hot encode low cardinality features
df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)
test_data = pd.get_dummies(test_data, columns=low_cardinality, drop_first=True)

# Align train and test sets so they have the same one-hot encoded columns
df, test_data = df.align(test_data, join='left', axis=1, fill_value=0)

# For high cardinality features, create a mapping using the union of categories from both sets
for col in high_cardinality:
    if col in df.columns and col in test_data.columns:
        combined = pd.concat([df[col], test_data[col]], axis=0)
        categories = sorted(combined.unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df[col] = df[col].map(mapping)
        test_data[col] = test_data[col].map(mapping)

# Ensure the target column is only in training data
if target in test_data.columns:
    test_data = test_data.drop(columns=[target])

# Recalculate the numerical columns after preprocessing (excluding the target)
num_cols_final = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()

# Scale numerical features: fit scaler on training data and transform both sets
scaler = MinMaxScaler()
df[num_cols_final] = scaler.fit_transform(df[num_cols_final])
test_data[num_cols_final] = scaler.transform(test_data[num_cols_final])

# If any categorical features remain (as object dtype), label encode them
remaining_cat = df.select_dtypes(include=['object']).columns.tolist()
if remaining_cat:
    le = LabelEncoder()
    for col in remaining_cat:
        df[col] = le.fit_transform(df[col])
    for col in remaining_cat:
        test_data[col] = le.transform(test_data[col])

# Split the training data into features and target
X = df.drop(columns=[target])
y = df[target]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model with optimized parameters
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = xgb.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")

# Make sure test_data has the same feature columns (and order) as X
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Predict on test_data
test_predictions = xgb.predict(test_data)

# Create submission file
submission = sample_submission.copy()
submission[target] = test_predictions.astype(int).astype(str)
submission.to_csv('xgboost_submission.csv', index=False)
print("XGBoost submission file created.")