# LIGHTGBM MODEL

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("/kaggle/input/System-Threat-Forecaster/train.csv")
test_data = pd.read_csv("/kaggle/input/System-Threat-Forecaster/test.csv")
sample_submission = pd.read_csv('/kaggle/input/System-Threat-Forecaster/sample_submission.csv')

target = 'target'

# Identify initial numerical and categorical columns
initial_num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target, errors='ignore')
cat_cols = df.select_dtypes(include=['object']).columns

# Impute missing values for numerical features using training statistics
imputer_num = SimpleImputer(strategy='mean')
df[initial_num_cols] = imputer_num.fit_transform(df[initial_num_cols])
test_data[initial_num_cols] = imputer_num.transform(test_data[initial_num_cols])

# Impute missing values for categorical features using the most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
test_data[cat_cols] = imputer_cat.transform(test_data[cat_cols])

# Process date columns: extract multiple date features and compute date difference
for date_col in ['DateAS', 'DateOS']:
    df[date_col + '_month'] = pd.to_datetime(df[date_col]).dt.month
    df[date_col + '_day'] = pd.to_datetime(df[date_col]).dt.day
    df[date_col + '_dayofweek'] = pd.to_datetime(df[date_col]).dt.dayofweek
    df[date_col + '_quarter'] = pd.to_datetime(df[date_col]).dt.quarter
    
    test_data[date_col + '_month'] = pd.to_datetime(test_data[date_col]).dt.month
    test_data[date_col + '_day'] = pd.to_datetime(test_data[date_col]).dt.day
    test_data[date_col + '_dayofweek'] = pd.to_datetime(test_data[date_col]).dt.dayofweek
    test_data[date_col + '_quarter'] = pd.to_datetime(test_data[date_col]).dt.quarter

# Calculate the difference between the two dates
df['date_diff'] = (pd.to_datetime(df['DateAS']) - pd.to_datetime(df['DateOS'])).dt.days
test_data['date_diff'] = (pd.to_datetime(test_data['DateAS']) - pd.to_datetime(test_data['DateOS'])).dt.days

# Drop the original date columns as they're now redundant
df = df.drop(['DateAS', 'DateOS'], axis=1)
test_data = test_data.drop(['DateAS', 'DateOS'], axis=1)

# Remove constant columns (low cardinality features with only one unique value)
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_cols)
test_data = test_data.drop(columns=constant_cols, errors='ignore')

# Handle categorical features:
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Split categorical features into low and high cardinality groups (threshold: <= 10)
low_cardinality = [col for col in categorical_cols if df[col].nunique() <= 10]
high_cardinality = [col for col in categorical_cols if df[col].nunique() > 10]

# One-hot encode low cardinality features
df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)
test_data = pd.get_dummies(test_data, columns=low_cardinality, drop_first=True)

# Align train and test so that they have the same dummy columns
df, test_data = df.align(test_data, join='left', axis=1, fill_value=0)

# For high cardinality features, encode using a mapping built from the union of train and test values
for col in high_cardinality:
    if col in df.columns and col in test_data.columns:
        combined = pd.concat([df[col], test_data[col]], axis=0)
        categories = sorted(combined.unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df[col] = df[col].map(mapping)
        test_data[col] = test_data[col].map(mapping)

# Recompute numerical columns after encoding (exclude the target)
num_cols_final = df.drop(columns=[target]).select_dtypes(include=['int64', 'float64']).columns.tolist()

# Scale numerical features: fit on training data and then transform test data accordingly
scaler = MinMaxScaler()
df[num_cols_final] = scaler.fit_transform(df[num_cols_final])
test_data[num_cols_final] = scaler.transform(test_data[num_cols_final])

# If any categorical features remain (as object dtype), apply label encoding
remaining_cat = df.select_dtypes(include=['object']).columns.tolist()
if remaining_cat:
    le = LabelEncoder()
    for col in remaining_cat:
        df[col] = le.fit_transform(df[col])
    for col in remaining_cat:
        test_data[col] = le.transform(test_data[col])

# Prepare training data
X = df.drop(columns=[target])
y = df[target]

# Clean feature names: replace special characters with underscores
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
test_data.columns = test_data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data_lgb = lgb.Dataset(X_train, label=y_train)
val_data_lgb = lgb.Dataset(X_val, label=y_val)

# Set LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_threads': -1,
}

# Train the LightGBM model with early stopping using callbacks (verbose_eval removed)
model = lgb.train(
    params,
    train_data_lgb,
    num_boost_round=1000,
    valid_sets=[val_data_lgb],
    callbacks=[lgb.early_stopping(50)]
)

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)
accuracy = accuracy_score(y_val, y_val_pred_binary)
print(f"LightGBM Model Accuracy: {accuracy:.4f}")

# Ensure test_data has the same feature columns (and order) as the training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Make predictions on test data
test_preds = model.predict(test_data)
test_preds_binary = (test_preds > 0.5).astype(int)

# Create submission file
submission = sample_submission.copy()
submission[target] = test_preds_binary.astype(int).astype(str)
submission.to_csv('lightgbm_submission.csv', index=False)
print("LightGBM submission file created.")