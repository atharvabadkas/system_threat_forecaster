#RANDOMFOREST MODEL

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("/kaggle/input/System-Threat-Forecaster/train.csv")
test_data = pd.read_csv("/kaggle/input/System-Threat-Forecaster/test.csv")
sample_submission = pd.read_csv('/kaggle/input/System-Threat-Forecaster/sample_submission.csv')

# Define target column
target = 'target'

# Identify numerical and categorical columns for imputation
initial_num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target, errors='ignore')
cat_cols = df.select_dtypes(include=['object']).columns

# Impute numerical columns with mean (fit on train and transform test)
imputer_num = SimpleImputer(strategy='mean')
df[initial_num_cols] = imputer_num.fit_transform(df[initial_num_cols])
test_data[initial_num_cols] = imputer_num.transform(test_data[initial_num_cols])

# Impute categorical columns with the most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
test_data[cat_cols] = imputer_cat.transform(test_data[cat_cols])

# Convert date columns to additional features: month, day, and dayofweek
for date_col in ['DateAS', 'DateOS']:
    # Process training data
    df[date_col + '_month'] = pd.to_datetime(df[date_col]).dt.month
    df[date_col + '_day'] = pd.to_datetime(df[date_col]).dt.day
    df[date_col + '_dayofweek'] = pd.to_datetime(df[date_col]).dt.dayofweek
    
    # Process test data
    test_data[date_col + '_month'] = pd.to_datetime(test_data[date_col]).dt.month
    test_data[date_col + '_day'] = pd.to_datetime(test_data[date_col]).dt.day
    test_data[date_col + '_dayofweek'] = pd.to_datetime(test_data[date_col]).dt.dayofweek

# Drop original date columns
df = df.drop(['DateAS', 'DateOS'], axis=1)
test_data = test_data.drop(['DateAS', 'DateOS'], axis=1)

# Remove constant features (columns with only one unique value)
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_cols)
test_data = test_data.drop(columns=constant_cols, errors='ignore')

# Handle categorical features
categorical_cols = df.select_dtypes(include=['object']).columns

# Split categorical columns into low and high cardinality groups
selected_features = [col for col in categorical_cols if df[col].nunique() <= 10]
high_cardinality_features = [col for col in categorical_cols if df[col].nunique() > 10]

# One-hot encode low cardinality features
df = pd.get_dummies(df, columns=selected_features, drop_first=True)
test_data = pd.get_dummies(test_data, columns=selected_features, drop_first=True)

# Align train and test sets to have the same dummy columns
df, test_data = df.align(test_data, join='left', axis=1, fill_value=0)

# For high cardinality features, create mapping based on the union of categories from train and test
for col in high_cardinality_features:
    # In case the column was lost during one-hot encoding, check if it exists
    if col in df.columns and col in test_data.columns:
        combined = pd.concat([df[col], test_data[col]], axis=0)
        categories = sorted(combined.unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df[col] = df[col].map(mapping)
        test_data[col] = test_data[col].map(mapping)

# Recalculate numerical columns after preprocessing (exclude target)
num_cols_final = df.drop(columns=target).select_dtypes(include=[np.number]).columns.tolist()

# Scale numerical features using MinMaxScaler fitted on train data
scaler = MinMaxScaler()
df[num_cols_final] = scaler.fit_transform(df[num_cols_final])
test_data[num_cols_final] = scaler.transform(test_data[num_cols_final])

# Split the data into features and target
if target not in df.columns:
    raise ValueError("Target column missing from training data after processing.")
X = df.drop(columns=[target])
y = df[target]

# Split train data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest with optimized parameters
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.4f}")

# Ensure test_data has the same features as the training set
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Make predictions on the test data and prepare submission
test_predictions = rf.predict(test_data)
submission = sample_submission.copy()
submission[target] = test_predictions.astype(int).astype(str)
submission.to_csv('random_forest_submission.csv', index=False)
print("Random Forest submission file created.")