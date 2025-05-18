import pandas as pd
df = pd.read_csv("/kaggle/input/System-Threat-Forecaster/train.csv")
test_data = pd.read_csv("/kaggle/input/System-Threat-Forecaster/test.csv")
sample_submission = pd.read_csv('/kaggle/input/System-Threat-Forecaster/sample_submission.csv')

df.head()

df.info()

from sklearn.impute import SimpleImputer

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = num_cols.drop('target', errors='ignore')

# Impute numerical columns with mean
imputer_num = SimpleImputer(strategy='mean')
df[num_cols] = imputer_num.fit_transform(df[num_cols])
test_data[num_cols] = imputer_num.transform(test_data[num_cols])

# Impute categorical columns with most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
test_data[cat_cols] = imputer_cat.transform(test_data[cat_cols])

print("Missing values handled successfully!")

df['DateAS'] = pd.to_datetime(df['DateAS']).dt.month
df['DateOS'] = pd.to_datetime(df['DateOS']).dt.month
test_data['DateAS'] = pd.to_datetime(test_data['DateAS']).dt.month
test_data['DateOS'] = pd.to_datetime(test_data['DateOS']).dt.month

cardinality = df.nunique().sort_values(ascending=False)

low_cardinality = cardinality[cardinality == 1].index.tolist()
df = df.drop(columns=low_cardinality)
test_data = test_data.drop(columns=low_cardinality)
df.shape

categorical_cols = df.select_dtypes(include=['object']).columns

selected_features = [col for col in categorical_cols if df[col].nunique() <= 10]
high_cardinality_features = [col for col in categorical_cols if df[col].nunique() > 10]
df = pd.get_dummies(df, columns=selected_features, drop_first=True)
test_data= pd.get_dummies(test_data, columns=selected_features, drop_first=True)
df

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in high_cardinality_features:
    # Apply Label Encoding
    df[col] = encoder.fit_transform(df[col])
    test_data[col] = encoder.fit_transform(test_data[col]) 

# Display the transformed dataset
df.head()

num_cols = df.select_dtypes(include=['int64','int32','float64']).columns.drop('target', errors='ignore')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
test_data[num_cols] = scaler.fit_transform(test_data[num_cols]) 


df= df.drop(columns=list(set(df.columns)-(set(test_data.columns)|{'target'})), axis=1)

from sklearn.preprocessing import LabelEncoder

# Automatically identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
test_categorical_cols = test_data.select_dtypes(include=['object']).columns

# Apply Label Encoding
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
for col in test_categorical_cols:
    test_data[col] = encoder.fit_transform(test_data[col])  

from sklearn.model_selection import train_test_split
X = df.drop(columns=['target'])   # Features
y = df['target']     # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Initialize the XGBoost classifier
xgb = XGBClassifier(n_estimators=500, random_state=42)

# Train the model
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy:.4f}")

test_data = test_data[X.columns]

test_predictions = xgb.predict(test_data)  # Generate predictions

# Create submission DataFrame (Ensure sample_submission exists)
submission = sample_submission.copy()
submission['target'] = test_predictions  

# If classification-like labels are needed (Assuming binary case)
# Ensure model output is properly rounded or thresholded
submission['target'] = submission['target'].apply(lambda x: '1' if x > 0.5 else '0')

# Save to CSV
submission.to_csv('submission.csv', index=False)

# Output file ready for submission
print("Submission file created.")