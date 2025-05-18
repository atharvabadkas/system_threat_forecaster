import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

# Set the style for visualizations
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [12, 8]

# Display settings for better notebook visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Loading data...")
# Load the datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# 1. Missing Values Analysis
print("\n=== Missing Values Analysis ===")
train_missing = train_data.isnull().sum()
missing_cols = train_missing[train_missing > 0]

if len(missing_cols) > 0:
    print("\nMissing values in Train data:")
    print(pd.DataFrame({
        'Missing Values': train_missing[train_missing > 0],
        'Percentage': (train_missing[train_missing > 0] / len(train_data) * 100).round(2)
    }))
    
    # Calculate percentage of missing values
    missing_percentage = (train_missing / len(train_data) * 100).sort_values(ascending=True)
    missing_percentage = missing_percentage[missing_percentage > 0]
    
    plt.figure(figsize=(12, 6))
    ax = missing_percentage.plot(kind='barh', color='crimson')
    plt.title('Percentage of Missing Values by Feature', fontsize=16)
    plt.xlabel('Percentage of Missing Values', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add percentage labels on the bars
    for i, v in enumerate(missing_percentage):
        ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# 2. Target Distribution Analysis
print("\n=== Target Distribution Analysis ===")
target_counts = train_data['target'].value_counts()
print("\nTarget value counts:")
print(pd.DataFrame({
    'Count': target_counts,
    'Percentage': (target_counts / len(train_data) * 100).round(2)
}))

plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=train_data, palette='viridis')
plt.title('Target Class Distribution', fontsize=16)
plt.xlabel('Target Class', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add percentage labels
total = len(train_data)
for i, count in enumerate(target_counts):
    percentage = count/total * 100
    plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')

plt.show()

# 3. Feature Correlations
print("\n=== Feature Correlation Analysis ===")
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('target')

# Get correlations with target
correlations = train_data[numerical_cols.tolist() + ['target']].corr()['target'].sort_values(ascending=False)
print("\nTop 10 features correlated with target:")
print(pd.DataFrame({
    'Correlation': correlations.drop('target').abs().sort_values(ascending=False).head(10)
}))

top_features = correlations.drop('target').abs().sort_values(ascending=False).head(10).index

# Correlation heatmap
plt.figure(figsize=(12, 10))
top_corr_matrix = train_data[top_features.tolist() + ['target']].corr()
mask = np.triu(top_corr_matrix)
sns.heatmap(top_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            mask=mask, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Top Features', fontsize=16)
plt.tight_layout()
plt.show()

# 4. Distribution of Top 3 Features
print("\n=== Distribution of Top Features ===")
top_3_features = correlations.drop('target').abs().sort_values(ascending=False).head(3).index

plt.figure(figsize=(15, 5))
for i, feature in enumerate(top_3_features):
    plt.subplot(1, 3, i+1)
    
    # Check if feature has enough variance for KDE
    use_kde = train_data[feature].nunique() > 5 and not np.isclose(train_data[feature].var(), 0)
    
    try:
        sns.histplot(data=train_data, x=feature, hue='target', kde=use_kde, palette='Set2')
    except np.linalg.LinAlgError:
        sns.histplot(data=train_data, x=feature, hue='target', kde=False, palette='Set2')
        
    plt.title(f'Distribution of {feature}', fontsize=12)
plt.tight_layout()
plt.show()

# 5. Date Analysis
print("\n=== Date Analysis ===")
# Convert dates to datetime
train_data['DateAS'] = pd.to_datetime(train_data['DateAS'])
train_data['DateOS'] = pd.to_datetime(train_data['DateOS'])

# Calculate date difference
train_data['date_diff'] = (train_data['DateAS'] - train_data['DateOS']).dt.days

print("\nDate difference statistics:")
print(train_data['date_diff'].describe())

plt.figure(figsize=(12, 6))
try:
    sns.histplot(train_data['date_diff'], bins=50, kde=True)
except np.linalg.LinAlgError:
    sns.histplot(train_data['date_diff'], bins=50, kde=False)
plt.title('Distribution of Days Between DateAS and DateOS', fontsize=16)
plt.xlabel('Days Difference')
plt.show()

# 6. Outlier Analysis
print("\n=== Outlier Analysis ===")
plt.figure(figsize=(15, 5))
for i, feature in enumerate(top_3_features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='target', y=feature, data=train_data, palette='viridis')
    plt.title(f'Boxplot of {feature}', fontsize=12)
plt.tight_layout()
plt.show()

# 7. Train vs Test Distribution
print("\n=== Train vs Test Distribution Comparison ===")
plt.figure(figsize=(15, 5))
for i, feature in enumerate(top_3_features):
    plt.subplot(1, 3, i+1)
    
    # Check if the feature has enough unique values for KDE
    unique_values = pd.concat([train_data[feature], test_data[feature]]).nunique()
    use_kde = unique_values > 5 and not np.isclose(pd.concat([train_data[feature], test_data[feature]]).var(), 0)
    
    try:
        sns.histplot(data=train_data, x=feature, label='Train', kde=use_kde, alpha=0.5)
        sns.histplot(data=test_data, x=feature, label='Test', kde=use_kde, alpha=0.5)
    except np.linalg.LinAlgError:
        sns.histplot(data=train_data, x=feature, label='Train', kde=False, alpha=0.5)
        sns.histplot(data=test_data, x=feature, label='Test', kde=False, alpha=0.5)
    
    plt.title(f'Distribution of {feature}', fontsize=12)
    plt.legend()
plt.tight_layout()
plt.show()

print("\nKey Findings:")
print(f"1. Dataset Size: {train_data.shape[0]} training samples, {train_data.shape[1]} features")
print(f"2. Target Distribution: {target_counts[1]} positives ({target_counts[1]/len(train_data):.2%}) and {target_counts[0]} negatives ({target_counts[0]/len(train_data):.2%})")
print(f"3. Missing Values: {len(missing_cols)} features have missing values")
print("4. Date Range Analysis: Check the distribution of days between DateAS and DateOS")
print("5. Feature Importance: Top 3 features by correlation with target:", ", ".join(top_3_features)) 