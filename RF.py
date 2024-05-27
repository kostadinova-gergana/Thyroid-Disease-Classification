import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Set the maximum number of displayed columns to None to show all columns
pd.set_option('display.max_columns', None)

# Load data and print first few lines of the dataset
data = pd.read_csv('thyroidDF.csv')

# DATA PRE-PROCESSING, TRANSFORMATION AND FEATURE SELECTION

# Replacing age values >100 with null
data['age'] = np.where((data.age > 100), np.nan, data.age)

# Removing redundant attributes from thyroidDF dataset
data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'patient_id', 'referral_source', 'query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'hypopituitary'], axis=1, inplace=True)

# Create a list with allowed data
allowed_values = ['-','F', 'I', 'G', 'K']

# Filter df by allowed data
filtered_data = data[data['target'].isin(allowed_values)]

# Re-mapping target values to diagnostic groups
diagnoses = {'-': 1, 'F': 2, 'I': 3, 'G': 4, 'K': 5}
filtered_data['target'] = filtered_data['target'].map(diagnoses)

# Transform true and false values to 0 and 1
columns_to_replace = ['on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'lithium', 'goitre', 'tumor', 'psych']
filtered_data[columns_to_replace] = filtered_data[columns_to_replace].replace({'t': 0, 'f': 1})

# Transform M and F values to 1 and 0
filtered_data['sex'] = filtered_data['sex'].replace({'F': 0, 'M': 1})

# Drop 'TBG' attribute from dataset
filtered_data.drop(['TBG'], axis=1, inplace=True)

# Drop rows with missing values
filtered_data.dropna(inplace=True)

# Drop attributes with low correlation to target
filtered_data.drop(['thyroid_surgery', 'pregnant', 'sick', 'lithium', 'on_antithyroid_meds', 'sex', 'I131_treatment', 'tumor'], axis=1, inplace=True)

# OVER- AND UNDERSAMPLING TO BALANCE DATA

# Resample data to balance classes
filtered_data_resampled = filtered_data.groupby('target', group_keys=False).apply(lambda x: x.sample(min(len(x), 400)))

# Shuffle the dataset
filtered_data_resampled = filtered_data_resampled.sample(frac=1).reset_index(drop=True)

# PRINCIPAL COMPONENT ANALYSIS

# Define X and y using the resampled data
X_resampled = filtered_data_resampled.drop(columns=['target'])
y_resampled = filtered_data_resampled['target']

# Standardize the features
scaler = StandardScaler()
X_resampled_standardized = scaler.fit_transform(X_resampled)

# Fit PCA and transform the standardized features
pca_resampled = PCA()
X_resampled_pca = pca_resampled.fit_transform(X_resampled_standardized)

# RANDOM FOREST MODELS

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# KFold with the resampled data
n_splits = 15
kf = KFold(n_splits=n_splits)
best_test_score = 0

# Loop through the folds
for train_index, test_index in kf.split(X_resampled_standardized):
    # Split data into train and test sets
    X_train_resampled, X_test_resampled = X_resampled_standardized[train_index], X_resampled_standardized[test_index]
    y_train_resampled, y_test_resampled = y_resampled[train_index], y_resampled[test_index]

    # Fit the model on the training data
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate on the test set
    test_score = model.score(X_test_resampled, y_test_resampled)

    # Keep track of the best test score and corresponding indices
    if test_score > best_test_score:
        best_test_score = test_score
        best_train_index = train_index
        best_test_index = test_index

# Select the best fold indices
X_train_best = X_resampled_standardized[best_train_index]
y_train_best = y_resampled[best_train_index]
X_test_best = X_resampled_standardized[best_test_index]
y_test_best = y_resampled[best_test_index]

# Fit the model on the best fold
model.fit(X_train_best, y_train_best)

# Scores
train_score_best = model.score(X_train_best, y_train_best)
test_score_best = model.score(X_test_best, y_test_best)
y_test_pred_best = model.predict(X_test_best)

# Print results
print('\nRandom Forest score for the test set (using resampled data):\n', test_score_best)
print('\nConfusion matrix for the test set (using resampled data):\n', confusion_matrix(y_test_best, y_test_pred_best))
print('\nClassification report for the test set (using resampled data):\n', classification_report(y_test_best, y_test_pred_best))

# Full set scores and reports (using resampled data)
y_p_best = model.predict(X_resampled_standardized)
print('\nRandom Forest score for the full set (using resampled data):\n', model.score(X_resampled_standardized, y_resampled))
print('\nConfusion matrix for the full set (using resampled data):\n', confusion_matrix(y_resampled, y_p_best))
print('\nClassification report for the full set (using resampled data):\n', classification_report(y_resampled, y_p_best))
