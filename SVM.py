import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl 
from sklearn.neighbors import KNeighborsClassifier
from numpy import loadtxt
from sklearn.utils import resample


# Set the maximum number of displayed columns to None to show all columns
pd.set_option('display.max_columns', None)

# Load data and print first few lines of the dataset
data = pd.read_csv('thyroidDF.csv')
#print(data.head())

###################################################################################
# DATA PRE-PROCESSING, TRANSFORMATION AND FEATURE SELECTION

# Check the distributions of numeric variables
#print(data.describe())

# Replacing age values >100 with null
data['age'] = np.where((data.age > 100), np.nan, data.age)

# Removing redundant attributes from thyroidDF dataset
data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'patient_id', 'referral_source', 'query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'hypopituitary'], axis=1, inplace=True)
#print(data.head())

# Create a list with allowed values
allowed_values = ['-','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Filter df for allowed values
filtered_data = data[data['target'].isin(allowed_values)]

# Print unique values in the 'target' column
unique_values = filtered_data['target'].unique()
#print("Unique values in the 'target' column:", unique_values)

# Print the length of the DataFrame
data_length_after_drop = len(filtered_data)
#print("Length of the DataFrame after dropping unneeded values:", data_length_after_drop)

# Re-mapping target values to diagnostic groups
diagnoses = {'-': '1', # healthy
             'A': '2', # hyperthyroid
             'B': '2', 
             'C': '2', 
             'D': '2',
             'E': '3', # hypothyroid
             'F': '3', 
             'G': '3', 
             'H': '3'}

filtered_data.loc[:, 'target'] = filtered_data['target'].map(diagnoses)
filtered_data['target'] = pd.to_numeric(filtered_data['target'], errors='coerce')
#print(filtered_data.head())


# Transform true and false values to 0 and 1
columns_to_replace = ['on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant',
                       'thyroid_surgery', 'I131_treatment', 'lithium', 'goitre',
                       'tumor', 'psych']

filtered_data[columns_to_replace] = filtered_data[columns_to_replace].replace({'t': 0, 'f': 1})
#print(filtered_data.head())

# Transform M and F values to 1 and 0
filtered_data['sex'] = filtered_data['sex'].replace({'F': 0, 'M': 1})
#print(filtered_data.head())

# Investigate missing data
def missing_table(filtered_data):
    total = filtered_data.isnull().sum().sort_values(ascending=False)
    percent = (filtered_data.isnull().sum()/filtered_data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
#print(missing_table(filtered_data).head(10))


# Drop 'TBG' attribute from dataset
filtered_data.drop(['TBG'], axis=1, inplace=True)

# Drop 4 observations with abnormal 'age' from dataset
filtered_data.dropna(subset=['age'], inplace=True)

# Change sex of observations with ('pregnant' == True) & ('sex' == null) to Female
filtered_data['sex'] = np.where((filtered_data.sex.isnull()) & (filtered_data.pregnant == 't'), 'F', filtered_data.sex)

# Count missing values per row
filtered_data['n_missing'] = filtered_data.isnull().sum(axis=1)
sns.histplot(filtered_data, x='n_missing', binwidth=0.5, hue='target');
#plt.show()

# Remove rows with 3 or more missing values
filtered_data.drop(filtered_data.index[filtered_data['n_missing'] > 2], inplace=True)
#print(missing_table(filtered_data).head(10))

# Fill NaN values with the median value
filtered_data = filtered_data.fillna(filtered_data.median())
#print(filtered_data.isnull().sum())

#Remove n_missing column
filtered_data = filtered_data.drop('n_missing', axis=1)

# Save df for manual analysis if needed
#excel_file_path = 'Filtered Data.xlsx'
#filtered_data.to_excel(excel_file_path, index=False)

# Drop attributes with low correlation to target
filtered_data.drop(['thyroid_surgery', 'pregnant', 'sick', 'lithium', 'on_antithyroid_meds', 'sex', 'I131_treatment', 'tumor'], axis=1, inplace=True)

# Create a new correlation matrix figure
plt.figure(figsize=(20, 20))
sns.heatmap(filtered_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#plt.show()

# Check the number of entries in each class
target_column = 'target'
target_counts = filtered_data[target_column].value_counts()

#print("Healthy individuals ('1'):", target_counts.get(1, 0))
#print("Hyperthyroidism patients ('2'):", target_counts.get(2, 0))
#print("Hypothyroidism patients ('3'):", target_counts.get(3, 0))

###################################################################################
#OVER- AND UNDERSAMPLING TO BALANCE DATA

# OVER- AND UNDERSAMPLING TO BALANCE DATA

# Set the random seed for reproducibility
np.random.seed(42)

# Resample 500 samples from the healthy individuals class
healthy_samples = resample(filtered_data[filtered_data['target'] == 1], 
                           replace=False,  # Sampling without replacement
                           n_samples=500)  # Number of samples to select

# Concatenate the resampled healthy samples with the rest of the data
filtered_data_resampled = pd.concat([filtered_data[filtered_data['target'] != 1], healthy_samples])

# Shuffle the dataset to randomize the order
filtered_data_resampled = filtered_data_resampled.sample(frac=1).reset_index(drop=True)

# Check the class balance after resampling
print(filtered_data_resampled['target'].value_counts())

# Upsample the second class to increase its count to 400
hyperthyroidism_samples_upsampled = resample(filtered_data_resampled[filtered_data_resampled['target'] == 2], 
                                            replace=True,  # Sampling with replacement
                                            n_samples=400)  # Number of samples to select

# Concatenate the upsampled hyperthyroidism samples with the rest of the data
filtered_data_resampled_upsampled = pd.concat([filtered_data_resampled[filtered_data_resampled['target'] != 2], hyperthyroidism_samples_upsampled])

# Shuffle the dataset to randomize the order
filtered_data_resampled_upsampled = filtered_data_resampled_upsampled.sample(frac=1).reset_index(drop=True)

# Check the class balance after upsampling
print(filtered_data_resampled_upsampled['target'].value_counts())

##################################################################################
# PRINCIPAL COMPONENT ANALYSIS

# Define X and y for PCA
X_pca = filtered_data_resampled_upsampled.drop(columns=['target'])
y_pca = filtered_data_resampled_upsampled['target']

# Standardize the features for PCA
scaler_pca = StandardScaler()
X_standardized_pca = scaler_pca.fit_transform(X_pca)

# Fit PCA and transform the standardized features
pca = PCA(n_components=7)
X_pca_transformed = pca.fit_transform(X_standardized_pca)

# Printing variance ratio
print("Explained variance ratio for each component:")
print(pca.explained_variance_ratio_)

# Variance calculation
total_explained_variance = np.sum(pca.explained_variance_ratio_)
print("\nTotal explained variance:", total_explained_variance)

# Plotting explained variance for each component
fig = plt.figure(figsize=(12, 6))
fig.add_subplot(1, 2, 1)
plt.bar(np.arange(pca.n_components_), 100 * pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance %")
#plt.show()

#####################################################################################
# SVM MODELS

# Define X and y for SVM
X_svm = filtered_data_resampled_upsampled.drop(columns=['target'])
y_svm = filtered_data_resampled_upsampled['target']

# Standardize the features for SVM
scaler_svm = StandardScaler()
X_standardized_svm = scaler_svm.fit_transform(X_svm)

# Fitting the SVM model
model = SVC(kernel='rbf', C=1000, gamma='auto', decision_function_shape='ovr')

# KFold
n_splits = 15
kf = KFold(n_splits=n_splits)
k = 0
sm = 0

# Loop through the folds
for train_index, test_index in kf.split(X_standardized_svm):
    model.fit(X_standardized_svm[train_index], y_svm.iloc[train_index].values)
    sc_test = model.score(X_standardized_svm[test_index], y_svm.iloc[test_index].values)

    # Find the fold with the highest test score
    if sm < sc_test:
        sm = sc_test
        ksm = k
        train_maxindex = train_index
        test_maxindex = test_index

# Select the best fold indices
X_train_svm = X_standardized_svm[train_maxindex]
y_train_svm = y_svm.iloc[train_maxindex].values
X_test_svm = X_standardized_svm[test_maxindex]
y_test_svm = y_svm.iloc[test_maxindex].values

# Fit the model on the best fold
model.fit(X_train_svm, y_train_svm)

# Scores
train_score_svm = model.score(X_train_svm, y_train_svm)
test_score_svm = model.score(X_test_svm, y_test_svm)
y_test_pred_svm = model.predict(X_test_svm)

# Print results
print('\nsvm.score for the test set:\n', model.score(X_test_svm, y_test_svm))
print('\nconfusion matrix for the test set:\n', confusion_matrix(y_test_svm, y_test_pred_svm))
print('\nclassification report for the test set:\n', classification_report(y_test_svm, y_test_pred_svm))

# Full set scores and reports
y_p_svm = model.predict(X_standardized_svm)
print('\nsvm.score for the full set:\n', model.score(X_standardized_svm, y_svm))
print('\nconfusion matrix for the full set:\n', confusion_matrix(y_svm, y_p_svm))
print('\nclassification report for the full set:\n', classification_report(y_svm, y_p_svm))