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

# Create a list with allowed data
allowed_values = ['-','F', 'I', 'G', 'K']

# Filter df by allowed data
filtered_data = data[data['target'].isin(allowed_values)]

# Print unique values in the 'target' column
unique_values = filtered_data['target'].unique()
#print("Unique values in the 'target' column:", unique_values)

# Print the length of the DataFrame
data_length_after_drop = len(filtered_data)
#print("Length of the DataFrame after dropping unneeded values:", data_length_after_drop)

# Re-mapping target values to diagnostic groups
diagnoses = {'-': '1', # healthy
             'F': '2', # primary hypothyroid
             'I': '3', # increased binding protein 
             'G': '4', # compensated hypothyroid
             'K': '5' # concurrent non-thyroidal illness
            }

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

# Create a new correlation matrix figure
plt.figure(figsize=(20, 20))

# Calculate correlation matrix
corr_matrix = filtered_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
print(corr_matrix)

# Drop attributes with low correlation to target
filtered_data.drop(['thyroid_surgery', 'pregnant', 'sick', 'lithium', 'on_antithyroid_meds', 'sex', 'I131_treatment', 'tumor'], axis=1, inplace=True)

# Create a new correlation matrix figure
plt.figure(figsize=(20, 20))

# Calculate correlation matrix
corr_matrix = filtered_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Check the number of entries in each class
target_column = 'target'
target_counts = filtered_data[target_column].value_counts()

print("Healthy individuals ('1'):", target_counts.get(1, 0))
print("Primary hypothyroid patients ('2'):", target_counts.get(2, 0))
print("Increased binding protein patients ('3'):", target_counts.get(3, 0))
print("Compensated hypothyroid patients ('4'):", target_counts.get(4, 0))
print("Concurrent non-thyroidal illness patients ('5'):", target_counts.get(5, 0))
###################################################################################
#OVER- AND UNDERSAMPLING TO BALANCE DATA

from sklearn.utils import resample

# Set the random seed for reproducibility
np.random.seed(42)

# Resample 500 samples from the healthy individuals class
healthy_samples = resample(filtered_data[filtered_data['target'] == 1], 
                           replace=False,  # Sampling without replacement
                           n_samples=400)  # Number of samples to select

# Concatenate the resampled healthy samples with the rest of the data
filtered_data_resampled = pd.concat([filtered_data[filtered_data['target'] != 1], healthy_samples])

# Shuffle the dataset to randomize the order
filtered_data_resampled = filtered_data_resampled.sample(frac=1).reset_index(drop=True)

# Check the class balance after resampling
print(filtered_data_resampled['target'].value_counts())


##################################################################################
#PRINCIPAL COMPONENT ANALYSIS

# Define X and y using the resampled data
X_resampled = filtered_data_resampled.drop(columns=['target'])
y_resampled = filtered_data_resampled['target']

# Standardize the features
scaler = StandardScaler()
X_resampled_standardized = scaler.fit_transform(X_resampled)

# Fit PCA and transform the standardized features
pca_resampled = PCA()
X_resampled_pca = pca_resampled.fit_transform(X_resampled_standardized)

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio_resampled = np.cumsum(pca_resampled.explained_variance_ratio_)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio_resampled) + 1), cumulative_variance_ratio_resampled, marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Number of Components (Resampled Data)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
#plt.show()

# Fit PCA with 8 components on the resampled data
pca_resampled = PCA(n_components=7)
X_resampled_pca = pca_resampled.fit_transform(X_resampled_standardized)

# Printing variance ratio
print("Explained variance ratio for each component (Resampled Data):")
print(pca_resampled.explained_variance_ratio_)

# Variance calculation
total_explained_variance_resampled = np.sum(pca_resampled.explained_variance_ratio_)
#print("\nTotal explained variance (Resampled Data):", total_explained_variance_resampled)

# Plotting explained variance for each component
fig = plt.figure(figsize=(12, 6))
fig.add_subplot(1, 2, 1)
plt.bar(np.arange(pca_resampled.n_components_), 100 * pca_resampled.explained_variance_ratio_)
plt.title('Relative information content of PCA components (Resampled Data)')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance %")
plt.show()



#####################################################################################

# KNN MODEL

# K-Fold Cross Validation
neighbors = list(range(1, 20, 2))
cv_scores = []
cv_k = []
cv_train_ind = []
cv_test_ind = []

kf = KFold(n_splits=10)

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_k.append(k)
    sm = 0
    
    for train_index, test_index in kf.split(X_resampled_pca):  # Use resampled PCA data
        knn.fit(X_resampled_pca[train_index], y_resampled[train_index])  # Use resampled data
        sc_test = knn.score(X_resampled_pca[test_index], y_resampled[test_index])
        if sm < sc_test:
            sm = sc_test
            train_maxindex = train_index
            test_maxindex = test_index

    cv_train_ind.append(train_maxindex)
    cv_test_ind.append(test_maxindex)
    scores = knn.score(X_resampled_pca, y_resampled)
    cv_scores.append(scores)

# Changing to misclassification error
MSE = [1 - x for x in cv_scores]

# Determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
optimal_MSE = cv_scores[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
print("The optimal value of MSE is %6.3f" % optimal_MSE)

# Fit KNN with optimal k on the entire dataset
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_resampled_pca[cv_train_ind[MSE.index(min(MSE))]], y_resampled[cv_train_ind[MSE.index(min(MSE))]])

# Evaluate on the test set
y_pred = knn.predict(X_resampled_pca[cv_test_ind[MSE.index(min(MSE))]])
print('\nknn.score for the test set:\n', knn.score(X_resampled_pca[cv_test_ind[MSE.index(min(MSE))]], y_resampled[cv_test_ind[MSE.index(min(MSE))]]))
print('\nConfusion matrix for the test set:\n', confusion_matrix(y_resampled[cv_test_ind[MSE.index(min(MSE))]], y_pred))
print('\nClassification report for the test set:\n', classification_report(y_resampled[cv_test_ind[MSE.index(min(MSE))]], y_pred))

# Evaluate on the full set
y_p = knn.predict(X_resampled_pca)
print('\nknn.score for the full set:\n', knn.score(X_resampled_pca, y_resampled))
print('\nConfusion matrix for the full set:\n', confusion_matrix(y_resampled, y_p))
print('\nClassification report for the full set:\n', classification_report(y_resampled, y_p))
