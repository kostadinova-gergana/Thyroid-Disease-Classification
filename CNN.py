import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

def plot_accuracy_metrics(history):
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val_Accuracy')
    
    # Calculate and plot precision, recall, and F1 score
    y_pred_train = np.argmax(model.predict(X_train), axis=1)
    y_pred_val = np.argmax(model.predict(X_test), axis=1)
    train_precision = precision_score(np.argmax(y_train, axis=1), y_pred_train, average='macro')
    val_precision = precision_score(np.argmax(y_test, axis=1), y_pred_val, average='macro')
    train_recall = recall_score(np.argmax(y_train, axis=1), y_pred_train, average='macro')
    val_recall = recall_score(np.argmax(y_test, axis=1), y_pred_val, average='macro')
    train_f1 = f1_score(np.argmax(y_train, axis=1), y_pred_train, average='macro')
    val_f1 = f1_score(np.argmax(y_test, axis=1), y_pred_val, average='macro')
    
    plt.plot(train_precision, label='Precision')
    plt.plot(val_precision, label='Val_Precision')
    plt.plot(train_recall, label='Recall')
    plt.plot(val_recall, label='Val_Recall')
    plt.plot(train_f1, label='F1 Score')
    plt.plot(val_f1, label='Val_F1 Score')

    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_metrics(history):
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val_Loss')
    
    # Calculate and plot mean absolute error (MAE) and mean squared error (MSE)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_test, y_pred_val)
    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_test, y_pred_val)
    
    plt.plot(train_mae, label='MAE')
    plt.plot(val_mae, label='Val_MAE')
    plt.plot(train_mse, label='MSE')
    plt.plot(val_mse, label='Val_MSE')

    plt.title('Loss Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Error')
    plt.legend()
    plt.grid(True)
    plt.show()

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

# Create a list with the allowed values
allowed_values = ['-','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Filter df for the allowed values
filtered_data = data[data['target'].isin(allowed_values)]

# Print unique values in the 'target' column
unique_values = filtered_data['target'].unique()
#print("Unique values in the 'target' column:", unique_values)

# Print the length of the DataFrame
data_length_after_drop = len(filtered_data)
#print("Length of the DataFrame after dropping unneeded values:", data_length_after_drop)

# Re-mapping target values to diagnostic groups
diagnoses = {'-': '0', # healthy
             'A': '1', # hyperthyroid
             'B': '1', 
             'C': '1', 
             'D': '1',
             'E': '2', # hypothyroid
             'F': '2', 
             'G': '2', 
             'H': '2'}

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

#print("Healthy individuals ('1'):", target_counts.get(0, 0))
#print("Hyperthyroidism patients ('2'):", target_counts.get(1, 0))
#print("Hypothyroidism patients ('3'):", target_counts.get(2, 0))

###################################################################################
#OVER- AND UNDERSAMPLING TO BALANCE DATA

from sklearn.utils import resample

# Set the random seed for reproducibility
np.random.seed(42)

# Resample 500 samples from the healthy individuals class
healthy_samples = resample(filtered_data[filtered_data['target'] == 0], 
                           replace=False,  # Sampling without replacement
                           n_samples=500)  # Number of samples to select

# Concatenate the resampled healthy samples with the rest of the data
filtered_data_resampled = pd.concat([filtered_data[filtered_data['target'] != 0], healthy_samples])

# Shuffle the dataset to randomize the order
filtered_data_resampled = filtered_data_resampled.sample(frac=1).reset_index(drop=True)

# Check the class balance after resampling
print(filtered_data_resampled['target'].value_counts())

# Upsample the second class to increase its count to 400
hyperthyroidism_samples_upsampled = resample(filtered_data_resampled[filtered_data_resampled['target'] == 1], 
                                            replace=True,  # Sampling with replacement
                                            n_samples=400)  # Number of samples to select

# Concatenate the upsampled hyperthyroidism samples with the rest of the data
filtered_data_resampled_upsampled = pd.concat([filtered_data_resampled[filtered_data_resampled['target'] != 1], hyperthyroidism_samples_upsampled])

# Shuffle the dataset to randomize the order
filtered_data_resampled_upsampled = filtered_data_resampled_upsampled.sample(frac=1).reset_index(drop=True)

# Check the class balance after upsampling
print(filtered_data_resampled_upsampled['target'].value_counts())


##################################################################################
#PRINCIPAL COMPONENT ANALYSIS

# Define X and y
X = filtered_data.drop(columns=['target'])
y = filtered_data['target']

# Standardise the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Fit PCA and transform the standardized features
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
#plt.show()

# Fit PCA with 8 components
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_standardized)

# Printing variance ratio
#print("Explained variance ratio for each component:")
#print(pca.explained_variance_ratio_)

# Variance calculation
total_explained_variance = np.sum(pca.explained_variance_ratio_)
#print("\nTotal explained variance:", total_explained_variance)


# Plotting explained variance for each component
fig = plt.figure(figsize=(12, 6))
fig.add_subplot(1, 2, 1)
plt.bar(np.arange(pca.n_components_), 100 * pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance %")
#plt.show()

##################################################################################
#CNN MODEL

# Define features (X) and target variable (y)
X = filtered_data_resampled_upsampled.drop(columns=['target']).values
y = pd.get_dummies(filtered_data_resampled_upsampled['target']).values  # Convert target variable to one-hot encoded format

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for 1D CNN input
input_shape = (X_train.shape[1], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(3, activation='softmax')  # Output layer with 3 units (assuming 3 classes)
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Plot accuracy and loss metrics
plot_accuracy_metrics(history)
plot_loss_metrics(history)


# After training, to print accuracy and loss metrics
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Training Accuracy:", train_accuracy)
print("Training Loss:", train_loss)
print("Validation Accuracy:", val_accuracy)
print("Validation Loss:", val_loss)