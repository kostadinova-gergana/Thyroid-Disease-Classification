import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def plot_accuracy_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_loss_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
# Set the maximum number of displayed columns to None to show all columns
pd.set_option('display.max_columns', None)

# Load data and print first few lines of the dataset
data = pd.read_csv('thyroidDF.csv')

###################################################################################
# DATA PRE-PROCESSING, TRANSFORMATION AND FEATURE SELECTION

# Replacing age values >100 with null
data['age'] = np.where((data.age > 100), np.nan, data.age)

# Removing redundant attributes from thyroidDF dataset
data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'patient_id', 'referral_source', 'query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'hypopituitary'], axis=1, inplace=True)

# Create a list with allowed data
allowed_values = ['-','F', 'I', 'G', 'K']

# Filter df by allowed data
filtered_data = data[data['target'].isin(allowed_values)]

# Print unique values in the 'target' column
unique_values = filtered_data['target'].unique()

# Print the length of the DataFrame
data_length_after_drop = len(filtered_data)

# Re-mapping target values to diagnostic groups
diagnoses = {'-': '1', # healthy
             'F': '2', # primary hypothyroid
             'I': '3', # increased binding protein 
             'G': '4', # compensated hypothyroid
             'K': '5' # concurrent non-thyroidal illness
            }

filtered_data.loc[:, 'target'] = filtered_data['target'].map(diagnoses)
filtered_data['target'] = pd.to_numeric(filtered_data['target'], errors='coerce')

# Transform true and false values to 0 and 1
columns_to_replace = ['on_thyroxine', 'on_antithyroid_meds', 'sick', 'pregnant',
                       'thyroid_surgery', 'I131_treatment', 'lithium', 'goitre',
                       'tumor', 'psych']

filtered_data[columns_to_replace] = filtered_data[columns_to_replace].replace({'t': 0, 'f': 1})

# Transform M and F values to 1 and 0
filtered_data['sex'] = filtered_data['sex'].replace({'F': 0, 'M': 1})

# Drop 'TBG' attribute from dataset
filtered_data.drop(['TBG'], axis=1, inplace=True)

# Drop 4 observations with abnormal 'age' from dataset
filtered_data.dropna(subset=['age'], inplace=True)

# Change sex of observations with ('pregnant' == True) & ('sex' == null) to Female
filtered_data['sex'] = np.where((filtered_data.sex.isnull()) & (filtered_data.pregnant == 't'), 'F', filtered_data.sex)

# Count missing values per row
filtered_data['n_missing'] = filtered_data.isnull().sum(axis=1)
sns.histplot(filtered_data, x='n_missing', binwidth=0.5, hue='target')

# Remove rows with 3 or more missing values
filtered_data.drop(filtered_data.index[filtered_data['n_missing'] > 2], inplace=True)

# Fill NaN values with the median value
filtered_data = filtered_data.fillna(filtered_data.median())

# Remove n_missing column
filtered_data = filtered_data.drop('n_missing', axis=1)

# Drop attributes with low correlation to target
filtered_data.drop(['thyroid_surgery', 'pregnant', 'sick', 'lithium', 'on_antithyroid_meds', 'sex', 'I131_treatment', 'tumor'], axis=1, inplace=True)

# OVER- AND UNDERSAMPLING TO BALANCE DATA

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

# Fit PCA with 8 components on the resampled data
pca_resampled = PCA(n_components=7)
X_resampled_pca = pca_resampled.fit_transform(X_resampled_standardized)

##################################################################################
# CNN MODEL

# Define the input features (X) and target variable (y)
X = X_resampled_pca
y = y_resampled - 1  # Adjust labels to be in range 0-4

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for 1D CNN input
input_shape = (X_train.shape[1], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# One-hot encode the target variables
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# Define CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(5, activation='softmax')  # Output layer with 5 units (for 5 classes)
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

# Predictions
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_test)

# Convert one-hot encoded labels back to categorical labels
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate precision, recall, and F1-score
train_precision = precision_score(y_train_labels, np.argmax(train_predictions, axis=1), average='macro')
train_recall = recall_score(y_train_labels, np.argmax(train_predictions, axis=1), average='macro')
train_f1_score = f1_score(y_train_labels, np.argmax(train_predictions, axis=1), average='macro')

val_precision = precision_score(y_test_labels, np.argmax(val_predictions, axis=1), average='macro')
val_recall = recall_score(y_test_labels, np.argmax(val_predictions, axis=1), average='macro')
val_f1_score = f1_score(y_test_labels, np.argmax(val_predictions, axis=1), average='macro')

# Print accuracy and loss metrics
print("Training Accuracy:", train_accuracy)
print("Training Loss:", train_loss)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F1 Score:", train_f1_score)

print("Validation Accuracy:", val_accuracy)
print("Validation Loss:", val_loss)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1_score)
