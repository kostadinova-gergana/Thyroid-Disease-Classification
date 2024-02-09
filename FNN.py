import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score

# Plotting function for accuracy-related metrics
def plot_accuracy_metrics(history):
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val_Accuracy')

    # Precision
    plt.plot(history.history['precision'], label='Precision')
    plt.plot(history.history['val_precision'], label='Val_Precision')

    # Recall
    plt.plot(history.history['recall'], label='Recall')
    plt.plot(history.history['val_recall'], label='Val_Recall')

    # F1 score
    plt.plot(history.history['f1_score'], label='F1 Score')
    plt.plot(history.history['val_f1_score'], label='Val_F1 Score')

    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function for loss-related metrics
def plot_loss_metrics(history):
    plt.figure(figsize=(12, 6))

    # Loss
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val_Loss')

    plt.title('Loss Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load data and print first few lines of the dataset
data = pd.read_csv('thyroidDF.csv')

# DATA PRE-PROCESSING, TRANSFORMATION AND FEATURE SELECTION

# Replacing age values >100 with null
data['age'] = np.where((data.age > 100), np.nan, data.age)

# Removing redundant attributes from thyroidDF dataset
data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'patient_id', 'referral_source', 'query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'hypopituitary'], axis=1, inplace=True)

# Create a list with the allowed values
allowed_values = ['-','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Filter df for the allowed values
filtered_data = data[data['target'].isin(allowed_values)]

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

# Remove rows with 3 or more missing values
filtered_data.drop(filtered_data.index[filtered_data['n_missing'] > 2], inplace=True)

# Fill NaN values with the median value
filtered_data = filtered_data.fillna(filtered_data.median())

# Remove n_missing column
filtered_data = filtered_data.drop('n_missing', axis=1)

# Drop attributes with low correlation to target
filtered_data.drop(['thyroid_surgery', 'pregnant', 'sick', 'lithium', 'on_antithyroid_meds', 'sex', 'I131_treatment', 'tumor'], axis=1, inplace=True)

# OVER- AND UNDERSAMPLING TO BALANCE DATA

# Resample 500 samples from the healthy individuals class
healthy_samples = resample(filtered_data[filtered_data['target'] == 0], 
                           replace=False,  # Sampling without replacement
                           n_samples=500)  # Number of samples to select

# Concatenate the resampled healthy samples with the rest of the data
filtered_data_resampled = pd.concat([filtered_data[filtered_data['target'] != 0], healthy_samples])

# Shuffle the dataset to randomize the order
filtered_data_resampled = filtered_data_resampled.sample(frac=1).reset_index(drop=True)

# Upsample the second class to increase its count to 400
hyperthyroidism_samples_upsampled = resample(filtered_data_resampled[filtered_data_resampled['target'] == 1], 
                                            replace=True,  # Sampling with replacement
                                            n_samples=400)  # Number of samples to select

# Concatenate the upsampled hyperthyroidism samples with the rest of the data
filtered_data_resampled_upsampled = pd.concat([filtered_data_resampled[filtered_data_resampled['target'] != 1], hyperthyroidism_samples_upsampled])

# Shuffle the dataset to randomize the order
filtered_data_resampled_upsampled = filtered_data_resampled_upsampled.sample(frac=1).reset_index(drop=True)

# PRINCIPAL COMPONENT ANALYSIS

# Define X and y
X = filtered_data.drop(columns=['target'])
y = filtered_data['target']

# Standardise the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Fit PCA and transform the standardized features
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# FNN MODEL

# Define the input features (X) and target variable (y)
X = filtered_data_resampled_upsampled.drop(columns=['target'])
y = filtered_data_resampled_upsampled['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Calculate precision, recall, and F1-score during training
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_test)

train_precision = precision_score(y_train, np.argmax(train_predictions, axis=1), average='macro')
train_recall = recall_score(y_train, np.argmax(train_predictions, axis=1), average='macro')
train_f1_score = f1_score(y_train, np.argmax(train_predictions, axis=1), average='macro')

val_precision = precision_score(y_test, np.argmax(val_predictions, axis=1), average='macro')
val_recall = recall_score(y_test, np.argmax(val_predictions, axis=1), average='macro')
val_f1_score = f1_score(y_test, np.argmax(val_predictions, axis=1), average='macro')

history.history['precision'] = train_precision
history.history['recall'] = train_recall
history.history['f1_score'] = train_f1_score
history.history['val_precision'] = val_precision
history.history['val_recall'] = val_recall
history.history['val_f1_score'] = val_f1_score

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy:', test_accuracy)

# Plot accuracy-related metrics
plot_accuracy_metrics(history)

# Plot loss-related metrics
plot_loss_metrics(history)

# After training, to print accuracy and loss metrics
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Training Accuracy:", train_accuracy)
print("Training Loss:", train_loss)
print("Validation Accuracy:", val_accuracy)
print("Validation Loss:", val_loss)


# After training, to print accuracy and loss metrics
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Training Accuracy:", train_accuracy)
print("Training Loss:", train_loss)
print("Validation Accuracy:", val_accuracy)
print("Validation Loss:", val_loss)