import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time

data = pd.read_csv('dataset.csv')

print(data.head())
print(data.info())
print(data.describe())
# Handle missing values, outliers, and scaling, if needed


# Define the features (X) and the target (y)
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Neural Network': MLPClassifier(),
    'K-NN': KNeighborsClassifier(),
    'SVM': SVC()
}

# Train and evaluate the classifiers
results = []

for name, classifier in classifiers.items():
    # Train the model
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Test the model
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    test_time = time.time() - start_time

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store the results
    results.append({
        'Classifier': name,
        'Confusion Matrix': conf_matrix,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Train Time': train_time,
        'Test Time': test_time
    })
# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

# Visualize the results (e.g., using Seaborn or Matplotlib)
# Example: Plotting the accuracy of different classifiers
plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Accuracy', data=results_df)
plt.title('Classifier Accuracy Comparison')
plt.show()