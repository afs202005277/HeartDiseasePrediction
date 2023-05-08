from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sb

train_data = pd.read_csv('fixed_train_data.csv')
train_data_07 = pd.read_csv('fixed_train_data_07.csv')
train_data_08 = pd.read_csv('fixed_train_data_08.csv')
train_data_09 = pd.read_csv('fixed_train_data_09.csv')

names = ['Train Data', 'Train Data 07', 'Train Data 08', 'Train Data 09']

datasets = [train_data, train_data_07, train_data_08, train_data_09]
target_column = "is_claim"
results = []
params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 5, 10, 20, 40, 80, 160, 320, 640, None],
    'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'max_features': [None] + [x for x in range(1, len(train_data.columns))]
}

for dataset in datasets:
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    y_pred = grid_search.predict(X_test)
    print(best_params)
    # Best params results:
    # {'criterion': 'entropy', 'max_depth': 20, 'max_features': 32, 'min_samples_leaf': 0.1, 'min_samples_split': 0.5, 'splitter': 'best'}
    # {'criterion': 'entropy', 'max_depth': 20, 'max_features': 3, 'min_samples_leaf': 0.1, 'min_samples_split': 0.6, 'splitter': 'best'}
    # {'criterion': 'entropy', 'max_depth': 640, 'max_features': 5, 'min_samples_leaf': 0.1, 'min_samples_split': 0.6, 'splitter': 'best'}
    # {'criterion': 'gini', 'max_depth': 10, 'max_features': 6, 'min_samples_leaf': 0.1, 'min_samples_split': 0.2, 'splitter': 'best'}

    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store the results
    results.append({
        'Confusion Matrix': conf_matrix,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'F1 Score': f1,
    })
for i, result in enumerate(results):
    # Create a heatmap of the confusion matrix
    sb.heatmap(result['Confusion Matrix'], annot=True, cmap='Blues')

    # Add labels and title to the plot
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for ' + names[i])

    # Show the plot
    plt.show()
metrics = ['Precision', 'Recall', 'Accuracy', 'F1 Score']
for metric in metrics:
    # Extract the metric values for each dataset
    values = [result[metric] for result in results]

    # Create a bar plot to compare the metric values
    sb.barplot(x=names, y=values)

    # Add labels and title to the plot
    plt.xlabel('Dataset')
    plt.ylabel(metric)
    plt.title(f'{metric} comparison across datasets')

    # Show the plot
    plt.show()
