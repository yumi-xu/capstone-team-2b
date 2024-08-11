from datetime import datetime
# section 1
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from serialization_utils import deserialize_from_file
import plotly.express as px
import os

file_name = os.path.basename(__file__)
distDir = f'dist/{file_name[:-3]}'
os.makedirs(distDir, exist_ok=True)

print("begin: " + str(datetime.now()))
tfidf_features_df = deserialize_from_file('tmp/tfidf_features_df.pkl')
y = deserialize_from_file('tmp/y.pkl')
y_cat = deserialize_from_file('tmp/y_cat.pkl')
y_encoded = deserialize_from_file('tmp/y_encoded.pkl')
top_aggregated_importance = deserialize_from_file('tmp/top_aggregated_importance.pkl')
top_features = top_aggregated_importance['Feature']

# Filter TF-IDF features to include only top features
selected_features = tfidf_features_df.loc[:, tfidf_features_df.columns.str.startswith(tuple(top_features))]

print("selected_features:")
print(selected_features)
print("y:")
print(y)
print("y_cat:")
print(y_cat)
print("y_encode")
print(y_encoded)

# Step 4: Splitting Data
#X_train, X_temp, y_train, y_temp = train_test_split(selected_features, y_encoded, test_size=0.30, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(selected_features, y, stratify=y, test_size=0.4, random_state=42)

print("before sample")
print('0:', y_train.value_counts()[0], '/', round(y_train.value_counts()[0] / len(y_train) * 100, 2),
      '% of the dataset')
print('1:', y_train.value_counts()[1], '/', round(y_train.value_counts()[1] / len(y_train) * 100, 2),
      '% of the dataset')
print('2:', y_train.value_counts()[2], '/', round(y_train.value_counts()[1] / len(y_train) * 100, 2),
      '% of the dataset')
print('3:', y_train.value_counts()[3], '/', round(y_train.value_counts()[1] / len(y_train) * 100, 2),
      '% of the dataset')

# random oversampling
ros = RandomOverSampler(random_state=40)
X_rd_over, y_rd_over = ros.fit_resample(X_train, y_train)
# smote
#sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=42)
smote = SMOTE(random_state=42)
X_SMOTE_over, y_SMOTE_over = smote.fit_resample(X_train, y_train)

print("after smote sample")
print('0:', y_SMOTE_over.value_counts()[0], '/', round(y_SMOTE_over.value_counts()[0] / len(y_SMOTE_over) * 100, 2),
      '% of the dataset')
print('1:', y_SMOTE_over.value_counts()[1], '/', round(y_SMOTE_over.value_counts()[1] / len(y_SMOTE_over) * 100, 2),
      '% of the dataset')
print('2:', y_SMOTE_over.value_counts()[2], '/', round(y_SMOTE_over.value_counts()[1] / len(y_SMOTE_over) * 100, 2),
      '% of the dataset')
print('3:', y_SMOTE_over.value_counts()[3], '/', round(y_SMOTE_over.value_counts()[1] / len(y_SMOTE_over) * 100, 2),
      '% of the dataset')

print("after random sample")
print('0:', y_rd_over.value_counts()[0], '/', round(y_rd_over.value_counts()[0] / len(y_rd_over) * 100, 2),
      '% of the dataset')
print('1:', y_rd_over.value_counts()[1], '/', round(y_rd_over.value_counts()[1] / len(y_rd_over) * 100, 2),
      '% of the dataset')
print('2:', y_rd_over.value_counts()[2], '/', round(y_rd_over.value_counts()[1] / len(y_rd_over) * 100, 2),
      '% of the dataset')
print('3:', y_rd_over.value_counts()[3], '/', round(y_rd_over.value_counts()[1] / len(y_rd_over) * 100, 2),
      '% of the dataset')


def evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        # 'Logistic Regression': LogisticRegression(max_iter=1000),
        # 'SVM': SVC()
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }

    return results


def print_results(results, description):
    print(f"Results using {description}")
    for model_name, result in results.items():
        print(f"Model: {model_name}")
        print(f"Validation Accuracy: {result['accuracy']}")
        print("Classification Report:")
        print(result['classification_report'])
        print("Confusion Matrix:")
        print(result['confusion_matrix'])
        print("\n")


def visualize_accuracies(results_list, labels, filename):
    model_names = list(results_list[0].keys())
    accuracies = {label: [results[model_name]['accuracy'] for model_name in model_names] for label, results in
                  zip(labels, results_list)}

    plt.figure(figsize=(10, 6))
    for label, acc in accuracies.items():
        sns.barplot(x=model_names, y=acc, alpha=0.6, label=label)
    plt.xlabel('Models')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy of Different Models')
    plt.legend()
    plt.savefig(filename)


# Evaluate models using original training data
results_original = evaluate_models(X_train, y_train, X_test, y_test)
print_results(results_original, "Original Data")

# Evaluate models using SMOTE
results_smote = evaluate_models(X_SMOTE_over, y_SMOTE_over, X_test, y_test)
print_results(results_smote, "SMOTE")

# Evaluate models using Random Oversampling
results_random = evaluate_models(X_rd_over, y_rd_over, X_test, y_test)
print_results(results_random, "Random Oversampling")

# Visualization of Results
results_list = [results_original, results_smote, results_random]
labels = ['Original', 'SMOTE', 'Random Oversampling']
visualize_accuracies(results_list, labels, f'{distDir}/validation_accuracy_of_different_models_comparison.png')


# Define a function to evaluate and visualize test set results
def evaluate_and_visualize_test_set(model_results, X_test, y_test, y_cat, save_filename):
    best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
    best_model = model_results[best_model_name]['model']

    # Predict on test set
    y_test_pred = best_model.predict(X_test)

    # Test set evaluation
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_classification_report = classification_report(y_test, y_test_pred)
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

    # Print evaluation results
    print(f"Test Accuracy of the Best Model ({best_model_name}): {test_accuracy}")
    print("Test Classification Report:")
    print(test_classification_report)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)

    # Visualization of Confusion Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_cat, yticklabels=y_cat)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix on Test Set using {best_model_name}')
    plt.savefig(save_filename)


# Example usage for each scenario
evaluate_and_visualize_test_set(results_original, X_test, y_test, y_cat,f'{distDir}/confusion_matrix_on_test_set_original.png')
# evaluate_and_visualize_test_set(results_smote, X_test, y_test, y_cat, f'{distDir}/confusion_matrix_on_test_set_smote.png')
# evaluate_and_visualize_test_set(results_random, X_test, y_test, y_cat, f'{distDir}/confusion_matrix_on_test_set_random.png')

print("end step 6: " + str(datetime.now()))
