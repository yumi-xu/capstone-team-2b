from datetime import datetime
# section 1
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from serialization_utils import deserialize_from_file
import plotly.express as px
import os

file_name = os.path.basename(__file__)
distDir = f'dist/{file_name[:-3]}'
os.makedirs(distDir, exist_ok=True)

tmpDir = 'tmp/version6_all'
os.makedirs(tmpDir, exist_ok=True)

print("begin: " + str(datetime.now()))
tfidf_features_df = deserialize_from_file(f'{tmpDir}/tfidf_features_df.pkl')
y = deserialize_from_file(f'{tmpDir}/y.pkl')
y_cat = deserialize_from_file(f'{tmpDir}/y_cat.pkl')
y_encoded = deserialize_from_file(f'{tmpDir}/y_encoded.pkl')
top_aggregated_importance = deserialize_from_file(f'{tmpDir}/top_aggregated_importance.pkl')
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

severities = ["Non-physical harm", "Hazard", "Death", "Injury"]

model_names = ['Logistic Regression', 'Random Forest', 'Decision Tree']
sample_methods = ['Original', 'SMOTE']

def format_timedelta(td):
    days = td.days
    seconds = td.seconds
    microseconds = td.microseconds

    # Convert total seconds to hours, minutes, and remaining seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create the formatted string
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    # Join the parts with commas and return the result
    return ", ".join(parts) if parts else "0 seconds"

_last_log_time = datetime.now()


def log(*args):
    time = str(datetime.now())
    print(f"[{time}]", *args)


def log_start(*args):
    global _last_log_time
    _last_log_time = datetime.now()
    print(f"[{_last_log_time}]", *args)


def log_end(*args):
    global _last_log_time
    before = _last_log_time
    now = datetime.now()
    _last_log_time = now
    td = now - before
    print(f"[{_last_log_time}]", *args, f"(cost {format_timedelta(td)})")


# Step 4: Splitting Data
# X_train, X_temp, y_train, y_temp = train_test_split(selected_features, y_encoded, test_size=0.30, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(selected_features, y, stratify=y, test_size=0.4, random_state=42)


def print_sample_information(y, sample_method):
    counts = y.value_counts()
    print(f"Sample Information[{sample_method}]")

    i = 0
    for severity in severities:
        count = counts[severity]
        print(f'[{i}]{severity}:', count, '/', round(count / len(y) * 100, 2),
              '% of the dataset')
        i = i + 1


_get_model_params_cache = {}


def _get_model_params(sample_method):
    if sample_method == 'Original':
        return [X_train, y_train, X_test, y_test]

    if sample_method == 'SMOTE':
        log_start('start smote.fit_resample(X_train, y_train)')
        smote = SMOTE(random_state=42)
        X_SMOTE_over, y_SMOTE_over = smote.fit_resample(X_train, y_train)
        log_end('end smote.fit_resample(X_train, y_train)')
        return [X_SMOTE_over, y_SMOTE_over, X_test, y_test]

    if sample_method == 'Random Oversampling':
        log_start('start ros.fit_resample(X_train, y_train)')
        ros = RandomOverSampler(random_state=40)
        X_rd_over, y_rd_over = ros.fit_resample(X_train, y_train)
        log_end('start ros.fit_resample(X_train, y_train)')
        return [X_rd_over, y_rd_over, X_test, y_test]

    raise ValueError(f'sample_method {sample_method} is not supported')


def get_model_params(sample_method):
    global _get_model_params_cache
    value = _get_model_params_cache.get(sample_method)
    if not value:
        value = _get_model_params(sample_method)
        _get_model_params_cache[sample_method] = value
    return value


def get_model_by_name(model_name):
    if model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000)

    if model_name == 'Random Forest':
        return RandomForestClassifier()

    if model_name == 'Decision Tree':
        return DecisionTreeClassifier()

    raise ValueError(f'Model name {model_name} not recognized')


def evaluate_models(sample_method):
    X_train, y_train, X_test, y_test = get_model_params(sample_method)
    print_sample_information(y_train, sample_method)
    results = {}
    for model_name in model_names:
        model = get_model_by_name(model_name)

        log_start(f"[{model_name}] start model.fit(X_train, y_train)")
        model.fit(X_train, y_train)
        log_end(f"[{model_name}] end model.fit(X_train, y_train)")

        log_start(f"[{model_name}] start model.predict(X_test)")
        y_test_pred = model.predict(X_test)
        log_end(f"[{model_name}] end model.predict(X_test)")

        log_start(f"[{model_name}] start accuracy_score(y_test, y_test_pred)")
        accuracy = accuracy_score(y_test, y_test_pred)
        log_end(f"[{model_name}] end accuracy_score(y_test, y_test_pred)")

        log_start(f"[{model_name}] start classification_report(y_test, y_test_pred)")
        classification_report_value = classification_report(y_test, y_test_pred)
        log_end(f"[{model_name}] start classification_report(y_test, y_test_pred)")

        log_start(f"[{model_name}] start confusion_matrix(y_test, y_test_pred)")
        confusion_matrix_value = confusion_matrix(y_test, y_test_pred)
        log_end(f"[{model_name}] start confusion_matrix(y_test, y_test_pred)")

        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': classification_report_value,
            'confusion_matrix': confusion_matrix_value
        }
    print_results(results, sample_method)
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


# Visualization of Results

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


results_list = [evaluate_models(sample_method) for sample_method in sample_methods]
visualize_accuracies(results_list, sample_methods, f'{distDir}/validation_accuracy_of_different_models_comparison.png')

# Example usage for each scenario
evaluate_and_visualize_test_set(results_list[0], X_test, y_test, y_cat,
                                f'{distDir}/confusion_matrix_on_test_set_original.png')
evaluate_and_visualize_test_set(results_list[1], X_test, y_test, y_cat, f'{distDir}/confusion_matrix_on_test_set_smote.png')
# evaluate_and_visualize_test_set(results_list[2], X_test, y_test, y_cat, f'{distDir}/confusion_matrix_on_test_set_random.png')

print("end step 6: " + str(datetime.now()))
