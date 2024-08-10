from datetime import datetime
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import numpy as np

print("begin: " + str(datetime.now()))

# Load the dataset
try:
    data = pd.read_csv('../csv/incidents.all.csv')
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Check and display the number of rows with missing `Severity` data
missing_severity_count = data['Severity'].isnull().sum()
print(f"\nNumber of rows with missing Severity: {missing_severity_count}")

# Drop rows with missing `Severity` data
data = data.dropna(subset=['Severity'])
print(f"\nDataset after dropping rows with missing Severity: {len(data)} rows remaining")

print("begin word2Vec: " + str(datetime.now()))

# Combine relevant text columns into one
text_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders',
                'Harm Types', 'Summary', 'Evidences', 'Concepts', 'Country']
data['combined_text'] = data[text_columns].fillna('').astype(str).agg(' '.join, axis=1)

# Preprocess text data and tokenize
data['tokenized_text'] = data['combined_text'].apply(gensim.utils.simple_preprocess)

# Print tokenized data sample
print(data['tokenized_text'].head())

# Train a Word2Vec model
try:
    w2v_model = Word2Vec(sentences=data['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)
    print("Word2Vec model trained successfully")
except Exception as e:
    print(f"Error training Word2Vec model: {e}")

# Print some information about the model
print(w2v_model)

# Vectorize the text using Word2Vec
def vectorize_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

data['vectorized_text'] = data['tokenized_text'].apply(lambda x: vectorize_text(x, w2v_model))

print("after word2Vec: " + str(datetime.now()))

# Print vectorized data sample
print(data['vectorized_text'].head())

# Prepare feature matrix and labels
X = np.vstack(data['vectorized_text'].values)
y = data['Severity']

# Convert labels to numeric values if they are not already
if y.dtype == 'object':
    y = y.astype('category').cat.codes

# Save mapping of Severity levels
severity_mapping = dict(enumerate(data['Severity'].astype('category').cat.categories))
inverse_severity_mapping = {v: k for k, v in severity_mapping.items()}

# Convert y to a NumPy array
y = y.to_numpy()

# Visualize the class distribution before SMOTE
plt.figure(figsize=(10, 5))
plt.hist(y, bins=len(set(y)), edgecolor='black')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.savefig(f'pic/class_distribution_before_smote.png')
plt.close()

# Define the models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# Function to perform stratified k-fold cross-validation
def k_fold_cv(X, y, models, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = {name: {"confusion_matrix": np.zeros((len(set(y)), len(set(y)))), "classification_report": None} for name in models.keys()}
    metrics = {name: {"accuracy": [], "precision": [], "recall": [], "f1": []} for name in models.keys()}

    for name, model in models.items():
        y_true_all = []
        y_pred_all = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        cm = confusion_matrix(y_true_all, y_pred_all)
        report = classification_report(y_true_all, y_pred_all, output_dict=True)
        results[name]["confusion_matrix"] = cm
        results[name]["classification_report"] = report

        # Collect metrics for visualization
        metrics[name]["accuracy"].append(accuracy_score(y_true_all, y_pred_all))
        metrics[name]["precision"].append(precision_score(y_true_all, y_pred_all, average='weighted'))
        metrics[name]["recall"].append(recall_score(y_true_all, y_pred_all, average='weighted'))
        metrics[name]["f1"].append(f1_score(y_true_all, y_pred_all, average='weighted'))

    return results, metrics

# Apply stratified k-fold cross-validation on the original data
print("Performing stratified k-fold cross-validation on original data...")
original_results, original_metrics = k_fold_cv(X, y, models, k=5)
print("Cross-validation results on original data completed")

# Function to perform stratified k-fold cross-validation with SMOTE
def k_fold_cv_smote(X, y, models, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = {name: {"confusion_matrix": np.zeros((len(set(y)), len(set(y)))), "classification_report": None} for name in models.keys()}
    metrics = {name: {"accuracy": [], "precision": [], "recall": [], "f1": []} for name in models.keys()}

    for name, model in models.items():
        y_true_all = []
        y_pred_all = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        cm = confusion_matrix(y_true_all, y_pred_all)
        report = classification_report(y_true_all, y_pred_all, output_dict=True)
        results[name]["confusion_matrix"] = cm
        results[name]["classification_report"] = report

        # Collect metrics for visualization
        metrics[name]["accuracy"].append(accuracy_score(y_true_all, y_pred_all))
        metrics[name]["precision"].append(precision_score(y_true_all, y_pred_all, average='weighted'))
        metrics[name]["recall"].append(recall_score(y_true_all, y_pred_all, average='weighted'))
        metrics[name]["f1"].append(f1_score(y_true_all, y_pred_all, average='weighted'))

    return results, metrics

# Apply stratified k-fold cross-validation with SMOTE
print("Performing stratified k-fold cross-validation with SMOTE...")
smote_results, smote_metrics = k_fold_cv_smote(X, y, models, k=5)
print("Cross-validation results on SMOTE data completed")

# Visualize the class distribution after SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
plt.figure(figsize=(10, 5))
plt.hist(y_res, bins=len(set(y)), edgecolor='black')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.savefig(f'pic/class_distribution_after_smote.png')
plt.close()

# Visualization of results
def plot_confusion_matrices(results, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(title)

    for ax, (name, result) in zip(axes, results.items()):
        cm = result["confusion_matrix"]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.xaxis.set_ticklabels([''] + [severity_mapping[i] for i in range(len(set(y)))])
        ax.yaxis.set_ticklabels([''] + [severity_mapping[i] for i in range(len(set(y)))])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes)
    plt.savefig(filename)
    plt.close()

print("Plotting results...")
plot_confusion_matrices(original_results, "Confusion Matrices on Original Data", 'pic/confusion_matrices_original_data.png')
plot_confusion_matrices(smote_results, "Confusion Matrices on SMOTE Data", 'pic/confusion_matrices_smote_data.png')

# Display classification reports
def print_classification_reports(results, title):
    print(title)
    for name, result in results.items():
        print(f"Model: {name}")
        report_df = pd.DataFrame(result["classification_report"]).transpose()
        # Rename the index to replace 0, 1, 2, 3 with severity labels
        report_df.index = [severity_mapping[int(index)] if index.isdigit() else index for index in report_df.index]
        print(report_df)
        print("\n")

print_classification_reports(original_results, "Classification Reports on Original Data")
print_classification_reports(smote_results, "Classification Reports on SMOTE Data")

# Plot model performance metrics
def plot_metrics(metrics, title, filename):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title)
    metric_names = ["accuracy", "precision", "recall", "f1"]

    for ax, metric_name in zip(axes, metric_names):
        for name, metric in metrics.items():
            ax.plot(range(1, len(metric[metric_name])+1), metric[metric_name], marker='o', label=name)
        ax.set_title(metric_name.capitalize())
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()

    plt.savefig(filename)
    plt.close()

plot_metrics(original_metrics, "Model Performance Metrics on Original Data", 'pic/metrics_original_data.png')
plot_metrics(smote_metrics, "Model Performance Metrics on SMOTE Data", 'pic/metrics_smote_data.png')

print("end: " + str(datetime.now()))
