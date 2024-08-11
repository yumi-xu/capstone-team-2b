from datetime import datetime
# section 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from serialization_utils import deserialize_from_file
import os

tmpDir = 'tmp/version6_1000'
os.makedirs(tmpDir, exist_ok=True)

print("begin: " + str(datetime.now()))
tfidf_features_df = deserialize_from_file(f'{tmpDir}/tfidf_features_df.pkl')
y_cat = deserialize_from_file(f'{tmpDir}/y_cat.pkl')
y_encoded = deserialize_from_file(f'{tmpDir}/y_encoded.pkl')
top_aggregated_importance = deserialize_from_file(f'{tmpDir}/top_aggregated_importance.pkl')
top_features = top_aggregated_importance['Feature']

# Filter TF-IDF features to include only top features
selected_features = tfidf_features_df.loc[:, tfidf_features_df.columns.str.startswith(tuple(top_features))]

print(selected_features)

# Step 4: Splitting Data
X_train, X_temp, y_train, y_temp = train_test_split(selected_features, y_encoded, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Step 5: Model Training and Evaluation
models = {
    'Random Forest': RandomForestClassifier()
    #'Logistic Regression': LogisticRegression(max_iter=1000),
    #'SVM': SVC()
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'classification_report': classification_report(y_val, y_val_pred),
        'confusion_matrix': confusion_matrix(y_val, y_val_pred)
    }

# Display results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Validation Accuracy: {result['accuracy']}")
    print("Classification Report:")
    print(result['classification_report'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("\n")

# Visualization of Results
model_names = list(results.keys())
accuracies = [results[model_name]['accuracy'] for model_name in model_names]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies)
plt.xlabel('Models')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy of Different Models')
plt.savefig('dist/validation_accuracy_of_different_models.png')
print("begin step 6: " + str(datetime.now()))
# Step 6: Predicting and Evaluating on Test Set
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

y_test_pred = best_model.predict(X_test)

# Test set evaluation
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_report = classification_report(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

print(f"Test Accuracy of the Best Model ({best_model_name}): {test_accuracy}")
print("Test Classification Report:")
print(test_classification_report)
print("Test Confusion Matrix:")
print(test_confusion_matrix)

# Visualization of Test Set Results
plt.figure(figsize=(10, 6))
sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_cat, yticklabels=y_cat)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.savefig('dist/confusion_matrix_on_test_set.png')
print("end step 6: " + str(datetime.now()))