from datetime import datetime
# section 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 1: Data Cleaning
print("begin step 1: " + str(datetime.now()))
df = pd.read_csv('csv/incidents.all.csv')

# Check and display rows with missing `Severity`
missing_severity_count = df['Severity'].isnull().sum()
print(f"\nNumber of rows with missing Severity: {missing_severity_count}")

# Plot missing `Severity` data
plt.figure(figsize=(6, 4))
plt.bar(['Missing Severity', 'Non-missing Severity'], [missing_severity_count, len(df) - missing_severity_count])
plt.ylabel('Number of Rows')
plt.title('Missing Severity Data')
plt.show()

# Drop rows with missing `Severity`
df = df.dropna(subset=['Severity'])
print(f"\nDataset after dropping rows with missing Severity: {len(df)} rows remaining")

print("\nColumns in the Dataset:")
print(df.columns)

# section 2
# Select relevant columns
relevant_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Severity', 'Summary', 'Evidences', 'Concepts']
df = df[relevant_columns]

# Remove rows where 'Severity' is empty
df = df.dropna(subset=['Severity'])

print("Data after cleaning:")
print(df.head())

labels = ["Non-physical harm", "Hazard", "Death", "Injury"]

harm_type = df["Severity"].value_counts().tolist()
print(harm_type)
values = [harm_type[0], harm_type[1], harm_type[2], harm_type[3]]

fig = px.pie(values=df['Severity'].value_counts(), names=labels, width=500, height=400, color_discrete_sequence=["skyblue", "black", "red", "orange"], title="AI incident Severities")
fig.show()

print("begin step 2: " + str(datetime.now()))

# Step 2: Feature Processing
# Define a function to convert text features to numeric using TF-IDF vectorization
def text_to_tfidf(df, column):
    df[column] = df[column].fillna('')  # Replace NaNs with empty strings
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df[column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.columns = [f"{column}_{col}" for col in tfidf_df.columns]  # Rename columns to include original column name
    return tfidf_df

# Initialize an empty dataframe to hold all TF-IDF features
tfidf_features_df = pd.DataFrame()
column_names = []

# Process each text column and merge into tfidf_features_df
for column in ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Summary', 'Evidences', 'Concepts']:
    tfidf_df = text_to_tfidf(df, column)
    tfidf_features_df = pd.concat([tfidf_features_df, tfidf_df], axis=1)
    column_names.extend([column] * tfidf_df.shape[1])

print("TF-IDF features dataframe shape:", tfidf_features_df.shape)

print("begin step 3: " + str(datetime.now()))

# Step 3: Mutual Information Calculation
# Calculate mutual information scores for each feature
X = tfidf_features_df
y = df['Severity']

# Ensure y is a category type
y = y.astype('category')
y_encoded = y.cat.codes

mi_scores = mutual_info_classif(X, y_encoded)

print("begin step 4: " + str(datetime.now()))

# Step 4: Feature Importance Ranking
# Aggregate importance scores for each original column
feature_importance_df = pd.DataFrame({'Feature': column_names, 'Importance': mi_scores})
aggregated_importance = feature_importance_df.groupby('Feature').sum().reset_index()
aggregated_importance = aggregated_importance.sort_values(by='Importance', ascending=False)

top_feature_num = 5
top_aggregated_importance = aggregated_importance.head(top_feature_num)

print(f"Top {top_feature_num} important features:")
print(top_aggregated_importance)

print("begin step 5: " + str(datetime.now()))

# Step 5: Visualization
# Plot the top {top_feature_num} most important features
plt.figure(figsize=(12, 8))
plt.barh(aggregated_importance['Feature'].head(top_feature_num), aggregated_importance['Importance'].head(top_feature_num))
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title(f"Top {top_feature_num} Important Features")
plt.gca().invert_yaxis()
plt.show()

# section 3
# Select top 5 important features for simplicity
top_features = top_aggregated_importance['Feature']

# Filter TF-IDF features to include only top features
selected_features = tfidf_features_df.loc[:, tfidf_features_df.columns.str.startswith(tuple(top_features))]

print(selected_features)

# Step 8: Splitting Data
X_train, X_temp, y_train, y_temp = train_test_split(selected_features, y_encoded, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print("before sample")
print('0:', y_train.value_counts()[0], '/', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
print('1:', y_train.value_counts()[1], '/',round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')
print('2:', y_train.value_counts()[2], '/',round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')
print('3:', y_train.value_counts()[3], '/',round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')

#add smote
# Step 7: Oversampling with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("after sample")
print('0:', y_resampled.value_counts()[0], '/', round(y_resampled.value_counts()[0]/len(y_resampled) * 100,2), '% of the dataset')
print('1:', y_resampled.value_counts()[1], '/',round(y_resampled.value_counts()[1]/len(y_resampled) * 100,2), '% of the dataset')
print('2:', y_resampled.value_counts()[2], '/',round(y_resampled.value_counts()[1]/len(y_resampled) * 100,2), '% of the dataset')
print('3:', y_resampled.value_counts()[3], '/',round(y_resampled.value_counts()[1]/len(y_resampled) * 100,2), '% of the dataset')


# Step 9: Model Training and Evaluation
models = {
    'Random Forest': RandomForestClassifier()
    #'Logistic Regression': LogisticRegression(max_iter=1000),
    #'SVM': SVC()
}

results = {}
for model_name, model in models.items():
    model.fit(X_resampled, y_resampled)
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
plt.show()

print("begin step 10: " + str(datetime.now()))

# Step 10: Predicting and Evaluating on Test Set
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
sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.cat.categories, yticklabels=y.cat.categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.show()
print("end step 10: " + str(datetime.now()))
