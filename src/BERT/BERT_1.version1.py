from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import seaborn as sns

print("begin: " + str(datetime.now()))

# Load data
file_path = '../csv/incidents.10000.clean.csv'
data = pd.read_csv(file_path)

# 检查并显示缺失 `Severity` 数据的行数
missing_severity_count = data['Severity'].isnull().sum()
print(f"\nNumber of rows with missing Severity: {missing_severity_count}")

# 丢弃缺失 `Severity` 数据的行
data = data.dropna(subset=['Severity'])
print(f"\nDataset after dropping rows with missing Severity: {len(data)} rows remaining")

print("begin BERT: " + str(datetime.now()))

# Combine text columns
text_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Summary', 'Evidences', 'Concepts']
data['Combined_Text'] = data[text_columns].fillna('').agg(' '.join, axis=1)

# Encode the severity column
label_encoder = LabelEncoder()
data['Severity'] = label_encoder.fit_transform(data['Severity'])

# Split the data
X = data['Combined_Text']
y = data['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("after split: " + str(datetime.now()))
print("before sample")
print('0:', y_train.value_counts()[0], '/', round(y_train.value_counts()[0] / len(y_train) * 100, 2),
      '% of the dataset')
print('1:', y_train.value_counts()[1], '/', round(y_train.value_counts()[1] / len(y_train) * 100, 2),
      '% of the dataset')
print('2:', y_train.value_counts()[2], '/', round(y_train.value_counts()[2] / len(y_train) * 100, 2),
      '% of the dataset')
print('3:', y_train.value_counts()[3], '/', round(y_train.value_counts()[3] / len(y_train) * 100, 2),
      '% of the dataset')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def encode_texts(texts):
    encoded_inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt')
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    return outputs.last_hidden_state[:, 0, :]

# Encode summaries using BERT
X_train_encoded = encode_texts(X_train)
X_test_encoded = encode_texts(X_test)
print("end BERT: " + str(datetime.now()))

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded.cpu(), y_train)

print("after smote sample")
print('0:', y_train_smote.value_counts()[0], '/', round(y_train_smote.value_counts()[0] / len(y_train_smote) * 100, 2),
      '% of the dataset')
print('1:', y_train_smote.value_counts()[1], '/', round(y_train_smote.value_counts()[1] / len(y_train_smote) * 100, 2),
      '% of the dataset')
print('2:', y_train_smote.value_counts()[2], '/', round(y_train_smote.value_counts()[2] / len(y_train_smote) * 100, 2),
      '% of the dataset')
print('3:', y_train_smote.value_counts()[3], '/', round(y_train_smote.value_counts()[3] / len(y_train_smote) * 100, 2),
      '% of the dataset')

print("begin train model: " + str(datetime.now()))
# Initialize models
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Train models
rf_model.fit(X_train_smote, y_train_smote)
dt_model.fit(X_train_smote, y_train_smote)
lr_model.fit(X_train_smote, y_train_smote)

# Evaluate models
models = {'Random Forest': rf_model, 'Decision Tree': dt_model, 'Logistic Regression': lr_model}
results = {}

for model_name, model in models.items():
    y_pred = model.predict(X_test_encoded.cpu())
    results[model_name] = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("after train model: " + str(datetime.now()))
# Visualize confusion matrices
for model_name, model in models.items():
    y_pred = model.predict(X_test_encoded.cpu())
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Choose the best model (here we use accuracy as the criterion, but you can choose based on other metrics)
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Predict with the best model
y_pred_best = best_model.predict(X_test_encoded.cpu())
print(f"Best Model Classification Report:\n", classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Visualize the best model's confusion matrix
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'{best_model_name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
