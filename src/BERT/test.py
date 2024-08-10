from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("begin: " + str(datetime.now()))

# Load data
file_path = '../csv/incidents.all.csv'
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

# 显示训练集中的类别分布
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train)
plt.title("Class Distribution in Training Set")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig('output/Class_Distribution_Train.png')
#plt.show()

# 显示文本长度分布
text_lengths = X_train.apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(text_lengths, bins=50)
plt.title("Text Length Distribution in Training Set")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.savefig('output/Text_Length_Distribution.png')
#plt.show()

print("begin tokenizing: " + str(datetime.now()))

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def tokenize_and_embed(texts):
    encoded_inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**encoded_inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Tokenize and embed text data
X_train_encoded = tokenize_and_embed(X_train)
X_test_encoded = tokenize_and_embed(X_test)
print("after tokenizing: " + str(datetime.now()))

# Training models on original data
print("Training models on original data...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

results_original = {}

for model_name, model in models.items():
    print(f"Training {model_name} on original data...")
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    accuracy = model.score(X_test_encoded, y_test)
    print(f"{model_name} Accuracy on original data: {accuracy:.4f}")
    results_original[model_name] = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_encoded)
    else:
        y_score = model.decision_function(X_test_encoded)
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(label_encoder.classes_):
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
        plt.plot(fpr[class_label], tpr[class_label], lw=2,
                 label=f'ROC curve of class {class_label} (area = {roc_auc[class_label]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name} (Original Data)')
    plt.legend(loc="lower right")
    plt.savefig(f'output/{model_name}_roc_curve_original.png')
    plt.close()

    # Confusion Matrix
    cm = results_original[model_name]['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'{model_name} Confusion Matrix (Original Data)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'output/{model_name}_confusion_matrix_original.png')
    plt.close()

print("after original model training: " + str(datetime.now()))

# SMOTE
print("Applying SMOTE and training models...")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train)

# 显示SMOTE后的类别分布
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_smote)
plt.title("Class Distribution in Training Set after SMOTE")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig('output/Class_Distribution_Train_SMOTE.png')
#plt.show()

results_smote = {}

for model_name, model in models.items():
    print(f"Training {model_name} on SMOTE data...")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_encoded)
    accuracy = model.score(X_test_encoded, y_test)
    print(f"{model_name} Accuracy on SMOTE data: {accuracy:.4f}")
    results_smote[model_name] = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_encoded)
    else:
        y_score = model.decision_function(X_test_encoded)
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(label_encoder.classes_):
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
        plt.plot(fpr[class_label], tpr[class_label], lw=2,
                 label=f'ROC curve of class {class_label} (area = {roc_auc[class_label]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name} (SMOTE Data)')
    plt.legend(loc="lower right")
    plt.savefig(f'output/{model_name}_roc_curve_smote.png')
    plt.close()

    # Confusion Matrix
    cm = results_smote[model_name]['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'{model_name} Confusion Matrix (SMOTE Data)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'output/{model_name}_confusion_matrix_smote.png')
    plt.close()

print("after train model: " + str(datetime.now()))

# Choose the best model (here we use accuracy as the criterion, but you can choose based on other metrics)
best_model_name = max(results_smote, key=lambda k: results_smote[k]['accuracy'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Predict with the best model
y_pred_best = best_model.predict(X_test_encoded)
print(f"Best Model Classification Report:\n",
      classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Visualize the best model's confusion matrix
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'{best_model_name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'output/{best_model_name}_best_confusion_matrix.png')
plt.close()

print("end: " + str(datetime.now()))
