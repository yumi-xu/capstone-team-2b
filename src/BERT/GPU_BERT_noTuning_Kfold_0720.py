from datetime import datetime
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

print("begin: " + str(datetime.now()))

# Load data
file_path = '../csv/incidents.all.csv'
data = pd.read_csv(file_path)

# 检查并显示缺失 Severity 数据的行数
missing_severity_count = data['Severity'].isnull().sum()
print(f"\nNumber of rows with missing Severity: {missing_severity_count}")

# 丢弃缺失 Severity 数据的行
data = data.dropna(subset=['Severity'])
print(f"\nDataset after dropping rows with missing Severity: {len(data)} rows remaining")

print("begin BERT: " + str(datetime.now()))

# Combine text columns
text_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Summary', 'Evidences',
                'Concepts']
data['Combined_Text'] = data[text_columns].fillna('').agg(' '.join, axis=1)

# Encode the severity column
label_encoder = LabelEncoder()
data['Severity'] = label_encoder.fit_transform(data['Severity'])

X = data['Combined_Text']
y = data['Severity']

# 显示文本长度分布
data['text_length'] = data['Combined_Text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=50, kde=True)
plt.title("Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig('output/Text_Length_Distribution.png')
# plt.show()

print("before sample")
print('0:', y.value_counts()[0], '/', round(y.value_counts()[0] / len(y) * 100, 2), '% of the dataset')
print('1:', y.value_counts()[1], '/', round(y.value_counts()[1] / len(y) * 100, 2), '% of the dataset')
print('2:', y.value_counts()[2], '/', round(y.value_counts()[2] / len(y) * 100, 2), '% of the dataset')
print('3:', y.value_counts()[3], '/', round(y.value_counts()[3] / len(y) * 100, 2), '% of the dataset')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def encode_texts(texts, batch_size=16):
    encoded_texts = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size].tolist()
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        encoded_texts.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(encoded_texts)


print("begin BERT: " + str(datetime.now()))
# Encode summaries using BERT
X_encoded = encode_texts(X)
print("end BERT: " + str(datetime.now()))


# BERT特征可视化
def visualize_features(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_features = reducer.fit_transform(features)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette="viridis", legend="full")
    plt.title(f'BERT Features Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best')
    plt.savefig(f'output/BERT_Features_{method.upper()}.png')
    # plt.show()


# 可视化BERT编码后的特征（使用PCA）
visualize_features(X_encoded, y, method='pca')

# 可视化BERT编码后的特征（使用t-SNE）
visualize_features(X_encoded, y, method='tsne')

# Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_encoded, y)

# 显示SMOTE处理后的类别分布
plt.figure(figsize=(10, 6))
sns.countplot(x=y_smote)
plt.title("Class Distribution after SMOTE")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig('output/Class_Distribution_SMOTE.png')
# plt.show()

print("after smote sample")
print('0:', y_smote.value_counts()[0], '/', round(y_smote.value_counts()[0] / len(y_smote) * 100, 2),
      '% of the dataset')
print('1:', y_smote.value_counts()[1], '/', round(y_smote.value_counts()[1] / len(y_smote) * 100, 2),
      '% of the dataset')
print('2:', y_smote.value_counts()[2], '/', round(y_smote.value_counts()[2] / len(y_smote) * 100, 2),
      '% of the dataset')
print('3:', y_smote.value_counts()[3], '/', round(y_smote.value_counts()[3] / len(y_smote) * 100, 2),
      '% of the dataset')

print("begin train model: " + str(datetime.now()))
# Initialize models
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Prepare Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Function to train and evaluate models using cross-validation
def cross_val_evaluate(model, X, y, cv):
    y_pred = cross_val_predict(Pipeline([('smote', smote), ('model', model)]), X, y, cv=cv, method='predict_proba')
    y_pred_labels = np.argmax(y_pred, axis=1)
    report = classification_report(y, y_pred_labels, target_names=label_encoder.classes_, output_dict=True)

    # 绘制混淆矩阵
    cm = confusion_matrix(y, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'{model.__class__.__name__} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'output/{model.__class__.__name__}_confusion_matrix.png')  # Save the confusion matrix as an image file
    plt.close()  # Close the plot to avoid display

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for i in range(len(label_encoder.classes_)):
        fpr, tpr, _ = roc_curve(y, y_pred[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {label_encoder.classes_[i]} (area = {roc_auc:.2f})')

    plt.title(f'{model.__class__.__name__} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig(f'output/{model.__class__.__name__}_roc_curve.png')  # Save the ROC curve as an image file
    plt.close()  # Close the plot to avoid display

    return report, y_pred_labels


# Train and evaluate models using cross-validation
models = {'Random Forest': rf_model, 'Decision Tree': dt_model, 'Logistic Regression': lr_model}
results = {}

for model_name, model in models.items():
    results[model_name], y_pred = cross_val_evaluate(model, X_encoded, y, skf)
    print(f"{model_name} Classification Report:\n",
          classification_report(y, y_pred, target_names=label_encoder.classes_))

print("after train model: " + str(datetime.now()))

# Choose the best model (here we use accuracy as the criterion, but you can choose based on other metrics)
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

print("end: " + str(datetime.now()))

