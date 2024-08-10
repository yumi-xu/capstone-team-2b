from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
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
gpu_available = torch.cuda.is_available()
print("GPU available: ", gpu_available)

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
text_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Summary', 'Evidences',
                'Concepts']
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
data['text_length'] = data['Combined_Text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=50, kde=True)
plt.title("Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig('output/Text_Length_Distribution.png')
#plt.show()

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
device = torch.device('cuda' if gpu_available else 'cpu')
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
X_train_encoded = encode_texts(X_train)
X_test_encoded = encode_texts(X_test)
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
    #plt.show()


# 可视化BERT编码后的特征（使用PCA）
visualize_features(X_train_encoded, y_train, method='pca')

# 可视化BERT编码后的特征（使用t-SNE）
visualize_features(X_train_encoded, y_train, method='tsne')

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
    print(f"{model_name} Original Classification Report:\n",
          classification_report(y_test, y_pred, target_names=label_encoder.classes_))

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
# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train)

# 显示SMOTE处理后的类别分布
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_smote)
plt.title("Class Distribution in Training Set after SMOTE")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig('output/Class_Distribution_Train_SMOTE.png')
#plt.show()

print("after smote sample")
print('0:', y_train_smote.value_counts()[0], '/', round(y_train_smote.value_counts()[0] / len(y_train_smote) * 100, 2),
      '% of the dataset')
print('1:', y_train_smote.value_counts()[1], '/', round(y_train_smote.value_counts()[1] / len(y_train_smote) * 100, 2),
      '% of the dataset')
print('2:', y_train_smote.value_counts()[2], '/', round(y_train_smote.value_counts()[2] / len(y_train_smote) * 100, 2),
      '% of the dataset')
print('3:', y_train_smote.value_counts()[3], '/', round(y_train_smote.value_counts()[3] / len(y_train_smote) * 100, 2),
      '% of the dataset')

# Hyperparameter grids
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_lr = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.1, 1.0, 10.0],
    'solver': ['saga']
}

# Models with GridSearchCV
models = {
    'Random Forest': GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1),
    'Decision Tree': GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=3, n_jobs=-1),
    'Logistic Regression': GridSearchCV(LogisticRegression(max_iter=10000, random_state=42), param_grid_lr, cv=3, n_jobs=-1)
}

print("begin train model: " + str(datetime.now()))

results_smote = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_encoded)
    y_proba = model.predict_proba(X_test_encoded)

    results_smote[model_name] = {
        'best_params': model.best_params_,
        'accuracy': model.score(X_test_encoded, y_test),
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_proba': y_proba,
        'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
    }

    # Print best parameters
    print(f"Best parameters for {model_name}: {model.best_params_}")
    # Print classification report
    print(f"Classification report for {model_name}:\n{results_smote[model_name]['classification_report']}")

# Visualize results for the best model
for model_name, result in results_smote.items():
    # ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test, result['y_proba'][:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i, color in zip(range(len(label_encoder.classes_)), sns.color_palette("viridis", len(label_encoder.classes_))):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
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
    cm = result['confusion_matrix']
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
plt.savefig(f'output/{best_model_name}_best_confusion_matrix.png')  # Save the confusion matrix as an image file
plt.close()  # Close the plot to avoid display

print("end: " + str(datetime.now()))
