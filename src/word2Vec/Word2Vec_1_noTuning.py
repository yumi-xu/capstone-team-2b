from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import numpy as np

print("begin: " + str(datetime.now()))
# Load the dataset
data = pd.read_csv('../csv/incidents.all.csv')

# 检查并显示缺失 `Severity` 数据的行数
missing_severity_count = data['Severity'].isnull().sum()
print(f"\nNumber of rows with missing Severity: {missing_severity_count}")

# 丢弃缺失 `Severity` 数据的行
data = data.dropna(subset=['Severity'])
print(f"\nDataset after dropping rows with missing Severity: {len(data)} rows remaining")

print("begin word2Vec: " + str(datetime.now()))
# Combine relevant text columns into one
text_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders',
                'Harm Types', 'Summary', 'Evidences', 'Concepts', 'Country']
data['combined_text'] = data[text_columns].fillna('').astype(str).agg(' '.join, axis=1)

# Preprocess text data and tokenize
data['tokenized_text'] = data['combined_text'].apply(gensim.utils.simple_preprocess)

# 打印标记化后的数据样本
print(data['tokenized_text'].head())

# Train a Word2Vec model
w2v_model = Word2Vec(sentences=data['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)
#w2v_model.save("word2vec.model")

# 打印模型的一些信息
print(w2v_model)

# Vectorize the text using Word2Vec
def vectorize_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

data['vectorized_text'] = data['tokenized_text'].apply(lambda x: vectorize_text(x, w2v_model))

print("after word2Vec: " + str(datetime.now()))

# 打印向量化后的数据样本
print(data['vectorized_text'].head())

# Prepare feature matrix and labels
X = np.vstack(data['vectorized_text'].values)
y = data['Severity']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models on the original data (without SMOTE)
print("Training models on original data...")
original_results = {}

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    original_results[name] = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

print("Original data training complete.")

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

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("after smote sample")
print('0:', y_train_res.value_counts()[0], '/', round(y_train_res.value_counts()[0] / len(y_train_res) * 100, 2),
      '% of the dataset')
print('1:', y_train_res.value_counts()[1], '/', round(y_train_res.value_counts()[1] / len(y_train_res) * 100, 2),
      '% of the dataset')
print('2:', y_train_res.value_counts()[2], '/', round(y_train_res.value_counts()[2] / len(y_train_res) * 100, 2),
      '% of the dataset')
print('3:', y_train_res.value_counts()[3], '/', round(y_train_res.value_counts()[3] / len(y_train_res) * 100, 2),
      '% of the dataset')

print("begin train model: " + str(datetime.now()))
# Train and evaluate models on the SMOTE data
print("Training models on SMOTE data...")
smote_results = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    smote_results[name] = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

print("SMOTE data training complete.")

# Visualization of results
def plot_confusion_matrices(results, title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(title)

    for ax, (name, result) in zip(axes, results.items()):
        cm = result["confusion_matrix"]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.xaxis.set_ticklabels(['Low', 'Medium', 'High'])
        ax.yaxis.set_ticklabels(['Low', 'Medium', 'High'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes)
    plt.show()


print("Plotting results...")
plot_confusion_matrices(original_results, "Confusion Matrices on Original Data")
plot_confusion_matrices(smote_results, "Confusion Matrices on SMOTE Data")


# Display classification reports
def print_classification_reports(results, title):
    print(title)
    for name, result in results.items():
        print(f"Model: {name}")
        print(pd.DataFrame(result["classification_report"]).transpose())
        print("\n")


print_classification_reports(original_results, "Classification Reports on Original Data")
print_classification_reports(smote_results, "Classification Reports on SMOTE Data")

print("end: " + str(datetime.now()))