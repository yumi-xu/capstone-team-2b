from datetime import datetime
# section 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from serialization_utils import serialize_to_file
import os
import plotly.express as px

top_feature_num = 5

file_name = os.path.basename(__file__)
distDir = f'dist/{file_name[:-3]}'
os.makedirs(distDir, exist_ok=True)

tmpDir = 'tmp/version6_all'
os.makedirs(tmpDir, exist_ok=True)

# Step 1: Data Cleaning
# 加载数据
print("begin step 1: " + str(datetime.now()))
df = pd.read_csv('../csv/incidents.all.csv')
print("Original Dataset:")
# print(df.head())

# 检查并显示缺失 `Severity` 数据的行数
missing_severity_count = df['Severity'].isnull().sum()
print(f"\nNumber of rows with missing Severity: {missing_severity_count}")

# 绘制缺失 `Severity` 数据的图表
plt.figure(figsize=(6, 4))
plt.bar(['Missing Severity', 'Non-missing Severity'], [missing_severity_count, len(df) - missing_severity_count])
plt.ylabel('Number of Rows')
plt.title('Missing Severity Data')
plt.savefig(f'{distDir}/missing_severity_data.png')

# 丢弃缺失 `Severity` 数据的行
df = df.dropna(subset=['Severity'])
print(f"\nDataset after dropping rows with missing Severity: {len(df)} rows remaining")

print("\nColumns in the Dataset:")
print(df.columns)

# section 2

# Select relevant columns
relevant_columns = ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Severity', 'Summary',
                    'Evidences', 'Concepts']
df = df[relevant_columns]

# Remove rows where 'Severity' is empty
df = df.dropna(subset=['Severity'])

print("Data after cleaning:")
print(df.head())

labels = ["Non-physical harm", "Hazard", "Death", "Injury"]

harm_type = df["Severity"].value_counts().tolist()
print(harm_type)
values = [harm_type[0], harm_type[1], harm_type[2], harm_type[3]]

fig = px.pie(values=df['Severity'].value_counts(), names=labels, width=500, height=400,
             color_discrete_sequence=["skyblue", "black", "red", "orange"]
             , title="AI incident Serverities")

fig.write_image(f'{distDir}/ai_incident_serverities.png')

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
for column in ['Title', 'Principles', 'Industries', 'Affected Stakeholders', 'Harm Types', 'Summary', 'Evidences',
               'Concepts']:
    tfidf_df = text_to_tfidf(df, column)
    tfidf_features_df = pd.concat([tfidf_features_df, tfidf_df], axis=1)
    column_names.extend([column] * tfidf_df.shape[1])

print("TF-IDF features dataframe shape:", tfidf_features_df.shape)
print("begin step 3: " + str(datetime.now()))
# Step 3: Mutual Information Calculation
# Calculate mutual information scores for each feature
X = tfidf_features_df
y = df['Severity']
print("y:")
print(y)
# Ensure y is a category type
y = y.astype('category')
y_encoded = y.cat.codes
print("y_encoded:")
print(y_encoded)
y_cat = y.cat.categories
print("y_cat:")
print(y_cat)

mi_scores = mutual_info_classif(X, y_encoded)
print("begin step 4: " + str(datetime.now()))
# Step 4: Feature Importance Ranking
# Aggregate importance scores for each original column
feature_importance_df = pd.DataFrame({'Feature': column_names, 'Importance': mi_scores})
aggregated_importance = feature_importance_df.groupby('Feature').sum().reset_index()
aggregated_importance = aggregated_importance.sort_values(by='Importance', ascending=False)

top_aggregated_importance = aggregated_importance.head(top_feature_num)

print(f"Top {top_feature_num} important features:")
print(top_aggregated_importance)
print("begin step 5: " + str(datetime.now()))
# Step 5: Visualization
# Plot the top {top_feature_num} most important features
plt.figure(figsize=(12, 8))
plt.barh(aggregated_importance['Feature'].head(top_feature_num),
         aggregated_importance['Importance'].head(top_feature_num))
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title(f"Top {top_feature_num} Important Features")
plt.gca().invert_yaxis()
plt.savefig(f'{distDir}/top_{top_feature_num}_important_features.png')
print("end step 5: " + str(datetime.now()))

serialize_to_file(tfidf_features_df, f'{tmpDir}/tfidf_features_df.pkl')
serialize_to_file(y, f'{tmpDir}/y.pkl')
serialize_to_file(y_cat, f'{tmpDir}/y_cat.pkl')
serialize_to_file(y_encoded, f'{tmpDir}/y_encoded.pkl')
serialize_to_file(top_aggregated_importance, f'{tmpDir}/top_aggregated_importance.pkl')
