import pandas as pd

def clean_csv(input_csv, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 过滤掉"Severity"字段为空值的行
    filtered_df = df[df['Severity'].notna()]

    # 保存过滤后的数据到新的CSV文件
    filtered_df.to_csv(output_csv, index=False)


# 示例使用
input_csv = 'csv/incidents.all.csv'
output_csv = 'csv/incidents.clean.csv'
n = 100  # 设定你需要随机抽取的行数

clean_csv(input_csv, output_csv)