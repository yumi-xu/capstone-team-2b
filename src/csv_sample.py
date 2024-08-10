import pandas as pd

def random_sample_csv(input_csv, n, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 检查n是否超过数据的行数
    if n > len(df):
        raise ValueError("n exceeds the number of rows in the CSV file")

    # 随机抽取n行数据
    sampled_df = df.sample(n)

    # 将抽取的数据写入新的CSV文件
    sampled_df.to_csv(output_csv, index=False)


# 示例使用
n = 10000  # 设定你需要随机抽取的行数
input_csv = 'csv/incidents.clean.csv'
output_csv = f'csv/incidents.{n}.clean.csv'

random_sample_csv(input_csv, n, output_csv)