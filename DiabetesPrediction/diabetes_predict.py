"""
# @Time    :  2020/9/6
# @Author  :  Jimou Chen
"""
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import missingno as msn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def label_distribution(data):
    p = data.Outcome.value_counts().plot(kind='bar')  # 使用柱状图画出
    plt.show()
    # 可视化数据发布, 有些数据本不该为0的却为0，其实是空的
    p = seaborn.pairplot(data, hue='Outcome')
    plt.show()
    # 把空值的用柱状图画出来
    p = msn.bar(data)
    plt.show()


def handle_data():
    data = pd.read_csv('data/diabetes.csv')
    # 查看标签分布
    print(data.Outcome.value_counts())
    # 把葡萄糖，血压，皮肤厚度，胰岛素，身体质量指数中的0替换为nan
    handle_col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[handle_col] = data[handle_col].replace(0, np.nan)

    # 设定阀值
    thresh_count = data.shape[0] * 0.8
    # 若某一列数据缺失的数量超过20%就会被删除
    data = data.dropna(thresh=thresh_count, axis=1)

    # 填充数据，得到新的数据集data
    data['Glucose'] = data['Glucose'].fillna(data['Glucose'].mean())
    data['BloodPressure'] = data['BloodPressure'].fillna(data['BloodPressure'].mean())
    data['BMI'] = data['BMI'].fillna(data['BMI'].mean())

    return data


if __name__ == '__main__':
    new_data = handle_data()
    # label_distribution(new_data)

    # 切分数据集
    x_data = new_data.drop('Outcome', axis=1)
    y_data = new_data.Outcome
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, stratify=y_data)

    # 建模
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # 预测
    pred = model.predict(x_test)
    # 评估
    print(classification_report(pred, y_test))