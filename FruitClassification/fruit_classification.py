"""
# @Time    :  2020/9/6
# @Author  :  Jimou Chen
"""
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def handle_data():
    data = pd.read_csv('fruit_data.csv')
    # 把英文标签编码为数字白标签
    le = LabelEncoder()
    data.iloc[:, 0] = le.fit_transform(data.iloc[:, 0])
    # print(data)
    # 切分数据集
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify=y_data, random_state=20)

    # 为了找一个更好的k值，设置一个测试分数列表
    score_list = []
    # 建模

    for k in range(1, 30):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        score_list.append(model.score(x_test, y_test))

    print(score_list)
    print(max(score_list))
    best_k = score_list.index(max(score_list))
    print(best_k + 1)

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    print(classification_report(prediction, y_test))


if __name__ == '__main__':
    handle_data()
