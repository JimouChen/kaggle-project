"""
# @Time    :  2020/9/15
# @Author  :  Jimou Chen
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

warnings.filterwarnings("ignore")


def draw_heat_map(df):
    # 画热力图，数值为两个变量之间的相关系数
    plt.figure(figsize=(20, 20))
    p = sns.heatmap(df.corr(), annot=True, square=True)
    plt.show()


def handle_data():
    data = pd.read_csv('data.csv')
    # 删除无用的列
    data = data.drop('id', axis=1)
    # 特征数值化
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # 查看标签分布
    print(data.diagnosis.value_counts())
    # 画热力图
    draw_heat_map(data)
    # 画出标签统计
    data.diagnosis.value_counts().plot(kind='bar')
    plt.show()

    # 切分数据
    x_data = data.drop('diagnosis', axis=1)
    y_data = data['diagnosis']

    return x_data, y_data


if __name__ == '__main__':
    x_data, y_data = handle_data()
    from sklearn.model_selection import train_test_split

    # 切分数据集，stratify=y表示切分后训练集和测试集中的数据类型的比例跟切分前y中的比例一致
    # 比如切分前y中0和1的比例为1:2，切分后y_train和y_test中0和1的比例也都是1:2
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, stratify=y_data)

    classifiers = [
        KNeighborsClassifier(3),
        LogisticRegression(),
        MLPClassifier(hidden_layer_sizes=(20, 50), max_iter=10000),
        DecisionTreeClassifier(),
        RandomForestClassifier(max_depth=9, min_samples_split=3),
        AdaBoostClassifier(),
        BaggingClassifier(),
    ]

    log = []
    for clf in classifiers:
        clf.fit(x_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        test_predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, test_predictions)
        print("Accuracy: {:.4%}".format(acc))

        log.append([name, acc * 100])

    print("=" * 30)

    log = pd.DataFrame(log)
    print(log)

    log.rename(columns={0: 'Classifier', 1: 'Accuracy'}, inplace=True)

    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()
