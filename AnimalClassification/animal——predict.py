"""
# @Time    :  2020/9/8
# @Author  :  Jimou Chen
"""
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import missingno  # 查看数据缺失情况
import seaborn  # 画热力图


def draw_heat_map(the_data):
    # 画热力图
    plt.figure(figsize=(20, 20))
    p = seaborn.heatmap(the_data.corr(), annot=True, annot_kws={'fontsize': 15}, square=True)
    plt.show()


def draw_data_isnan(the_data):
    # 查看数据缺失情况
    p = missingno.bar(the_data)
    plt.show()


def model_optimization(x_train, y_train, x_test, y_test):
    param_grid = {'max_depth': [5, 10, 15, 20, 25],
                  'min_samples_split': [2, 3, 4, 5, 6],
                  'min_samples_leaf': [1, 2, 3, 4]}
    model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, iid=True)
    model.fit(x_train, y_train)
    print(model.best_estimator_)

    print(model.score(x_test, y_test))


if __name__ == '__main__':
    data = pd.read_csv('zoo.csv')

    '''获取训练数据和标签'''
    # 去除无用的特征
    x_data = data.drop(['animal_name', 'class_type'], axis=1)
    y_data = data['class_type']

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.3, stratify=y_data)

    # 建模
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(x_train, y_train)

    # 预测
    predict = model.predict(x_test)
    # 评估
    print(classification_report(predict, y_test))
    print(model.score(x_test, y_test))

    model_optimization(x_train, y_train, x_test, y_test)
