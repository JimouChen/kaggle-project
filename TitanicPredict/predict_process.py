"""
# @Time    :  2020/9/2
# @Author  :  Jimou Chen
"""
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd


# 先处理空缺的数据
def deal_train(train_data):
    # 处理空缺的年龄，设为平均年龄
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    # print(train_data.describe())
    # 处理性别，转化维0和1,loc是取数据的，里面传行，列
    train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 1
    train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 0
    # print(train_data.loc[:, 'Sex'])

    # 处理Embarked，登录港口
    # print(train_data['Embarked'].unique())  # 看一下里面有几类
    # 由于'S'比较多，就把空值用S填充
    train_data['Embarked'] = train_data['Embarked'].fillna('S')
    # 转化为数字
    train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
    train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
    train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2

    '''接下来选取有用的特征'''
    feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_data = train_data[feature]
    y_data = train_data['Survived']  # 预测的标签

    # 数据标准化
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    return x_data, y_data


# 处理测试集数据
def deal_test(test_data, label_data):
    # 填充年龄和Fare
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
    # 处理性别字符串为数值
    test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 1
    test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 0
    # 处理登岸地点为数值
    test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
    test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
    test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2
    # 接下来选取有用的特征'''
    feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_data = test_data[feature]
    y_data = label_data['Survived']

    # 数据标准化
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    return x_data, y_data


if __name__ == '__main__':
    # 读入训练集和测试集
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    real_label_data = pd.read_csv('data/gender_submission.csv')
    # 队训练集和测试集进行处理
    x_train, y_train = deal_train(train_data)
    x_test, y_test = deal_test(test_data, real_label_data)

    # 建立模型
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=4)
    bagging = BaggingClassifier(rf, n_estimators=12)
    bagging.fit(x_train, y_train)

    # 预测
    prediction = bagging.predict(x_test)

    # 评估
    print(bagging.score(x_test, y_test))
    print((classification_report(prediction, y_test)))

    # 保存预测结果为csv
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('predict.csv', index=False)
