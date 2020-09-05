"""
# @Time    :  2020/9/5
# @Author  :  Jimou Chen
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# train_data = pd.read_csv('data/Churn-Modelling.csv')
# test_data = pd.read_csv('data/Churn-Modelling-Test-Data.csv')


def deal_train(path):
    train_data = pd.read_csv(path)
    # 处理国家转换为数字
    train_data.loc[train_data['Geography'] == 'France', 'Geography'] = 1
    train_data.loc[train_data['Geography'] == 'Spain', 'Geography'] = 2
    train_data.loc[train_data['Geography'] == 'Germany', 'Geography'] = 3
    # 处理性别
    train_data.loc[train_data['Gender'] == 'Female', 'Gender'] = 0
    train_data.loc[train_data['Gender'] == 'Male', 'Gender'] = 1

    # 选取有用的特征
    feature = ['CreditScore', 'Geography', 'Gender',
               'Age', 'Tenure', 'Balance', 'NumOfProducts',
               'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    x_data = train_data[feature]
    y_data = train_data['Exited']

    # 对数据进行标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)

    return x_data, y_data


if __name__ == '__main__':
    x_train_data, y_train_data = deal_train('data/Churn-Modelling.csv')
    x_test, y_test = deal_train('data/Churn-Modelling-Test-Data.csv')

    # 建模,可以多试试其他模型
    lr = LogisticRegression()
    lr.fit(x_train_data, y_train_data)

    # 预测
    pred = lr.predict(x_test)
    print(classification_report(pred, y_test))
