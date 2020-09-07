"""
# @Time    :  2020/9/7
# @Author  :  Jimou Chen
"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':

    data = pd.read_csv('wine_data.csv')
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0]

    # 切分数据
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

    # 标准化数据
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # 建模
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    model.fit(x_train, y_train)

    # 评估
    prediction = model.predict(x_test)
    print(classification_report(prediction, y_test))
    print(confusion_matrix(y_test, prediction))
