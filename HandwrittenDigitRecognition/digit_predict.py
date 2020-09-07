"""
# @Time    :  2020/9/7
# @Author  :  Jimou Chen
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler  # 减去平均值再除以方差
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':

    digits_data = load_digits()
    x_data = digits_data.data
    y_data = digits_data.target

    # 对数据进行标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    # 切分数据
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    # 建模
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100)
    model.fit(x_train, y_train)

    # 预测
    prediction = model.predict(x_test)
    # 评估
    print(classification_report(prediction, y_test))
    print(confusion_matrix(y_test, prediction))

