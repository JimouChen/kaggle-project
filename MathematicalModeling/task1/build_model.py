"""
# @Time    :  2020/9/10
# @Author  :  Jimou Chen
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 使用pca处理
def handle_new_in_ticket():
    data = pd.read_csv('new_in_data.csv')
    x_data = data.iloc[:, 2:-1]
    y_data = data.iloc[:, -1]
    pca = PCA(n_components=3)
    new_data = pca.fit_transform(x_data)
    # print(new_data)
    # 画出降维后的数据
    new_x_data = new_data[:, 0]
    new_y_data = new_data[:, 1]
    new_z_data = new_data[:, 2]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(new_x_data, new_y_data, new_z_data, c=y_data, s=10)
    # ax.scatter(new_x_data, new_y_data, new_z_data, c=y_data, s=10)
    plt.show()

    # 画出二维图像
    # 画出降维后的数据
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(x_data)
    new_x_data = new_data[:, 0]
    new_y_data = new_data[:, 1]
    plt.scatter(new_x_data, new_y_data, c=y_data, s=10)
    plt.show()


def other_model_add_PCA():
    data = pd.read_csv('new_in_data.csv')
    x_data = data.iloc[:, 2:-1]
    y_data = data.iloc[:, -1]
    # 数据标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    # 降2维后用其他模型预测
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(x_data)
    # new_x_data = new_data[:, 0]
    # new_y_data = new_data[:, 1]
    # 切分数据集
    x_train, x_test, y_train, y_text = train_test_split(new_data, y_data)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    # print(model.score(x_test, y_text))
    pred = model.predict(new_data)
    new_x_data = new_data[:, 0]
    new_y_data = new_data[:, 1]
    plt.scatter(new_x_data, new_y_data, c=y_data, s=10)
    plt.show()


if __name__ == '__main__':
    # handle_new_in_ticket()
    other_model_add_PCA()
