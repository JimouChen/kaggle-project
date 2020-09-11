"""
# @Time    :  2020/9/11
# @Author  :  Jimou Chen
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn  # 画热力图


# 定义一个相关性的热力图，更加直观地判断
def heat_map(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(data.shape[0], data.shape[1]))  # 尺寸大小与data一样
    correlation_mat = data.corr()
    sns.heatmap(correlation_mat, annot=True, cbar=True, square=True, fmt='.2f', annot_kws={'size': 10})
    plt.show()


def draw_heat_map(the_data):
    # 画热力图
    plt.figure(figsize=(20, 20))
    p = seaborn.heatmap(the_data.corr(), annot=True, annot_kws={'fontsize': 15}, square=True)
    plt.show()


def handle_in_data():
    data = pd.read_excel('in_data.xlsx')

    data = data.iloc[:, 1:]
    print(type(data))
    # 数据标准化
    # sc = StandardScaler()
    # data = sc.fit_transform(data)
    # print(data)
    # data = pd.DataFrame(data)

    # print('相关系数矩阵：\n', np.round(data.corr(method='pearson'), 3))
    draw_heat_map(data)

    # 多元线性回归模型
    # model = LinearRegression()

    # y_data = data.iloc[:, -1]

    # # 建模
    # model = KMeans(n_clusters=4)
    # model.fit(data)
    # print('center points: ', model.cluster_centers_)
    # # 打印系数,有几个自变量打印出来就有几个系数
    # print('系数：', model.coef_)
    # # 打印截距
    # print('截距：', model.intercept_)
    # print('相关系数矩阵：\n', np.round(data.corr(method='pearson'), 3))
    # # 标准化数据
    # sc = StandardScaler()
    # data = sc.fit_transform(data)
    # # draw_heat_map(data)
    # print('相关系数矩阵：\n', np.round(data.corr(method='pearson'), 3))

    #
    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)  # 点为红色三角形
    # x0 = x_data[:, 0]
    # x1 = x_data[:, 1]
    # # 生成网格矩阵
    # x0, x1 = np.meshgrid(x0, x1)
    # z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]
    # # 画3D图
    # ax.plot_surface(x0, x1, z)
    # # 设置坐标轴
    # ax.set_xlabel('税额')
    # ax.set_ylabel('合计')
    # ax.set_zlabel('发票状态')
    #
    # # 显示图像
    # plt.show()


if __name__ == '__main__':
    handle_in_data()
