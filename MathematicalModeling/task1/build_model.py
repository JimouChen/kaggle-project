import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


# 使用pca处理
def handle_new_in_ticket():
    data = pd.read_csv('in_out_data.csv')
    x_data = data.iloc[:, 3:-1]
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


def LogicRegression_add_PCA():
    data = pd.read_csv('in_out_data.csv')
    x_data = data.iloc[:, 2:4]
    y_data = data.iloc[:, -1]
    # 数据标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    # 降2维后用其他模型预测
    # pca = PCA(n_components=2)
    # x_data = pca.fit_transform(x_data)
    # new_x_data = new_data[:, 0]
    # new_y_data = new_data[:, 1]

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    print(model.score(x_test, y_test))
    # print(classification_report(model.predict(x_test), y_test))
    print('系数: \n', model.coef_)
    print('截距\n', model.intercept_)

    print(classification_report(model.predict(x_test), y_test))
    return model
    # x_data1 = x_data[:, 0]
    # x_data2 = x_data[:, 1]
    #
    # plt.scatter(x_data1, x_data2, c=y_data, s=2)
    # plt.show()


def de_tree():
    data = pd.read_csv('in_out_data.csv')
    x_data = data.iloc[:, 2:4]
    y_data = data.iloc[:, -1]
    # 数据标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5)

    d_tree = DecisionTreeClassifier()
    d_tree.fit(x_train, y_train)
    print(d_tree.score(x_test, y_test))
    print(d_tree.get_depth())

    return d_tree


def rf_model():
    data = pd.read_csv('in_out_data.csv')
    x_data = data.iloc[:, 3:-1]
    y_data = data.iloc[:, -1]
    # 数据标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    # 降3维后用其他模型预测
    # pca = PCA(n_components=3)
    # x_data = pca.fit_transform(x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    rf = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf.fit(x_train, y_train)
    score = rf.score(x_test, y_test)
    print(score)
    return rf


def bagging_model():
    data = pd.read_csv('in_out_data.csv')
    x_data = data.iloc[:, 3:-1]
    y_data = data.iloc[:, -1]
    # 数据标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    '''建决策树模型'''
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    '''加入决策树的集成学习'''
    bagging_tree = BaggingClassifier(tree, n_estimators=100)
    bagging_tree.fit(x_train, y_train)
    print(bagging_tree.score(x_test, y_test))


def calculate(a, b):
    coef = [[-0.16235911, - 0.06396034]]
    inter = [-3.47714514]
    # 得到具体公式
    g = coef[0][0] * a + coef[0][1] * b + inter[0]
    L = 1 / (1 + pow(np.e, -g))
    return L


def detail_cal_LR():
    data = pd.read_csv('in_out_data.csv')
    L_value = []
    feature_data = data.iloc[:, 2:4]
    # 数据标准化
    sc = StandardScaler()
    feature_data = sc.fit_transform(feature_data)
    # print(feature_data)
    for i in feature_data:
        a = i[0]
        b = i[1]
        L_value.append(calculate(a, b))
    print(L_value)


def handle_file2():
    train_data = pd.read_csv('in_out_data.csv')
    x_train = train_data.iloc[:, 2:4]
    y_train = train_data.iloc[:, -1]

    data = pd.read_csv('file2_data.csv')
    feature = data.iloc[:, 2:]
    # 数据标准化
    sc = StandardScaler()
    test_feature = sc.fit_transform(feature)
    test_feature = pd.DataFrame(test_feature)
    x_train = sc.fit_transform(x_train)

    # 用题一的模型
    model = de_tree()
    model.fit(x_train, y_train)
    # 预测
    prediction = model.predict(test_feature)
    print(prediction)
    c = 0
    d = 0
    for i in prediction:
        if i == 0:
            c += 1
        else:
            d += 1
    print(c, d)

    # pred_df = {'是否违约': prediction}
    # pred_df = pd.DataFrame(pred_df)
    # 合并列
    data['是否违约'] = prediction
    # 保存data
    save = pd.DataFrame(data)
    save.to_csv('new_file2.csv', encoding='utf_8_sig')
    print('save finished')


# 预测是否违约
def pred_file2():
    new_data = pd.read_csv('new_file2.csv')
    lay = []
    for i in range(0, 302):
        lay.append(0)
    num = list(range(124, 426))
    c = 0
    for j in num:
        for i in new_data.企业代号:
            if i == j and new_data.loc[c, '是否违约'] == 1:
                lay[j - 124] = 1
                break
            else:
                continue
        c += 1

    for i in range(len(lay)):
        if i in list(range(0, 302)):
            lay[i] = 1


def pred_level():
    new_file2 = pd.read_csv('pred_file2.csv')
    # 最后把违约的设为D
    lay = list(new_file2.iloc[:, -1])
    print(lay)
    data = pd.read_csv('new_file2.csv')
    x_data = data.iloc[:, 3:]
    # 数据标准化
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
    print(x_data)
    # 建模
    model = KMeans(n_clusters=4)
    model.fit_transform(x_data)
    # 预测
    prediction = model.predict(x_data)
    print('预测评级:\n', prediction)
    # 保存
    data['预测评级'] = prediction
    # 保存data
    save = pd.DataFrame(data)
    save.to_csv('level_file2.csv', encoding='utf_8_sig')
    print('save finished')

    a = 0
    b = 0
    c = 0
    d = 0
    for i in prediction:
        if i == 0:
            a += 1
        elif i == 1:
            b += 1
        elif i == 2:
            c += 1
        elif i == 3:
            d += 1

    print('a,b,c,d\n', a, b, c, d)

    center = model.cluster_centers_
    print(center)
    colors = ['r', 'b', 'y', 'g']
    # 画出
    new_x_data = center[:, 0]
    new_y_data = center[:, 1]
    new_z_data = center[:, 2]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(new_x_data, new_y_data, new_z_data, c=colors, s=30)
    plt.show()


def add_pred_to_file():
    data = pd.read_csv('level_file2.csv')
    print(data)
    c = 0
    level_list = []
    num_list = []
    for i in range(0, data.shape[0]):
        level_list.append(data.loc[i, '预测评级'])
        num_list.append(data.loc[i, '企业代号'])

    sum_list = []
    count = 0
    start = 124
    for each in level_list:
        if each == start:
            count += each
        sum_list.append(count)
        start += 1
        count = 0


if __name__ == '__main__':
    # handle_new_in_ticket()
    # LogicRegression_add_PCA()  # 0.96
    # de_tree()  # 0.79
    # rf_model()  # 0.77
    # test_data()
    # bagging_model()  # 0.79
    # handle_file2()
    # pred_file2()
    # pred_level()
    add_pred_to_file()
