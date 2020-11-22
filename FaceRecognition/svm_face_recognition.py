"""
# @Time    :  2020/11/22
# @Author  :  Jimou Chen
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people  # 使用LFW的人脸数据集
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def build_model(param_grid):
    model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid)

    model.fit(x_train_pca, y_train)
    pred = model.predict(x_test_pca)
    print(classification_report(pred, y_test, target_names=people.target_names))

    print('best params:', model.best_params_)
    print(model.best_estimator_)


# 人脸数据集的各个属性
def look_attributes(data):
    # 查看有多少张照片、高度、宽度,也就是每张图片有h*w个像素点，即h*w个特征
    n, h, w = data.images.shape
    print(n, h, w)
    # 查看类别和类别的名字，即人名
    print(data.target)
    print(data.target_names)
    # 查看有几个人的照片
    print(data.target_names.shape[0])


if __name__ == '__main__':
    # 加载数据，第一次可能下载到本地比较久
    # 只要这个人的照片张数>=min_faces_per_person,就加载进来
    people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    # plt.imshow(people.images[6], cmap='gray')
    # plt.show()
    look_attributes(people)

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(people.data, people.target)

    # 直接不降维建模
    # model = SVC(kernel='rbf', class_weight='balanced')
    # model.fit(x_train, y_train)
    #
    # predictions = model.predict(x_test)
    # # 加上target_names可以显示出标签名字
    # print(classification_report(y_test, predictions, target_names=people.target_names))

    '''下面通过pca降维提高准确率'''
    # 比如把上面h*w个维度降成100个维度(特征)
    n_class = 100
    # 针对于图片，whiten降低输入的冗余性
    pca = PCA(n_components=n_class, whiten=True).fit(people.data)

    # 降维后的训练集和测试集的数据
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    C = [i for i in range(1, 5)]
    gamma = []
    for i in range(1, 10):
        gamma.append(i / 1000)

    param_grid = {
        'C': C,
        'gamma': gamma
    }
    # param_grid = {'C': [0.1, 0.6, 1, 2, 3],
    #               'gamma': [0.003, 0.004, 0.005, 0.006, 0.007]}
    build_model(param_grid)

    # 查看训练集的尺寸,发现已降到100维
    # print(x_train_pca.shape)
    # model = SVC(kernel='rbf', class_weight='balanced')
    # model.fit(x_train_pca, y_train)
    # predictions = model.predict(x_test_pca)
    # print(classification_report(predictions, y_test))
    #
    # '''调参找到最好的model'''
    # param_grid = {'C': [0.1, 0.6, 1, 2, 3],
    #               'gamma': [0.003, 0.004, 0.005, 0.006, 0.007]}
    # model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid)
    # model.fit(x_train_pca, y_train)
    # predictions = model.predict(x_test_pca)
    # print(classification_report(predictions, y_test))
    # # 查看最好的参数
    # print(model.best_estimator_)
    # print(model.best_params_)
