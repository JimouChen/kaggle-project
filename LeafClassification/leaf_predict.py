"""
# @Time    :  2020/9/8
# @Author  :  Jimou Chen
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train.csv')
# print(data.species.unique())
# 把叶子的类别字符串转成数字形式
labels = LabelEncoder().fit_transform(data.species)
# lb = LabelEncoder().fit(data.species)
# labels = lb.transform(data.species)
# 去掉'species', 'id'的列
data = data.drop(['species', 'id'], axis=1)
# 切分数据集
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, stratify=labels)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
print(tree.score(x_test, y_test))
print(tree.score(x_train, y_train))

'''模型优化'''

# max_depth:树的最大深度
# min_samples_split:内部节点再划分所需最小样本数
# min_samples_leaf:叶子节点最少样本数
param_grid = {'max_depth': [30, 40, 50, 60, 70],
              'min_samples_split': [2, 3, 4, 5, 6],
              'min_samples_leaf': [1, 2, 3, 4]}
# 网格搜索
model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3)
model.fit(x_train, y_train)

print(model.best_estimator_)
print(model.score(x_test, y_test))
print(model.score(x_train, y_train))
