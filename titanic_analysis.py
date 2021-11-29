import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import graphviz

# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# 模块1.数据探索
print(train_data.info())
print('-' * 30)
print(train_data.describe())
print('-' * 30)
print(train_data.describe(include=['O']))
print('-' * 30)
print(train_data.head())
print('-' * 30)
print(train_data.tail())

# 模块2.数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# 模块3.特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 模块4.构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
print(clf.__dict__)
# 决策树训练
clf.fit(train_features, train_labels)

test_features = dvec.transform(test_features.to_dict(orient='record'))
# 模块5.决策树预测&评估
# 将(包含预测结果)的测试集写入csv文件
pred_labels = clf.predict(test_features)
test_data['Survived'] = pred_labels
test_data.to_csv('test.csv')
# 使用K折交叉验证 统计决策树准确率
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

# 模块6.决策树可视化
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.view()
