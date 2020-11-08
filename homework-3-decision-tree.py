#!/usr/bin/python
#encoding:utf-8

import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
from IPython.display import Image

def outlook_type(s):
    it = {b'sunny':1, b'overcast':2, b'rainy':3}
    return it[s]
def temperature(s):
    it = {b'hot':1, b'mild':2, b'cool':3}
    return it[s]
def humidity(s):
    it = {b'high':1, b'normal':0}
    return it[s]
def windy(s):
    it = {b'TRUE':1, b'FALSE':0}
    return it[s]

def play_type(s):
    it = {b'yes': 1, b'no': 0}
    return it[s]

play_feature_E = 'outlook', 'temperature', 'humidity', 'windy'
play_class = 'yes', 'no'

data = np.loadtxt("data.txt", delimiter=" ", dtype=str,  converters={0:outlook_type, 1:temperature, 2:humidity, 3:windy,4:play_type})
x, y = np.split(data,(4,),axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=play_feature_E, class_names=play_class,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf('result.pdf')
Image(graph.create_png())
