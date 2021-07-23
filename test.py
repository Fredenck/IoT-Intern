import cv2
import numpy as np
import os
import time
import _pickle as cPickle

from skimage import feature
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


start_time = time.time()

data = cPickle.loads(open("models/smallerRetinaVeinData.cpickle", "rb").read())
labels = cPickle.loads(open("models/smallerRetinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/smallerRetinaVein.cpickle", "rb").read())

# data = cPickle.loads(open("models/retinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/retinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/retinaVein.cpickle", "rb").read())



# data = cPickle.loads(open("models/retinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/retinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/smallerRetinaVein.cpickle", "rb").read())

# data = cPickle.loads(open("models/smallerRetinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/smallerRetinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/retinaVein.cpickle", "rb").read())




# for the smallerRetinaVein:
classifiers = [
    ("knn", KNeighborsClassifier(3)), # 93.6
    ("SVC", SVC(kernel="linear", C=0.025)), # 50.7
    # ("GPC", GaussianProcessClassifier(1.0 * RBF(1.0))), #91.6 takes much longer than others
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)), #90.6
    ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)), # 85.9
    ("MLP, alpha1", MLPClassifier(alpha=1, max_iter=1000)), #64.8
    ("AdaBoost", AdaBoostClassifier()), #89.8
    ("GaussianNB", GaussianNB()), #82.03
    ("MLP hidden layer", MLPClassifier(hidden_layer_sizes=(3), max_iter=10000000)), # 86.2
    ("Quadratic Discrim", QuadraticDiscriminantAnalysis())#89.1; variables are colinear warning
]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)

file = open("results/model-results-smallerRetinaVein.txt", "w")
file.write("Results of different AI models on a mini-set of data for retina veins. "
           "Total 3665 images: 15% for testing. \n")

for t in classifiers:
    c = t[1]
    file.write(t[0] + "\n")
    model = c.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    file.write(str(score) + "\n")

    file.write(str((f1_score(y_test, predictions, average="macro"))) + "\n")
    file.write(str((precision_score(y_test, predictions, average="macro"))) + "\n")
    file.write(str((recall_score(y_test, predictions, average="macro"))) + "\n")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)

    file.write('precision: {}'.format(precision) + "\n")
    file.write('recall: {}'.format(recall) + "\n")
    file.write('fscore: {}'.format(fscore) + "\n")
    file.write('support: {}'.format(support) + "\n")
    file.write("--------------------" + "\n")

file.write("Time Taken: " + str(time.time() - start_time) + "\n")

file.close()
