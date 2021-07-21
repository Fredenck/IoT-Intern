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

# for the smallerRetinaVein:
classifiers = [
    KNeighborsClassifier(3), # 93.6
    # SVC(kernel="linear", C=0.025), # 50.7
    # GaussianProcessClassifier(1.0 * RBF(1.0)), #91.6 takes much longer than others
    DecisionTreeClassifier(max_depth=5), #90.6
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), # 85.9
    MLPClassifier(alpha=1, max_iter=1000), #64.8
    AdaBoostClassifier(), #89.8
    GaussianNB(), #82.03
    QuadraticDiscriminantAnalysis()] #89.1; variables are colinear warning
    # for Quadratic Discrim:
# precision: [0.87231869 0.91255869]
# recall: [0.91976306 0.86149584]
# fscore: [0.89541284 0.88629239]
# support: [1857 1805]


start_time = time.time()

# data = cPickle.loads(open("models/smallerRetinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/smallerRetinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/smallerRetinaVein.cpickle", "rb").read())

data = cPickle.loads(open("models/retinaVeinData.cpickle", "rb").read())
labels = cPickle.loads(open("models/retinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/retinaVein.cpickle", "rb").read())



# data = cPickle.loads(open("models/retinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/retinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/smallerRetinaVein.cpickle", "rb").read())

# data = cPickle.loads(open("models/smallerRetinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/smallerRetinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/retinaVein.cpickle", "rb").read())

for c in classifiers:
    print("A")
    model = c.fit(data, labels)
    predictions = model.predict(data)
    score = accuracy_score(labels, predictions)
    print(score)

    print((f1_score(labels, predictions, average="macro")))
    print((precision_score(labels, predictions, average="macro")))
    print((recall_score(labels, predictions, average="macro")))

    precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

print(time.time() - start_time)
