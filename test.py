import cv2
import numpy as np
import os
import time
import _pickle as cPickle

from skimage import feature
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

start_time = time.time()

# data = cPickle.loads(open("models/smallerRetinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/smallerRetinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/smallerRetinaVein.cpickle", "rb").read())

data = cPickle.loads(open("models/retinaVeinData.cpickle", "rb").read())
labels = cPickle.loads(open("models/retinaVeinLabels.cpickle", "rb").read())
model = cPickle.loads(open("models/smallerRetinaVein.cpickle", "rb").read())

# data = cPickle.loads(open("models/smallerRetinaVeinData.cpickle", "rb").read())
# labels = cPickle.loads(open("models/smallerRetinaVeinLabels.cpickle", "rb").read())
# model = cPickle.loads(open("models/retinaVein.cpickle", "rb").read())

predictions = model.predict(data)
print((f1_score(labels, predictions, average="micro")))
print((precision_score(labels, predictions, average="macro")))
print((recall_score(labels, predictions, average="weighted")))

precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print(time.time() - start_time)
