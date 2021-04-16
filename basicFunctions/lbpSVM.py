'''
Adapted from https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
LBP and LinearSVM
'''
from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
from  matplotlib import pyplot as plt
from imutils import paths
import argparse
import cv2
import os
# import cPickle
# f = open("model.cpickle", "w")
# f.write(cPickle.dumps(model))
# f.close()
# model = cPickle.loads(open("model.cpickle").read())

# image = cv2.imread("lenna.png", 0)
numPoints = 24
radius = 8
eps = 1e-7
data = []
labels = []

# loop over the trainingCR images
for dirname, _, filenames in os.walk('../input/trainingCR'):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        print(imagePath)
        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (240, 180))

        # describe
        lbp = feature.local_binary_pattern(resized, numPoints, radius, method="uniform")
        (desc, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        print(desc)
        plt.plot(desc)
        plt.show()
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        # extract the label from the image path, then update the
        # label and data lists
        print(imagePath.split(os.path.sep)[-2])
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(desc)

# train a Linear SVM on the data
model = LinearSVC(C=100.0, max_iter=10000, random_state=42)
model.fit(data, labels)

# loop over the testingCR images
for dirname, _, filenames in os.walk('../input/testingCR'):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        # load the image, convert it to grayscale, describe it,
        # and classify it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (240, 180))

        # describe
        lbp = feature.local_binary_pattern(resized, numPoints,
                                           radius, method="uniform")
        (desc, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        prediction = model.predict(desc.reshape(1, -1))

        # display the image and the prediction
        cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
