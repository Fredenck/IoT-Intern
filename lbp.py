from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os

image = cv2.imread("lenna.png", 0)
numPoints = 24
radius = 8
eps = 1e-7

lbp = feature.local_binary_pattern(image, numPoints,
                                   radius, method="uniform")
(desc, _) = np.histogram(lbp.ravel(),
                         bins=np.arange(0, numPoints + 3),
                         range=(0, numPoints + 2))
# normalize the histogram
desc = desc.astype("float")
desc /= (desc.sum() + eps)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images("\\input\\training"):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

# loop over the testing images
for imagePath in paths.list_images("\\input\\testing"):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))

    # display the image and the prediction
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
