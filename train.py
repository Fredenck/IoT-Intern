import cv2
import numpy as np
import os
import time
import _pickle as cPickle

from skimage import feature
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC



def process(im):
    # blur = cv2.GaussianBlur(im, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1
    # brightness = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    # resized = cv2.resize(brightness, (224, 224))
    # imc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)

    imc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im)

    adapT = cv2.adaptiveThreshold(imc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
    median = cv2.medianBlur(adapT, 5)

    return median

start_time = time.time()
labels = []
data = []
eps = 1e-7
numPoints = 24
bookmark = 0
for dirname, _, filenames in os.walk('input\\retinaVein'):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        print(imagePath)
        # load the image, convert it to grayscale, and describe it
        im = cv2.imread(imagePath, 0)
        processed = process(im)

        image8bit = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        lbp = feature.local_binary_pattern(image8bit, 24, 8, method="uniform")
        (desc, _) = np.histogram(lbp.ravel(),
                                bins=np.arange(0, numPoints + 3),
                                range=(0, numPoints + 2))
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        data.append(desc)
        label = imagePath.split(os.path.sep)[-2]
        # labels.append(label)
        if (label == "No_DR"):
            labels.append(label)
        else:
            labels.append("DR")
        bookmark += 1
    #     if (bookmark == 10) :
    #         break;
    # if (bookmark == 10):
    #     break;
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.8, random_state=42)

model = LinearSVC(C=100.0, max_iter=10000, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

f = open("models/retinaVeinData.cpickle", "wb")
f.write(cPickle.dumps(data))
f.close()
f = open("models/retinaVeinLabels.cpickle", "wb")
f.write(cPickle.dumps(labels))
f.close()
f = open("models/retinaVein.cpickle", "wb")
f.write(cPickle.dumps(model))
f.close()

print(time.time() - start_time)
