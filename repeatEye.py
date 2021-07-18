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
    # III: PRE-PROCESSING
    # A: Noise Removal (typical blur)
    blur = cv2.GaussianBlur(im, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1

    # B: Dispelling Illumination (transform bright/dim)
    brightness = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    # C: Normalize (resize + center), 75% of original 320x240
    resized = cv2.resize(brightness, (240, 180))
    imc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    adapT = cv2.adaptiveThreshold(imc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
    median = cv2.medianBlur(adapT, 5)

    return median
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(median, connectivity=8)  # need to fix
    massRem = np.zeros(labels.shape)
    sizes = stats[1:, -1]

    for i in range(num_labels - 1):
        if sizes[i] >= 50:
            massRem[labels == i + 1] = 255
    return massRem


start_time = time.time()
labels = []
data = []
eps = 1e-7
numPoints = 24
bookmark = 0
if input("Train? ") == "Yes":
    for dirname, _, filenames in os.walk('input\\retinaVein'):
        for filename in filenames:
            imagePath = os.path.join(dirname, filename)
            print(imagePath)
            # print(imagePath.split(os.path.sep)[-1][7:9])
            # load the image, convert it to grayscale, and describe it
            im = cv2.imread(imagePath, 0)
            processed = process(im)

            image8bit = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            lbp = feature.local_binary_pattern(image8bit, 24, 8, method="uniform")
            (desc, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, numPoints + 3),
                                     range=(0, numPoints + 2))
            # print(desc)
            # normalize the histogram
            desc = desc.astype("float")
            desc /= (desc.sum() + eps)

            # data.append(desc)
            data.append(desc)
            label = imagePath.split(os.path.sep)[-2]
            if (label == "No_DR"):
                labels.append(label)
            else:
                labels.append("DR")
    f = open("models/retinaVeinData.cpickle", "wb")
    f.write(cPickle.dumps(data))
    f.close()
    f = open("models/retinaVeinLabels.cpickle", "wb")
    f.write(cPickle.dumps(labels))
    f.close()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.8, random_state=42)

    model = LinearSVC(C=100.0, max_iter=10000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    f = open("models/retinaVein.cpickle", "wb")
    f.write(cPickle.dumps(model))
    f.close()
else:
    data = cPickle.loads(open("models/retinaVeinData.cpickle", "rb").read())
    labels = cPickle.loads(open("models/retinaVeinLabels.cpickle", "rb").read())
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.8, random_state=42)
    model = cPickle.loads(open("models/retinaVein.cpickle", "rb").read())
    # model = torch.hub.load_state_dict_from_url(torch.load("input/export.pkl"))
    # model = load_learner(open("input/export.pkl", "rb"))

# loop over the testing images
# for dirname, _, filenames in os.walk('input/veinTesting'):
# bookmark = False
# correct = 0
# total = 0
# for i in range(len(X_test)):
#     prediction = model.predict(X_test[i].reshape(1, -1))
#     print(prediction)
#     print("predict: " + prediction[0] + " and actual: " + y_test[i])
#     if prediction[0] == y_test[i]:
#         correct += 1
#     total += 1

"""
macro:
0.16952224728316587
0.1471050852282837
0.2

micro:
0.7355254261414185
0.7355254261414185
0.7355254261414185

weighted:
0.6234396158670075
0.5409976525005152
0.7355254261414185

0.7041386427529269
0.7041386427529269
0.7041386427529269
C:\cse430\venv2\lib\site-packages\sklearn\metrics\_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
precision: [0.         0.         0.74475843 0.10096427 0.1       ]
recall: [0.         0.         0.94866709 0.31010453 0.00144509]
fscore: [0.         0.         0.83443624 0.15233205 0.002849  ]
support: [ 1938  4228 20669   574   692]
"""
predictions = model.predict(X_test)
print((f1_score(y_test, predictions, average="micro")))
print((precision_score(y_test, predictions, average="macro")))
print((recall_score(y_test, predictions, average="weighted")))

precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

"""
for dirname, _, filenames in os.walk('..\\input\\retinaVein'):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        if imagePath.split(os.path.sep)[-3] == "082":
            bookmark = True
            continue
        if not bookmark:
            continue
        # load the image, convert it to grayscale, describe it,
        # and classify it
        original = cv2.imread(imagePath, 0)

        processed = process(original)
        image8bit = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # llbp = LLBP(image8bit)
        lbp = feature.local_binary_pattern(image8bit, 24, 8, method="uniform")
        (desc, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # print(desc)
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        prediction = model.predict(desc.reshape(1, -1))

        print(imagePath)
        print("predict: " + prediction[0] + " and actual: " + imagePath.split(os.path.sep)[-2])
        if prediction[0] == imagePath.split(os.path.sep)[-2]:
            correct += 1
        total += 1
        # print(model.predict_proba(desc))
"""
# print(correct)
# print(total)
# print(correct / total)
print(time.time() - start_time)
