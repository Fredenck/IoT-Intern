from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import _pickle as cPickle
# model = cPickle.loads(open("model.cpickle").read())

def ill(im):
    img = im.copy()
    rows, cols = img.shape
    maax = max(map(max, img))
    miin = min(map(min, img))
    for i in range(rows):
        for j in range(cols):
            img[i][j] = ((img[i][j]-miin)*255) / (maax-miin)
    return img

def process(image):
    # III: PRE-PROCESSING
    # A: Noise Removal (typical blur)
    blur = cv2.GaussianBlur(original, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1

    # B: Dispelling Illumination (transform bright/dim)
    # brightness = cv2.normalize(blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    brightness = ill(blur)

    # C: Normalize (resize + center)
    resized = cv2.resize(brightness, (240, 180))

    ret, thresh = cv2.threshold(resized, 90, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    top_pos = y;
    bottom_pos = y + h;
    left_pos = x;
    right_pos = x + w

    cropped = resized.copy()
    cropped = cropped[top_pos:bottom_pos][left_pos:right_pos]

    # clah = cv2.createCLAHE(tileGridSize=(15,15)).apply(cropped)

    # IV: Vein Extraction
    # A: Adaptive Threshold
    adapThresh = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)

    # B: Median Filtering
    median = cv2.medianBlur(adapThresh, 5)

    # C: Massive Noise Removal
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(median, connectivity=8)  # need to fix
    massRem = np.zeros(labels.shape)
    sizes = stats[1:, -1]

    for i in range(num_labels - 1):
        if sizes[i] >= 100:
            massRem[labels == i + 1] = 255
    # for i in range(1, num_labels): #0 is background, 1+ is components
    #     area = stats[i][4]
    #     if area < 400:
    return ~massRem.astype(int)

numPoints = 24
radius = 8
eps = 1e-7
data = []
labels = []

# loop over the trainingCR images

for dirname, _, filenames in os.walk('input\\veinDB'):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        if imagePath.split(os.path.sep)[-1][7:9] == "db": #Ignoring thumbs.db
            continue
        # if imagePath.split(os.path.sep)[-1] != "index_1.bmp":
        #     continue
        if imagePath.split(os.path.sep)[-3] == "082":
            break
        print(imagePath)
        # print(imagePath.split(os.path.sep)[-1][7:9])
        # load the image, convert it to grayscale, and describe it
        original = cv2.imread(imagePath, 0)

        processed = process(original)

        # describe
        lbp = feature.local_binary_pattern(processed, numPoints, radius, method="uniform")
        (desc, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        # extract the label from the image path, then update the
        # label and data lists
        # print(imagePath.split(os.path.sep)[-2])
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(desc)
    else:
        continue
    break

# train a Linear SVM on the data
model = LinearSVC(C=100.0, max_iter=10000, random_state=42)
model.fit(data, labels)

f = open("model.cpickle", "w")
f.write(cPickle.dumps(model))
f.close()

model = cPickle.loads(open("model.cpickle").read())

# loop over the testing images
# for dirname, _, filenames in os.walk('input/veinTesting'):
bookmark = False
correct = 0
total = 0
for dirname, _, filenames in os.walk('input\\veinDB'):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        if imagePath.split(os.path.sep)[-1][7:9] == "db": #Ignoring thumbs.db
            continue
        if imagePath.split(os.path.sep)[-3] == "082":
            bookmark = True
            continue
        if not bookmark:
            continue
        # load the image, convert it to grayscale, describe it,
        # and classify it
        original = cv2.imread(imagePath, 0)

        processed = process(original)

        # describe
        lbp = feature.local_binary_pattern(processed, numPoints,
                                           radius, method="uniform")
        (desc, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        prediction = model.predict(desc.reshape(1, -1))

        print(imagePath)
        print(prediction[0] + ": " + imagePath.split(os.path.sep)[-2])
        if prediction[0] == imagePath.split(os.path.sep)[-2]:
            correct += 1
        total += 1
        # display the image and the prediction
        cv2.putText(original, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 3)
        # cv2.imshow("Image", original)
        # cv2.waitKey(0)
print(correct)
print(total)
print(correct/total)