from skimage import feature
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import cv2
import os
import _pickle as cPickle
import time
start_time = time.time()


def process(image):
    # III: PRE-PROCESSING
    # A: Noise Removal (typical blur)
    blur = cv2.GaussianBlur(original, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1

    # B: Dispelling Illumination (transform bright/dim)
    brightness = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    # C: Normalize (resize + center), 75% of original 320x240
    resized = cv2.resize(brightness, (240, 180))

    ret, thresh = cv2.threshold(resized, 90, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    top_pos = y
    bottom_pos = y + h
    left_pos = x
    right_pos = x + w

    cropped = resized.copy()
    cropped = cropped[top_pos:bottom_pos][left_pos:right_pos]
    cropped = cv2.resize(cropped, (40, 30))
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


def get_pixel(img, center, x, y):
    thresh = 0
    try:
        if img[x][y] >= center:
            thresh = 1
    except:
        pass
    return thresh


def llbp_calculated_pixel(img, x, y):
    center = img[x,y]

    harr = []
    for i in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        harr.append(get_pixel(img, center, x+i, y))
    power_val = [32, 16, 8, 4, 2, 1, 1, 2, 4, 8, 16, 32]
    hval = 0
    for i in range(len(harr)):
        hval += harr[i] * power_val[i]

    varr = []
    for i in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        varr.append(get_pixel(img, center, x, y+i))
    power_val = [32, 16, 8, 4, 2, 1, 1, 2, 4, 8, 16, 32]
    vval = 0
    for i in range(len(harr)):
        vval += harr[i] * power_val[i]

    return (hval**2 + vval**2)**0.5


def LLBP(img):
    height, width = img.shape
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = llbp_calculated_pixel(img, i, j)
    return img_lbp


numPoints = 24
radius = 8
eps = 1e-7
data = []
labels = []

# loop over the trainingCR images
if input("Train? ") == "Yes":
    for dirname, _, filenames in os.walk('input\\veinDB'):
        for filename in filenames:
            imagePath = os.path.join(dirname, filename)
            if imagePath.split(os.path.sep)[-1][7:9] == "db": #Ignoring thumbs.db
                continue
            if imagePath.split(os.path.sep)[-3] == "082":
                break
            print(imagePath)
            # print(imagePath.split(os.path.sep)[-1][7:9])
            # load the image, convert it to grayscale, and describe it
            original = cv2.imread(imagePath, 0)
            processed = process(original)

            # describe
            # each [0,numPoints+2], is possible rotation invarient prototypes
            # lbp = feature.local_binary_pattern(processed, numPoints, radius, method="uniform")
            lbp = LLBP(processed)
            # print(str(len(lbp)) + " " + str(len(lbp[0])) + " " + str(lbp[0][0]))
            (desc, _) = np.histogram(lbp.ravel(), # p points -> p+1 uniform patterns
                                     bins=np.arange(0, numPoints + 3),
                                     range=(0, numPoints + 2))
            # how many values for each rotation invarient prototype
            # print(desc)
            # normalize the histogram, sum to 1
            desc = desc.astype("float")
            desc /= (desc.sum() + eps)

            # print(desc)
            # plt.plot(desc)
            # plt.show()

            # extract the label from the image path, then update the label and data lists
            labels.append(imagePath.split(os.path.sep)[-2])
            # labels.append(filename[0:len(filename)-6])
            data.append(desc)
        else:
            continue
        break

    # ss = StandardScaler()
    # vein_stand = ss.fit_transform(data)
    # pca = PCA(n_components=500)
    # vein_pca = ss.fit_transform(vein_stand)

    # train a Linear SVM on the data
    model = LinearSVC(C=1, max_iter=10000, random_state=42)
    # model.fit(vein_pca, labels)
    model.fit(data, labels)

    f = open("models/model.cpickle", "wb")
    f.write(cPickle.dumps(model))
    f.close()

model = cPickle.loads(open("models/model.cpickle", "rb").read())

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
        # lbp = feature.local_binary_pattern(processed, numPoints, radius, method="uniform")
        lbp = LLBP(processed)
        (desc, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # normalize the histogram
        desc = desc.astype("float")
        desc /= (desc.sum() + eps)

        prediction = model.predict(desc.reshape(1, -1))

        print(imagePath)
        print("predict: " + prediction[0] + " and actual: " + imagePath.split(os.path.sep)[-2])
        if prediction[0] == imagePath.split(os.path.sep)[-2]:
            correct += 1
        total += 1

        # print("predict: " + prediction[0] + "and actual: " + filename[0:len(filename)-6])
        # if prediction[0] == filename[0:len(filename)-6]:
        #     correct += 1
        # total += 1

        # display the image and the prediction
        # cv2.putText(original, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0, (255, 255, 255), 3)
        # cv2.imshow("Image", original)
        # cv2.waitKey(0)
print(correct)
print(total)
print(correct/total)
print(time.time() - start_time)