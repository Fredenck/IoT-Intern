import sys

from scipy import spatial
from skimage import feature
import numpy as np
from sklearn import svm, cluster
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
    resized = resized[:][40:200]
    cropped = resized.copy()

    ret, thresh = cv2.threshold(resized, 90, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    top_pos = y
    bottom_pos = y + h
    left_pos = x
    right_pos = x + w

    cropped = cropped[top_pos:bottom_pos][left_pos:right_pos]
    # cropped = cv2.resize(cropped, (40, 30))
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8)).apply(cropped)

    # IV: Vein Extraction
    # A: Adaptive Threshold
    adapThresh = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)

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
    # ~massRem
    return massRem.astype(int)


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


bookmark = False

sift = cv2.SIFT_create()
i = 0
# total_SIFT_features = [[0]*200]*(36*82)
total_SIFT_features = np.zeros((300000, 128))
labels = []

# loop over the trainingCR images
if input("Train? ") == "Yes":
    for dirname, _, filenames in os.walk('input\\veinDB'):
        for filename in filenames:
            imagePath = os.path.join(dirname, filename)
            # Ignoring thumbs.db
            if imagePath.split(os.path.sep)[-1][7:9] == "db":
                continue
            if imagePath.split(os.path.sep)[-3] == "082":
                bookmark = True
                break
            print(imagePath)
            # print(imagePath.split(os.path.sep)[-1][7:9])
            # load the image, convert it to grayscale, and describe it
            original = cv2.imread(imagePath, 0)
            processed = process(original)
            image8bit = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            keypoints, descriptors = sift.detectAndCompute(image8bit, None)

            desc_samples = descriptors[np.random.randint(descriptors.shape[0], size=20)]

            print(desc_samples.shape)
            print(total_SIFT_features.shape)
            total_SIFT_features[20*i:20*i+20] = desc_samples
            i += 1
        if bookmark:
            break

    vocab = cluster.k_means(total_SIFT_features, 200)
    feats = []
    labels = []

    for dirname, _, filenames in os.walk('input\\veinDB'):
        for filename in filenames:
            imagePath = os.path.join(dirname, filename)
            # Ignoring thumbs.db
            if imagePath.split(os.path.sep)[-1][7:9] == "db":
                continue
            if imagePath.split(os.path.sep)[-3] == "082":
                bookmark = True
                break
            print(imagePath)
            # print(imagePath.split(os.path.sep)[-1][7:9])
            # load the image, convert it to grayscale, and describe it
            original = cv2.imread(imagePath, 0)
            processed = process(original)
            image8bit = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            keypoints, descriptors = sift.detectAndCompute(image8bit, None)

            dist = spatial.distance.cdist(descriptors, vocab, "euclidean")
            bin_assignment = np.argmin(dist, axis=1)

            image_feats = np.zeros(200)
            for a in bin_assignment:
                image_feats[a] += 1

            feats.append(image_feats)
            if (imagePath.split(os.path.sep)[-2]=="left"):
                labels.append(0)
            else:
                labels.append(1)

    feats = np.asarray(feats)
    feats_norm_div = np.linalg.norm(feats, axis=1)
    for i in range(0, feats.shape[0]):
        feats[i] = feats[i]/feats_norm_div[i]


    # train a Linear SVM on the data
    # vocab = cluster.kMeans(data, 200)
    model = svm.SVC(C=1.0, kernel="linear", probability=True, max_iter=10e5)
    model.fit(feats, labels.reshape(labels.shape[0],))

    f = open("models/modelSIFT.cpickle", "wb")
    f.write(cPickle.dumps(model))
    f.close()

model = cPickle.loads(open("models/modelSIFT.cpickle", "rb").read())

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
        image8bit = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(image8bit, None)

        dist = spatial.distance.cdist(descriptors, vocab, "euclidean")
        bin_assignment = np.argmin(dist, axis=1)

        image_feats = np.zeros(200)
        for a in bin_assignment:
            image_feats[a] += 1

        prediction = model.predict(image_feats)

        print(imagePath)
        print("predict: " + prediction[0] + " and actual: " + imagePath.split(os.path.sep)[-2])
        if prediction[0] == imagePath.split(os.path.sep)[-2]:
            correct += 1
        total += 1

print(correct)
print(total)
print(correct/total)
print(time.time() - start_time)