import cv2
import numpy as np

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


def process(image):
    # III: PRE-PROCESSING
    # A: Noise Removal (typical blur)
    blur = cv2.GaussianBlur(image, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1

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
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8)).apply(brightness)

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


im = cv2.imread("input\\DRIVE\\test\\images\\02_test.tif", 0)
# im = cv2.imread("usefulImages/modelVein.jpg", 0)
cv2.imshow("o", im)
cv2.waitKey(0)

imc = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8)).apply(im)


adapT = cv2.adaptiveThreshold(imc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
adapT = cv2.adaptiveThreshold(adapT, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
adapT = cv2.adaptiveThreshold(adapT, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
adapT = cv2.adaptiveThreshold(adapT, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
adapT = cv2.adaptiveThreshold(adapT, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
median = cv2.medianBlur(adapT, 5)
median = cv2.medianBlur(median, 5)
median = cv2.medianBlur(median, 5)
median = cv2.medianBlur(median, 5)
median = cv2.medianBlur(median, 5)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(median, connectivity=8)  # need to fix
massRem = np.zeros(labels.shape)
sizes = stats[1:, -1]

for i in range(num_labels - 1):
    if sizes[i] >= 100:
        massRem[labels == i + 1] = 255

cv2.imshow("mass", massRem)
cv2.waitKey(0)

p = process(im)
p = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

cv2.imshow("processed", p)
cv2.waitKey(0)
