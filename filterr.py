import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import feature


def ill(im):
    img = im.copy()
    rows, cols = img.shape
    maax = max(map(max, img))
    miin = min(map(min, img))
    for i in range(rows):
        for j in range(cols):
            img[i][j] = ((img[i][j]-miin)*255) / (maax-miin)
    return img


def get_pixel(img, center, x, y):
    thresh = 0
    try:
        if img[x][y] >= center:
            thresh = 1
    except:
        pass
    return thresh


def lbp_calculated_pixel(img, x, y):
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
        varr.append(get_pixel(img, center, x+i, y))
    power_val = [32, 16, 8, 4, 2, 1, 1, 2, 4, 8, 16, 32]
    vval = 0
    for i in range(len(harr)):
        vval += harr[i] * power_val[i]

    return (hval**2 + vval**2)**0.5


for dirname, _, filenames in os.walk('input\\veinDB'):
    for filename in filenames:
        # Directory
        print(os.path.join(dirname, filename))

        # Set up Plots
        fig, ax_list = plt.subplots(1, 7)

        # original = cv2.imread(os.path.join(dirname, filename), 0)
        original = cv2.imread('modelVein.jpg', 0)
        # original = cv2.imread('llbpVein.png', 0)
        ax_list[0].imshow(original, cmap='gray')
        ax_list[0].set_title('Original')
        ax_list[0].set_xticks([]), ax_list[0].set_yticks([])

        # III: PRE-PROCESSING
        # A: Noise Removal (typical blur)
        blur = cv2.GaussianBlur(original, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1

        # B: Dispelling Illumination (transform bright/dim)
        # brightness = cv2.normalize(blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        brightness = ill(blur)

        # C: Normalize (resize + center)
        resized = cv2.resize(brightness, (340, 240))

        ret, thresh = cv2.threshold(resized, 90, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        top_pos = y; bottom_pos = y+h; left_pos = x; right_pos = x+w

        cropped = resized.copy()
        cropped = cropped[top_pos:bottom_pos][left_pos:right_pos]

        # clah = cv2.createCLAHE(tileGridSize=(15,15)).apply(cropped)

        # LLBP
        height, width = cropped.shape

        img_lbp = np.zeros((height, width, 3), np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(cropped, i, j)
        img_lbp = cv2.cvtColor(img_lbp, cv2.COLOR_BGR2GRAY)
        adap_img_lbp = cv2.adaptiveThreshold(img_lbp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)

        def_lbp = feature.local_binary_pattern(cropped, 24, 8, method="uniform").astype('uint8')
        print(def_lbp.shape)
        # def_lbp = cv2.cvtColor(def_lbp, cv2.COLOR_BGR2GRAY)
        adap_def_lbp = cv2.adaptiveThreshold(def_lbp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)


        # Plot
        ax_list[1].imshow(blur, cmap='gray')
        ax_list[1].set_title('Gaussian Blur')
        ax_list[1].set_xticks([]), ax_list[1].set_yticks([])

        cv2.rectangle(thresh, (left_pos, top_pos), (right_pos, bottom_pos), (255, 255, 255), 10)
        # ax_list[2].imshow(thresh, cmap='gray')
        ax_list[2].imshow(brightness, cmap='gray')
        ax_list[2].set_title('Normalize Light')
        ax_list[2].set_xticks([]), ax_list[2].set_yticks([])

        ax_list[3].imshow(cropped, cmap='gray')
        ax_list[3].set_title('Refit')
        ax_list[3].set_xticks([]), ax_list[3].set_yticks([])

        ax_list[4].imshow(adap_img_lbp, cmap='gray')
        ax_list[4].set_title('lbp')
        ax_list[4].set_xticks([]), ax_list[4].set_yticks([])

        ax_list[5].imshow(adap_def_lbp, cmap='gray')
        ax_list[5].set_title('default_lbp')
        ax_list[5].set_xticks([]), ax_list[5].set_yticks([])


        plt.draw()
        plt.show()

