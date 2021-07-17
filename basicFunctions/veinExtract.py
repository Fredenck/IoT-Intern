import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

def singleScaleRetinex(img,variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    print(img_retinex.shape)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex


variance_list = [15, 80, 30]
variance = 300

for dirname, _, filenames in os.walk('../input/veinDB'):
    for filename in filenames:
        # Directory
        print(os.path.join(dirname, filename))

        # Set up Plots
        fig, ax_list = plt.subplots(1, 7)

        # original = cv2.imread(os.path.join(dirname, filename))
        original = cv2.imread('../usefulImages/modelVein.jpg')
        # original = cv2.imread('../usefulImages/llbpVein.png')
        # original = cv2.imread('../usefulImages/clear-vein.png')
        ax_list[0].imshow(original, cmap='gray')
        ax_list[0].set_title('Original')
        ax_list[0].set_xticks([]), ax_list[0].set_yticks([])

        # III: PRE-PROCESSING
        # A: Noise Removal (typical blur)
        blur = cv2.GaussianBlur(original, (3, 3), 1, 1, cv2.BORDER_DEFAULT)  # 3x3 matrix, becaue r = 1; stdev of 1

        # B: Dispelling Illumination (transform bright/dim)
        brightness = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

        # C: Normalize (resize + center)
        # resized = cv2.resize(brightness, (340, 240))

        ret, thresh = cv2.threshold(brightness, 90, 255, cv2.THRESH_BINARY)
        print(thresh.shape)
        cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY))
        top_pos = y; bottom_pos = y+h; left_pos = x; right_pos = x+w;

        cropped = brightness.copy()
        # cropped = cropped[top_pos:bottom_pos][left_pos:right_pos]
        cropped = cropped[top_pos:bottom_pos, left_pos:right_pos]

        ssr = SSR(cropped, variance)
        ssr = cv2.cvtColor(ssr, cv2.COLOR_BGR2GRAY)

        # IV: Vein Extraction
        # A: Adaptive Threshold
        # clahe = cv2.createCLAHE(tileGridSize=(15, 15)).apply(cropped)
        clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8)).apply(ssr)
        # plt.hist(clahed.flat, bins=100, range=(100, 255))
        # plt.show()
        # ret, adapThresh = cv2.threshold(clahed, 160, 150, cv2.THRESH_BINARY)
        # ret, adapThresh = cv2.threshold(clahed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adapThresh = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)

        # B: Median Filtering
        median = cv2.medianBlur(adapThresh, 5)

        # C: Massive Noise Removal
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(median, connectivity=8) # need to fix
        massRem = np.zeros(labels.shape)
        sizes = stats[1:, -1]

        for i in range (num_labels-1):
            if sizes[i] >= 400:
                massRem[labels == i + 1] = 255

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

        ax_list[4].imshow(~adapThresh, cmap='gray')
        ax_list[4].set_title('Adaptive')
        ax_list[4].set_xticks([]), ax_list[4].set_yticks([])

        ax_list[5].imshow(~median, cmap='gray')
        ax_list[5].set_title('Median')
        ax_list[5].set_xticks([]), ax_list[5].set_yticks([])

        ax_list[6].imshow(~massRem.astype(int), cmap='gray')
        ax_list[6].set_title('CCL')
        ax_list[6].set_xticks([]), ax_list[6].set_yticks([])

        plt.draw()
        plt.show()

