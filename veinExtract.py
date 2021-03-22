import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


def ill(im):
    img = im.copy()
    rows, cols = img.shape
    maax = max(map(max, img))
    miin = min(map(min, img))
    for i in range(rows):
        for j in range(cols):
            img[i][j] = ((img[i][j]-miin)*255) / (maax-miin)
    return img


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
        resized = cv2.resize(brightness, (240, 180))

        ret, thresh = cv2.threshold(resized, 90, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        top_pos = y; bottom_pos = y+h; left_pos = x; right_pos = x+w

        cropped = resized.copy()
        cropped = cropped[top_pos:bottom_pos][left_pos:right_pos]

        # clah = cv2.createCLAHE(tileGridSize=(15,15)).apply(cropped)

        # IV: Vein Extraction
        # A: Adaptive Threshold
        adapThresh = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)

        # B: Median Filtering
        median = cv2.medianBlur(adapThresh, 5)

        # C: Massive Noise Removal
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(median, connectivity=8) # need to fix
        massRem = np.zeros(labels.shape)
        sizes = stats[1:, -1]

        for i in range (num_labels-1):
            if sizes[i] >= 100:
                massRem[labels == i + 1] = 255
        # for i in range(1, num_labels): #0 is background, 1+ is components
        #     area = stats[i][4]
        #     if area < 400:



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

