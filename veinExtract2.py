import cv2
import os
from matplotlib import pyplot as plt
import numpy

for dirname, _, filenames in os.walk('input\\veinDB'):
    for filename in filenames:
        # Directory
        print(os.path.join(dirname, filename))

        # Set up Plots
        fig, ax_list = plt.subplots(1, 4)

        original = cv2.imread(os.path.join(dirname, filename), 0)
        ax_list[0].imshow(original, cmap='gray')
        ax_list[0].set_title('Original')
        ax_list[0].set_xticks([]), ax_list[0].set_yticks([])

        # III: PRE-PROCESSING
        # A: Noise Removal (typical blur)
        blur = cv2.GaussianBlur(original, (1, 1), 0)  # kernel width = height = 2, becaue r = 1

        # B: Dispelling Illumination (transform bright/dim)
        brightness = cv2.normalize(blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # C: Normalize (resize + center)
        resized = cv2.resize(brightness, (240, 180))

        ret, thresh = cv2.threshold(resized, 40, 255, cv2.THRESH_BINARY)
        # x, y, w, h = cv2.boundingRect(thresh[:, 50:190])  # x,y is top left, ignore the left and right etra
        # top_pos = y
        # bottom_pos = y+h
        # left_pos = x+50
        # right_pos = x+w+50

        top_pos = 0
        bottom_pos = 180
        left_pos = 0
        right_pos = 240
        meet = False
        for i in range(0, 179):
            if (thresh[i][120]==255 and not meet):
                top_pos = i
                meet = True
            if (thresh[i][120]==0 and meet):
                bottom_pos = i
        meet = False
        for i in range(0, 239):
            if (thresh[90][i]==255 and not meet):
                left_pos = i
                meet = True
            if (thresh[90][i]==0 and meet):
                right_pos = i
        print(left_pos, top_pos, right_pos, bottom_pos)
        xx = (int) (top_pos+bottom_pos)/2
        # yy = (int) (left_pos+right_pos)/2

        x_trans = int(resized.shape[0] // 2 - xx)
        # y_trans = int(resized.shape[1] // 2 - yy)

        # Pad and remove pixels from image to perform translation

        if x_trans > 0:
            temp = numpy.pad(resized, ((x_trans, 0), (0, 0)), mode='constant')
            temp = temp[:resized.shape[0] - x_trans, :]
        else:
            temp = numpy.pad(resized, ((0, -x_trans), (0, 0)), mode='constant')
            temp = temp[-x_trans:, :]

        # if y_trans > 0:
        #     fit = numpy.pad(temp, ((0, 0), (y_trans, 0)), mode='constant')
        #     fit = fit[:, :resized.shape[0] - y_trans]
        #
        # else:
        #     fit = numpy.pad(temp, ((0, 0), (0, -y_trans)), mode='constant')
        #     fit = fit[:, -y_trans:]

        # whites = numpy.nonzero(normalized)
        # top_pos = normalized[whites[0][len(whites[0]) - 1]][whites[1][len(whites[1]) - 1]]
        # bottom_pos = normalized[whites[0][len(whites[0]) - 1]][whites[1][len(whites[1]) - 1]]


        # IV: Vein Extraction
        


        # Plot
        ax_list[1].imshow(blur, cmap='gray')
        ax_list[1].set_title('Gaussian Blur')
        ax_list[1].set_xticks([]), ax_list[1].set_yticks([])

        # cv2.rectangle(brightness, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), -1)
        ax_list[2].imshow(brightness, cmap='gray')
        ax_list[2].set_title('Normalize Light')
        ax_list[2].set_xticks([]), ax_list[2].set_yticks([])

        ax_list[3].imshow(temp, cmap='gray')
        ax_list[3].set_title('Refit')
        ax_list[3].set_xticks([]), ax_list[3].set_yticks([])


        plt.draw()
        plt.show()

