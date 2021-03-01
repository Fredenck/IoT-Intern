import cv2
import os
from matplotlib import pyplot as plt
import numpy

fig, ax_list = plt.subplots(1, 4)

im1 = cv2.imread('bright.JPG', 0)
im2 = cv2.imread('dark.JPG', 0)
norm1 = cv2.normalize(im1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
norm2 = cv2.normalize(im2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

ax_list[0].imshow(im1, cmap='gray')
ax_list[0].set_title('Original')
ax_list[0].set_xticks([]), ax_list[0].set_yticks([])

ax_list[1].imshow(im2, cmap='gray')
ax_list[1].set_title('Original')
ax_list[1].set_xticks([]), ax_list[1].set_yticks([])

ax_list[2].imshow(norm1, cmap='gray')
ax_list[2].set_title('Original')
ax_list[2].set_xticks([]), ax_list[2].set_yticks([])

ax_list[3].imshow(norm2, cmap='gray')
ax_list[3].set_title('Original')
ax_list[3].set_xticks([]), ax_list[3].set_yticks([])

plt.draw()
plt.show()