"""
Adapted from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
Basic Canny Edge Detection
"""
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

fig, ax_list = plt.subplots(1, 3)
# fig = plt.figure()

img = cv2.imread('../usefulImages/simpleFace.jpg', 0)
defaultCanny = cv2.Canny(img, 100, 200)

ax_list = ax_list.ravel()

ax_list[0].imshow(img, cmap='gray')
ax_list[0].set_title('Original')
ax_list[0].set_xticks([]), ax_list[0].set_yticks([])

ax_list[1].imshow(defaultCanny, cmap='gray')
ax_list[1].set_title('Default Canny')
ax_list[1].set_xticks([]), ax_list[0].set_yticks([])

ax_list[2].imshow(defaultCanny, cmap='gray')
ax_list[2].set_title('Adjustable Image')
ax_list[2].set_xticks([]), ax_list[0].set_yticks([])

# plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 2), plt.imshow(defaultCanny, cmap='gray')
# plt.title('Default Edge Image'), plt.xticks([]), plt.yticks([])

# mod = plt.subplot(1, 3, 3)
# plt.imshow(defaultCanny, cmap='gray')
# plt.title('Adjustable Image'), plt.xticks([]), plt.yticks([])

axcolor = 'lightgoldenrodyellow'
axmin = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axmax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
smin = Slider(axmin, 'Minimum Threshold', 50, 300, valinit=100, valstep=5.0)
smax = Slider(axmax, 'Maximum Threshold', 50, 300, valinit=200)


def update(val):
    global fig, ax_list
    ax_list = ax_list.ravel()

    min = smin.val
    max = smax.val

    ax_list[2].imshow(cv2.Canny(img, min, max), cmap='gray')
    plt.draw()
    # plt.subplot(1, 3, 3).clear()
    # mod.imshow(cv2.Canny(img, min, max), cmap='gray')
    # plt.title('Adjustable Image')
    fig.canvas.draw_idle()


smin.on_changed(update)
smax.on_changed(update)

plt.show()
