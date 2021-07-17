"""
Adapted from https://medium.com/data-breach/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
aka
https://github.com/deepanshut041/feature-detection/tree/master/sift

"""

# # SIFT (Scale-Invariant Feature Transform)

# ## Import resources and display image

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('../usefulImages/modelVein.jpg', 0)
# Create test image by adding Scale Invariance and Rotational Invariance
distortedImg = cv2.pyrDown(image)
distortedImg = cv2.pyrDown(distortedImg)
num_rows, num_cols = distortedImg.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
distortedImg = cv2.warpAffine(distortedImg, rotation_matrix, (num_cols, num_rows))

# Display traning image and testing image
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Training Image")
plots[0].imshow(image)

plots[1].set_title("Testing Image")
plots[1].imshow(distortedImg)


# ## Detect keypoints and Create Descriptor

# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT_create()
print("goood")
train_keypoints, train_descriptor = sift.detectAndCompute(image, None)
test_keypoints, test_descriptor = sift.detectAndCompute(distortedImg, None)
print("good")

keypoints_without_size = np.copy(image)
keypoints_with_size = np.copy(image)

cv2.drawKeypoints(image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

cv2.drawKeypoints(image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with and without keypoints size
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Train keypoints With Size")
plots[0].imshow(keypoints_with_size, cmap='gray')

plots[1].set_title("Train keypoints Without Size")
plots[1].imshow(keypoints_without_size, cmap='gray')

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))


# ## Matching Keypoints


# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

# Perform the matching between the SIFT descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(image, train_keypoints, distortedImg, test_keypoints, matches, distortedImg, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))