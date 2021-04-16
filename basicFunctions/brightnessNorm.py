import cv2

print("0")
bright = cv2.imread('../usefulImages/bright.JPG', 0)
dark = cv2.imread('../usefulImages/dark.JPG', 0)

bright = cv2.resize(bright, (240, 180))
dark = cv2.resize(dark, (240, 180))

bNorm = cv2.normalize(bright, None, 0, 255, cv2.NORM_MINMAX)
dNorm = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("obright", bright)
cv2.waitKey(0)
cv2.imshow("odark", dark)
cv2.waitKey(0)

cv2.imshow("bright", bNorm)
cv2.waitKey(0)
cv2.imshow("dark", dNorm)
cv2.waitKey(0)

