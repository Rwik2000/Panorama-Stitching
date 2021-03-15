import cv2

img = cv2.imread("Outputs/I2.JPG")
img = img[50:-50][:]
cv2.imshow("x",img)
cv2.waitKey(0)