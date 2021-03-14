import cv2
import numpy as np
k = np.zeros((200,500,3))

for i in range(0,k.shape[0], 5):
    for j in range(k.shape[1]):
        k[i,j] = (255,255,255)
cv2.imshow("k",k)
cv2.waitKey(0)

