
import numpy as np
import cv2
import os
import imutils
import random

def warpPerspective(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    # print(output_shape[0], output_shape[1])
    x = np.zeros(( output_shape[0] + 1000, output_shape[1] + 1000,3))
    for i in range(0, im.shape[1]): #x coord iter
        # print(i)
        for j in range(0, im.shape[0]): #y coord iter
            # np.linalg.inv(A)
            M = A.dot(np.array([i,j,1]).T)
            M = M/M[2]
            p, q = M[0], M[1] #new transformed coord
            rp = int(round(p))
            rq = int(round(q))
            try:
                if rq>=0 and rp>=0:
                    x[rq , rp] = im[j, i]
            except:
                continue

    return x