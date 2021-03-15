
import numpy as np
import cv2
import os
import imutils
import random


class Warp():
    def __init__(self):
        self.xOffset = 1000
        self.yOffset = 1000
    def warpPerspective(self, im, A, output_shape):
        """ Warps (h,w) image im using affine (3,3) matrix A
        producing (output_shape[0], output_shape[1]) output image
        with warped = A*input, where warped spans 1...output_size.
        Uses nearest neighbor interpolation."""
        # print(output_shape[0], output_shape[1])
        warpImage = np.zeros(( output_shape[0], output_shape[1],3))
        for i in range(0, im.shape[1]): #warpImage coord iter
            # print(i)
            for j in range(0, im.shape[0]): #y coord iter
                # np.linalg.inv(A)
                M = A.dot(np.array([i,j,1]).T)
                M = M/M[2]
                p, q = M[0], M[1] #new transformed coord
                rp = int(round(p))
                rq = int(round(q))
                try:
                    if rq>=0 - self.xOffset and rp>=0 - self.yOffset:
                        warpImage[rq + self.xOffset, rp + self.yOffset] = im[j, i]
                        for k in range(4):
                            warpImage[rq + self.xOffset, rp + k + self.yOffset] = im[j, i]
                            warpImage[rq + self.xOffset, rp - k + self.yOffset] = im[j, i]
                            warpImage[rq + k + self.xOffset, rp + self.yOffset] = im[j, i]
                            warpImage[rq - k + self.xOffset, rp + self.yOffset] = im[j, i]


                except:
                    continue

        return warpImage

    def InvWarpPerspective(self, im, invA, A,output_shape):
        """ Warps (h,w) image im using affine (3,3) matrix A
        producing (output_shape[0], output_shape[1]) output image
        with warped = A*input, where warped spans 1...output_size.
        Uses nearest neighbor interpolation."""
        # print(output_shape[0], output_shape[1])

        x1 = [0,0,1]
        x2 = [im.shape[1], im.shape[0],1]
        x1_trnsf = A.dot(np.array(x1).T)
        x1_trnsf = list(x1_trnsf/x1_trnsf[2])
        x2_trnsf = A.dot(np.array(x2).T)
        x2_trnsf = list(x2_trnsf/x2_trnsf[2])

        y_min = int(min(x1_trnsf[1], x2_trnsf[1]))
        y_max = int(max(x1_trnsf[1], x2_trnsf[1]))
        x_min = int(min(x1_trnsf[0], x2_trnsf[0]))
        x_max = int(max(x1_trnsf[0], x2_trnsf[0]))

        warpImage = np.zeros(( output_shape[0], output_shape[1],3))
        for i in range(x_min, x_max): #warpImage coord iter
            # print(i)
            for j in range(y_min, y_max): #y coord iter
                # np.linalg.inv(A)
                M = invA.dot(np.array([i,j,1]).T)
                M = M/M[2]
                p, q = M[0], M[1] #new transformed coord
                rp = int(round(p))
                rq = int(round(q))

                try:
                    if rq>=0 and rp>=0 and j+self.xOffset>=0:
                        warpImage[j +self.xOffset, i +self.yOffset] = im[rq, rp]

                except:                    
                    continue              
        return warpImage