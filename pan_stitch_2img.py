# Code by Rwik Rana rwik.rana@iitgn.ac.in

import numpy as np
import cv2
import os
import imutils
import random

from Homography import homographyRansac
from Warp import warpPerspective
# from homography import ransac

class panaroma_stitching():    
    def stitch(self, images, imageAname, imageBname, ratio=0.75, reprojThresh=4.0,showMatches=False):

        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        H = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)

        result = warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # result = cv2.GaussianBlur(result, (9,9), 10)
        # result1 = cv2.warpPerspective(imageA, H,
            # (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        # print(result)
        result = np.uint8(result)
        cv2.imshow("hey", result)
        # cv2.imshow("hey1", result1)

        # cv2.waitKey(0)
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
            # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            homographyFunc = homographyRansac(4,100)
            H = homographyFunc.getHomography(ptsA, ptsB)
            return H
        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis
        

imageA = cv2.imread('Dataset/I3/3_1.JPG')
imageB = cv2.imread('Dataset/I3/3_2.JPG')
imageC = cv2.imread('Dataset/I3/3_2.JPG')
imageD = cv2.imread('Dataset/I3/3_4.JPG')
imageE = cv2.imread('Dataset/I3/3_5.JPG')



# 'Dataset/I5/DSC03003.JPG',
# 'Dataset/I5/DSC03004.JPG',
# 'Dataset/I5/DSC03005.JPG']

# imageA = cv2.imread('Dataset/I4/DSC02930.JPG')
# imageB = cv2.imread('Dataset/I4/DSC02931.JPG')
# imageC = cv2.imread('Dataset/I4/DSC02932.JPG')
# imageD = cv2.imread('Dataset/I4/DSC02933.JPG')
# imageE = cv2.imread('Dataset/I4/DSC02934.JPG')


imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
imageC = imutils.resize(imageC, width=400)
imageD = imutils.resize(imageD, width=400)
imageE = imutils.resize(imageE, width=400)


stitcher = panaroma_stitching()
# result = stitcher.stitch([imageA, imageB], showMatches=True)
# result= result[:imageA.shape[0],:]

# result = stitcher.stitch([result, imageC], showMatches=True)
result = stitcher.stitch([imageA, imageB],'Dataset/I4/DSC02930.JPG', 'Dataset/I4/DSC02931.JPG',showMatches=True)
cv2.imshow("Result2", result)
# cv2.imshow("Result3", result3)
# cv2.imshow("Keypoint Matches", vis)
cv2.waitKey(0)