# Code by Rwik Rana rwik.rana@iitgn.ac.in

import numpy as np
import cv2
import os
import imutils
import random
# from homography import ransac

class panaroma_stitching():
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
    
    def stitch(self, images, imageAname, imageBname, ratio=0.75, reprojThresh=4.0,showMatches=False):

        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        H = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)

        result = self.warpPerspective(imageA, H,
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
            
            # print(ptsA, ptsB)
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            # H = self.homography_ransac(ptsA, ptsB)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return H
        # otherwise, no homograpy could be computed
        return None
    
    def _single_homography(self, matches):
        num_rows = 2*len(matches)
        num_cols = 9
        A = np.zeros((num_rows, num_cols))
        # print("Damn")
        for i in range(len(matches)):
            # print(matches[i])
            x_1 = (matches[i][0][0])
            y_1 = (matches[i][0][1])
            z_1 = 1 # wi
            # try:
            x_2 = (matches[i][1][0])
            y_2 = (matches[i][1][1])
        # except:
            #     # print(matches[i])
            #     # print(matches[i][0])
            #     # exit()
            #     pass
            z_2 = 1 #wi'
            A[2*i,:] = [x_1,y_1,1,0,0,0,-x_2*x_1,-x_2*y_1,-x_2*z_1]
            A[2*i+1,:]=[0,0,0,x_1,y_1,1,-y_2*x_1,-y_2*y_1,-y_2*z_1]
        
        U, D, V_t = np.linalg.svd(A)
        h = V_t[-1]
        H = np.zeros((3,3))
        H[0] = h[:3]
        H[1] = h[3:6]
        H[2] = h[6:9]
        H = H/H[2,2]

        return H

    def homography_ransac(self,ptsA, ptsB):
        final_H = 0
        Bestcount= 0
        for i in range(100):
            _inliers = []
            rand_indices = random.sample(range(1, len(ptsA)),4)
            matches = [[ptsA[i], ptsB[i]] for i in rand_indices]
            # print(matches, i)
            # print()
            H = self._single_homography(matches)
            threshold = 4
            count = 0

            for iter in range(len(ptsA)):
                curr_loc = ptsA[iter]
                trans_curr_loc = np.array([curr_loc[0], curr_loc[1], 1]).T
                transf_loc = H.dot(trans_curr_loc)
                transf_loc = transf_loc/transf_loc[2]
                actualNewLoc = np.array([ptsB[iter][0], ptsB[iter][1], 1])
                # print(transf_loc, actualNewLoc)

                if np.linalg.norm(transf_loc - actualNewLoc) <= threshold:
                    count+=1
                    _inliers.append([ptsA[iter],ptsB[iter]])
            if count>Bestcount:
                Bestcount = count
                final_H = self._single_homography(_inliers)
        
        return final_H


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
    
    

    def warpPerspective(self, im, A, output_shape):
        """ Warps (h,w) image im using affine (3,3) matrix A
        producing (output_shape[0], output_shape[1]) output image
        with warped = A*input, where warped spans 1...output_size.
        Uses nearest neighbor interpolation."""
        # print(output_shape[0], output_shape[1])
        x = np.zeros(( output_shape[0] + 1000, output_shape[1] + 1000,3))
        # img_or = [0,0,1]
        # M = A.dot(np.array([img_or[1],img_or[0],1]).T)
        # img_or_trns = M/M[2]

        # img_end = [im.shape[1]-1, im.shape[0]-1, 1]
        # M = A.dot(np.array([img_end[1],img_end[0],1]).T)
        # img_end_trns = M/M[2]

        # print(img_or_trns[:2],img_end_trns[:2])
        temp_j = 0
        temp_i = 0
        ch = 0
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
                except Exception as e:
                    # if ch==0:
                    # print(e)
                    # print(rq,rp)
                    ch=1

        return x
        

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