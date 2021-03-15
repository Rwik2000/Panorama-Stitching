# Code by Rwik Rana rwik.rana@iitgn.ac.in

import numpy as np
import cv2
import os
import shutil
import imutils
import random

from Homography import homographyRansac
from Warp import Warp

class panaroma_stitching():    
    def __init__(self):
        self.isLeft = 1
        self.isFinal = 0
        self.LoweRatio = 0.75
    def stitchTwoImg(self, images):

        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        H, invH = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB)

        
        warpClass = Warp()
        if self.isLeft:
            warpClass.yOffset = imageA.shape[1]
        else:
            warpClass.yOffset = 200
        warpClass.xOffset = 50
        offsets = [warpClass.xOffset, warpClass.yOffset]

        result = warpClass.InvWarpPerspective(imageA, invH,H,
            (imageB.shape[0] + 100, imageB.shape[1] + imageA.shape[1]))
        result = np.uint8(result)
        if self.isFinal:
            cv2.imshow("finalwarp", result)
            cv2.waitKey(0)
        if self.isFinal==0:
            result[0 + offsets[0]:imageB.shape[0] + offsets[0], 0 +  offsets[1]:imageB.shape[1]+  offsets[1]] = imageB
        else:
            print(H)
            for i in range(0 + offsets[0], imageB.shape[0] + offsets[0]):
                for j in range(0 +  offsets[1], imageB.shape[1]+  offsets[1]):
                    blank = result[i][j]==[0,0,0]
                    if blank.all():
                        result[i][j] = imageB[i - offsets[0]][j - offsets[1]]
        return result

    def MultiStitch(self, images):
        num_imgs = len(images)
        if num_imgs%2 == 0:
            left = images[:num_imgs//2]
            right = images[num_imgs//2:]
        else:
            left = images[:num_imgs//2 + 1]
            right = images[num_imgs//2:]
        print("========> Stitching Left Side ...")
        tempLeftStitch = images[0]
        while len(left) >1:
            temp_imgB = left.pop(0)
            temp_imgA = left.pop(0)
            tempLeftStitch = self.stitchTwoImg([temp_imgA, temp_imgB])
            left.insert(0, tempLeftStitch)
        tempRightStitch = images[-1]
        self.isLeft = 0
        print("========> Done! \n========> Stitching Right Side ...")
        while len(right)>1:
            temp_imgA = right.pop(0)
            temp_imgB = right.pop(0)
            tempRightStitch = self.stitchTwoImg([temp_imgA, temp_imgB])
            right.insert(0, tempRightStitch)
        # tempLeftStitch = tempLeftStitch[50:-50][-2*images[0].shape[0]: ]
        # tempRightStitch = tempRightStitch[50:-50][: 2*images[0].shape[0]]

        # cv2.imshow("left", tempLeftStitch)
        # cv2.imshow("right", tempRightStitch)
        # cv2.waitKey(3)
        print("========> Done! \n========> Stitching Both Sides ...")
        self.isLeft = 0
        self.isFinal = 1
        final = self.stitchTwoImg([tempLeftStitch, tempRightStitch])        
        return final, tempLeftStitch, tempRightStitch
        

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
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * self.LoweRatio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])      
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            homographyFunc = homographyRansac(4,400)
            H = homographyFunc.getHomography(ptsA, ptsB)
            invH = np.linalg.inv(H)
            return H, invH
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

# Datasets = ["I1","I2","I3","I4","I5","I6"]
Datasets = ["I5"]
for Dataset in Datasets:
    print("Stitching Dataset : ", Dataset)
    Path = "Dataset/"+Dataset
    images=[]
    for filename in os.listdir(Path):
        if filename.endswith(".JPG") or filename.endswith(".PNG"):
            img_dir = Path+'/'+str(filename)
            images.append(cv2.imread(img_dir))

    for i in range(len(images)):
        images[i] = imutils.resize(images[i], width=500)

    # print(images[1].shape)
    images = images[:4]
    stitcher = panaroma_stitching()
    result, left, right = stitcher.MultiStitch(images)
    # result = result[100:-100][:]
    print("========>Done! Final Image Saved in Outputs Dir!")
    if os.path.exists("Outputs/"+Dataset):
        shutil.rmtree("Outputs/"+Dataset)
    os.makedirs("Outputs/"+Dataset, )
    cv2.imwrite("Outputs/"+Dataset+"/"+Dataset+".JPG", result)
    cv2.imwrite("Outputs/"+Dataset+"/"+Dataset+"_left.JPG", left)
    cv2.imwrite("Outputs/"+Dataset + "/"+Dataset+"_right.JPG", right)


