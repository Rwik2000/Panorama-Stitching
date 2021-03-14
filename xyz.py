import numpy as np
import cv2
import sys
from matchers import matchers
import time
def InterpolationNearest(img, TransformedWithoutInterpolation, xBound, yBound, row, col):
    
    TransformedWithInterpolation = np.zeros((xBound, yBound, 3)).astype(int)
    Black = np.array(
                [0,0,0]
            )
    # finding the nearest neighbour and mapping it      
    for x in range(xBound):
        for y in range(yBound):
            present = np.array(
                        [x, y, 1]
                    )
            value = np.matmul(TransformedWithoutInterpolation, present)
            value = np.round(value)
            Foundx = int(value[0])
            Foundy = int(value[1])
            
            # checking if in bound or not
            if Foundx < 0 or Foundx >= row:
                TransformedWithInterpolation[x][y] = Black
            elif Foundy < 0 or Foundy >= col:
                TransformedWithInterpolation[x][y] = Black
            else:
                TransformedWithInterpolation[x][y] = img[Foundx, Foundy]
    
    return TransformedWithInterpolation

class Stitch:
    def __init__(self, filenames):
        # self.path = args
        # fp = open(self.path, 'r')
        # filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
        # print(filenames)
        self.filenames = filenames
        self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in self.filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [],None
        self.matcher_obj = matchers()
        self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d"%self.count)
        self.centerIdx = self.count/2 
        print("Center index image : %d"%self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if(i<=self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")

    def leftshift(self):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
            print("Homography is : ", H)
            xh = np.linalg.inv(H)
            print("Inverse Homography :", xh)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
            ds = ds/ds[-1]
            print("final ds=>", ds)
            f1 = np.dot(xh, np.array([0,0,1]))
            f1 = f1/f1[-1]
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
            print("image dsize =>", dsize)
            tmp = cv2.warpPerspective(a, xh, dsize)
            # cv2.imshow("warped", tmp)
            # cv2.waitKey()
            tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
            a = tmp

        self.leftImage = tmp

		
    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            print("Homography :", H)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz/txyz[-1]
            dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            cv2.imshow("tp", tmp)
            cv2.waitKey()
            # tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
            tmp = self.mix_and_match(self.leftImage, tmp)
            print("tmp shape",tmp.shape)
            print("self.leftimage shape=", self.leftImage.shape)
            self.leftImage = tmp
            
    def warpPerspective(self, im, A, output_shape):
        """ Warps (h,w) image im using affine (3,3) matrix A
        producing (output_shape[0], output_shape[1]) output image
        with warped = A*input, where warped spans 1...output_size.
        Uses nearest neighbor interpolation."""
        # print(output_shape[0], output_shape[1])
        x = np.zeros(( output_shape[0], output_shape[1],3))
        # img_or = [0,0,1]
        # M = A.dot(np.array([img_or[1],img_or[0],1]).T)
        # img_or_trns = M/M[2]

        # img_end = [im.shape[1]-1, im.shape[0]-1, 1]
        # M = A.dot(np.array([img_end[1],img_end[0],1]).T)
        # img_end_trns = M/M[2]

        # print(img_or_trns[:2],img_end_trns[:2])

        for i in range(1, im.shape[1]): #x coord iter
            # print(i)
            for j in range(1, im.shape[0]): #y coord iter
                # np.linalg.inv(A)
                M = A.dot(np.array([i,j,1]).T)
                M = M/M[2]
                p, q = M[0], M[1] #new transformed coord
                rp = int(round(p))
                rq = int(round(q))
                try:
                    # if rp<im.shape[0] and rp>=0 and rq<im.shape[1] and rq>=0:
                        # print(im[rp, rq])
                        # print(im)
                    x[rq , rp] = im[j, i]
                except:
                    pass

        return x



    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print(leftImage[-1,-1])

        t = time.time()
        black_l = np.where(leftImage == np.array([0,0,0]))
        black_wi = np.where(warpedImage == np.array([0,0,0]))
        print(time.time() - t)
        print(black_l[-1])

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                        # print "BLACK"
                        # instead of just putting it with black, 
                        # take average of all nearby values and avg it.
                        warpedImage[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(warpedImage[j,i],[0,0,0])):
                            # print "PIXEL"
                            warpedImage[j,i] = leftImage[j,i]
                        else:
                            if not np.array_equal(leftImage[j,i], [0,0,0]):
                                bw, gw, rw = warpedImage[j,i]
                                bl,gl,rl = leftImage[j,i]
                                # b = (bl+bw)/2
                                # g = (gl+gw)/2
                                # r = (rl+rw)/2
                                warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage




    def trim_left(self):
        pass

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
            # cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()

# filenames  = ['Dataset/I1/STA_0031.JPG',
# 'Dataset/I1/STB_0032.JPG',
# 'Dataset/I1/STC_0033.JPG',
# 'Dataset/I1/STD_0034.JPG']

filenames = ['Dataset/I3/3_1.JPG',
'Dataset/I3/3_2.JPG',
'Dataset/I3/3_3.JPG',
'Dataset/I3/3_4.JPG']
s = Stitch(filenames)
s.leftshift()
# s.showImage('left')
s.rightshift()
# print "done"
cv2.imwrite("test12.jpg", s.leftImage)
# print "image written"
cv2.destroyAllWindows()
# if __name__ == '__main__':
# 	try:
# 		args = sys.argv[1]
# 	except:
# 		args =("txtlists/files1.txt")
# 	finally:
# 		print "Parameters : ", args
# 	s = Stitch(args)
# 	s.leftshift()
# 	# s.showImage('left')
# 	s.rightshift()
# 	print "done"
# 	cv2.imwrite("test12.jpg", s.leftImage)
# 	print "image written"
# 	cv2.destroyAllWindows()