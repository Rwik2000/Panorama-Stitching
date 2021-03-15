# Panorama Stitching

Code by Rwik Rana, IITGn
View the project on https://github.com/Rwik2000/Panorama-Stitching for better viewing 

This is a Code to stitch multiple images to create panoramas. This is robust for stitching 4 images. This is the tree for the entire repo:\

```
Panorama Stitching
        |
        |___Dataset
        |       |--I1
        |       |--I2 ....
        |
        |___Outputs
        |
        |___blend.py
        |___Homography.py
        |___Main.py
        |___Warp.py

```

In this code, I have used concepts of inverse homography, 
warping, RANSAC and Laplacian Blending to solve the problem statement.

### To use the code, 
In the [main.py](https://github.com/Rwik2000/Panorama-Stitching/blob/main/main.py), 
1. Add your dataset in the Dataset Directory.
2. In [line 147](https://github.com/Rwik2000/Panorama-Stitching/blob/main/main.py#L147), add your datasets in the Datasets array.
3. The output of the same would be save in the Output directory.

### Flow of the Code:
1. SIFT to detect features.
2. Matching features using KNN
3. Finding the the homography matrix i.e. the mapping from one image plane to another.
4. warping the images using the inverse of homography matrix to 
   ensure no distortion/blank spaces remain in the transformed image.
5. Stitching all the warped images together

#### Implementation of Ransac:
Refer to the `homographyRansac()` in [Homography.py](https://github.com/Rwik2000/Panorama-Stitching/blob/main/homography.py). Access the object `getHomography()` of the class to get the desired homography matrix.
```python
        # Set the threshold and number of iterations neede in RANSAC.
        self.ransacThresh = ransacThresh
        self.ransaciter = ransacIter

```


