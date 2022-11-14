import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#feature detection

img = cv.imread('06/target/kitkat.png', cv.IMREAD_GRAYSCALE)
img_scene = cv.imread('06/images/kitkat_scene.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.equalizeHist(img)

sift = cv.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img, None)
kp_scene, desc_scene = sift.detectAndCompute(img_scene, None)

# img = cv.drawKeypoints(img, keypoints, None)

# cv.imshow("img",img)
# cv.waitKey(0)

INDEX_KDTREE = 0
index = dict(algorithm=INDEX_KDTREE)
search = dict(checks=50)
flann = cv.FlannBasedMatcher(index, search)


# knnMatch(desc object, desc scene, k neighbor value)
matches = flann.knnMatch(descriptors, desc_scene, 2)

sceneMask = [[0,0] for j in range(0, len(matches))]

for i,(m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        sceneMask[i] = [1, 0]

img = cv.drawMatchesKnn(img, keypoints, img_scene, kp_scene, matches, None, matchColor=[0, 255, 0], singlePointColor=[255,0,0],matchesMask=sceneMask)

plt.imshow(img, cmap='gray')
plt.title("best match result")
plt.show()

