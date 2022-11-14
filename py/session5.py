import enum
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv.imread('images/corner1.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

#Harris
harris = cv.cornerHarris(img_gray, 2, 5, 0.04)

img_res = img.copy()

img_res[harris > 0.01 * harris.max()] = [255,0,0]

img_res = cv.cvtColor(img_res, cv.COLOR_BGR2RGB)

# plt.imshow(img_res)
# plt.title("Harris corner detection")
# plt.show()

#harris corner with subPix  
_, thr = cv.threshold(harris, 0.01 * harris.max(), 255, 0)
thr = np.uint8(thr)

_,_,_, centroids = cv.connectedComponentsWithStats(thr)

criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 0.001)

centroids = np.float32(centroids)

corners = cv.cornerSubPix(img_gray, centroids, (5, 5), (-1, -1), criteria)

img_res_subpix = img.copy()

corners = np.int16(corners)

for corner in corners:
    corner_y = corner[1]
    corner_x = corner[0]
    img_res_subpix[corner_y, corner_x] = [0,0,255]

img_res_subpix = cv.cvtColor(img_res_subpix, cv.COLOR_BGR2RGB)

# plt.imshow(img_res_subpix)
# plt.title("Harris with subPix")
# plt.show()

#FAST feature detection

img2 = cv.imread('images/corner3.jpg')

fast = cv.FastFeatureDetector_create()
keypoints = fast.detect(img2)
img2_res = img2.copy()
cv.drawKeypoints(img2, keypoints, img2_res, (255,0,0))
img2_res = cv.cvtColor(img2_res, cv.COLOR_BGR2RGB)

# plt.imshow(img2_res)
# plt.title('FAST Feature Detector')
# plt.show()

#ORB Feature detection

orb = cv.ORB_create()
keypoints = orb.detect(img2)
img2_res = img2.copy()
cv.drawKeypoints(img2, keypoints, img2_res, (0,0,255))
img2_res = cv.cvtColor(img2_res, cv.COLOR_BGR2RGB)

# plt.imshow(img2_res)
# plt.title('ORB Feature Detector')
# plt.show()

#Image enumeration

for i, filename in enumerate(os.listdir('images')):
    image = cv.imread('images/'+filename)
    fast = cv.FastFeatureDetector_create()
    keypoints = fast.detect(image)
    fast_res = image.copy()

    cv.drawKeypoints(image, keypoints, fast_res, (255,0,0))
    fast_res = cv.cvtColor(fast_res, cv.COLOR_BGR2RGB)

    plt.subplot(len(os.listdir('images')), 2, i*2+1)
    plt.imshow(fast_res)

    orb = cv.ORB_create()
    keypoints = orb.detect(image)
    orb_res = image.copy()

    cv.drawKeypoints(image, keypoints, orb_res, (0,0,255))
    orb_res = cv.cvtColor(orb_res, cv.COLOR_BGR2RGB)

    plt.subplot(len(os.listdir('images')), 2, i*2+2)
    plt.imshow(orb_res)

plt.show()