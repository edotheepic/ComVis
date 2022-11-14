import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

base_path = './dataset/data'

img = cv.imread('./dataset/object.jpg', cv.IMREAD_GRAYSCALE)
# img - cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.equalizeHist(img)
# img = cv.GaussianBlur(img, (3,3), 1.4)

# cv.imshow('image',img)
# cv.waitKey(0)

dataset = []
for i in os.listdir(base_path):
    img_path = cv.imread(base_path+'/'+i, cv.IMREAD_GRAYSCALE)
    img_path = cv.equalizeHist(img_path)

    dataset.append(img_path)

sift = cv.SIFT_create()
# akaze = cv.AKAZE_create()

target_kp, target_desc = sift.detectAndCompute(img, None)
target_desc = target_desc.astype('f')

all_mask = []
total_match = 0
best_idx = -1
best_kp = None
best_matches = None

for idx, i in enumerate(dataset):
    scene_kp, scene_desc = sift.detectAndCompute(i, None)
    scene_desc = scene_desc.astype('f')

    index_param = dict(algorithm=1)
    search_param = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_param, search_param)
    matches = flann.knnMatch(scene_desc, target_desc, k=2)

    scene_mask = [[0,0]] * len(matches)
    match_count = 0

    for j,(m,n) in enumerate(matches):
        if m.distance < n.distance * 0.7:
            scene_mask[j] = [1,0]
            match_count += 1
    
    all_mask.append(scene_mask)
    if match_count>= total_match:
        total_match = match_count
        best_idx = idx
        best_kp = scene_kp
        best_matches = matches

result_img = cv.drawMatchesKnn(
    dataset[best_idx],
    best_kp,
    img,
    target_kp,
    best_matches,
    None,
    matchColor=[0,255,0],
    singlePointColor=[255,0,0],
    matchesMask=all_mask[best_idx]
)

plt.imshow(result_img, cmap='gray')
plt.title("best match")
plt.show()
