# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/Img01.jpg')
im2 = cv2.imread('../inputs/Img02.jpg')


im_gray1 = cv2.imread('../inputs/Img01.jpg', 0)
im_gray2 = cv2.imread('../inputs/Img02.jpg', 0)

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

# Feature matching
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(im_gray1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(im_gray2, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)
knnMatches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
img_matches = cv2.drawMatches(im_gray1, keypoints_1, im_gray2, keypoints_2, matches[:50], im_gray2, flags=2)
# cv2.namedWindow('matches', 0)
# cv2.imshow('matches', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Affine transform
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 =sift.detectAndCompute(im1, None)
kp2, des2 =sift.detectAndCompute(im2, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
good = matches
COUNT = 10
if len(good) > COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 计算从 im1 到 im2 的旋转矩阵M
    # 有RANSAC
    # findHomography: 计算单应矩阵 H
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # 无RANSAC
    # M, mask = cv2.findHomography(src_pts, dst_pts)
    matchesMask = mask.ravel().tolist()
    h, w = im1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # perspectiveTransform: 根据单应矩阵 H 对坐标进行投影（旋转和平移）操作
    dst = cv2.perspectiveTransform(pts, M)
    warpImg = cv2.warpPerspective(im2, np.linalg.inv(M), (im1.shape[1]+im2.shape[1], im2.shape[0]))
    direct = warpImg.copy()
    direct[0:im1.shape[0], 0:im1.shape[1]] = im1
    rows, cols = im1.shape[:2]

    for col in range(0, cols):
        if im1[:, col].any() and warpImg[:, col].any():
            left = col
            break

    for col in range(cols-1, 0, -1):
        if im1[:, col].any() and warpImg[:, col].any():
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not im1[row, col].any():
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = im1[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(im1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
    # 从res中拿出一个分量（比如B分量）赋值给warpImg
    warpImg[0:im1.shape[0], 0:im1.shape[1]] = res
    panorama_RANSAC = warpImg
    # panorama_noRANSAC = warpImg

##########################################################################################

# cv2.imwrite('../results/ex3a_stitched_noRANSAC.jpg', panorama_noRANSAC)
cv2.imwrite('../results/ex3a_stitched_RANSAC.jpg', panorama_RANSAC)