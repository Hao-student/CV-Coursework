# SIFT matching using OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt


im_gray1 = cv2.imread('../inputs/sift_input1.jpg', 0)
im_gray2 = cv2.imread('../inputs/sift_input2.jpg', 0)

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

# Run SIFT descriptor and draw the keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(im_gray1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(im_gray2, None)
img_sift_kp_1 = cv2.drawKeypoints(im_gray1, keypoints_1, im_gray1)
img_sift_kp_2 = cv2.drawKeypoints(im_gray2, keypoints_2, im_gray2)

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors_1, descriptors_2)
matches1 = sorted(matches, key=lambda x: x.distance)  # 将匹配从最好到最差排序
matches2 = sorted(matches, key=lambda x: x.distance, reverse=True)  # 将匹配从最差到最好排序
knnMatches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
# 选择最好的50个匹配
img_most50 = cv2.drawMatches(im_gray1, keypoints_1, im_gray2, keypoints_2, matches1[:50], im_gray2, flags=2)
# 选择最差的50个匹配
img_least50 = cv2.drawMatches(im_gray1, keypoints_1, im_gray2, keypoints_2, matches2[:50], im_gray2, flags=2)

##########################################################################################

# Keypoint maps
cv2.imwrite('../results/ex2d_sift_input1.jpg', np.uint8(img_sift_kp_1))
cv2.imwrite('../results/ex2d_sift_input2.jpg', np.uint8(img_sift_kp_2))


# Feature Matching outputs
cv2.imwrite('../results/ex2d_matches_least50.jpg', np.uint8(img_least50))
cv2.imwrite('../results/ex2d_matches_most50.jpg', np.uint8(img_most50))