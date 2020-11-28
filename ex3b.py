# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

# im1 = cv2.imread('../inputs/building.jpg')
im1 = cv2.imread('../inputs/YOUR_OWN.jpg')
#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

im1 = cv2.GaussianBlur(im1, (3, 3), 0)
edges = cv2.Canny(im1, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 118)
im_result = im1.copy()
for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    if (theta < (np.pi/4. )) or (theta > (3. * np.pi/4.0)):
        pt1 = (int(rho/np.cos(theta)), 0)
        pt2 = (int((rho - im_result.shape[0] * np.sin(theta))/np.cos(theta)), im_result.shape[0])
        cv2.line(im_result, pt1, pt2, 255)
    else:
        pt1 = (0, int(rho/np.sin(theta)))
        pt2 = (im_result.shape[1], int((rho - im_result.shape[1] * np.cos(theta))/np.sin(theta)))
        cv2.line(im_result, pt1, pt2, 255, 1)

##########################################################################################

# cv2.imwrite('../results/ex3b_building_hough.jpg', im_result)
cv2.imwrite('../results/YOUR_OWN_hough.jpg', im_result)