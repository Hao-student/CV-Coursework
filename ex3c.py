# Adaptive Thresholding
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt


def adaptive_thres(input, n, b_value):
    # --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

    output = cv2.adaptiveThreshold(input, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*n+1, b_value)


    ##########################################################################################

    return output


im = cv2.imread('../inputs/writing_ebu7240.png')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

output = adaptive_thres(im_gray, 2, 0.4)
cv2.imwrite('../results/ex3c_thres_0.4.jpg', output)
output = adaptive_thres(im_gray, 2, 0.6)
cv2.imwrite('../results/ex3c_thres_0.6.jpg', output)
output = adaptive_thres(im_gray, 2, 0.8)
cv2.imwrite('../results/ex3c_thres_0.8.jpg', output)

