# Bilateral filtering without OpenCV
import numpy as np
import cv2
import sys
import math


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE

def gaus_kernel(winsize, gsigma):
    r = int(winsize/2)
    c = r
    kernel = np.zeros((winsize, winsize))
    sigma1 = 2*gsigma*gsigma
    for i in range(-r, r):
        for j in range(-c, c):
            kernel[i+r][j+c] = np.exp(-float(float((i*i+j*j))/sigma1))
    return kernel


def bilateral_filter_gray(image, winsize, gsigma, ssigma):
    r = int(winsize / 2)
    c = r
    image1 = np.pad(image, ((r, c), (r, c)), constant_values=0)
    image = image1
    row, col = image.shape
    sigma2 = 2 * ssigma * ssigma
    gkernel = gaus_kernel(winsize, gsigma)
    kernel = np.zeros((winsize, winsize))
    bilater_image = np.zeros((row, col))
    for i in range(1, row - r):
        for j in range(1, col - c):
            skernel = np.zeros((winsize, winsize))
            for m in range(-r, r):
                for n in range(-c, c):
                    skernel[m + r][n + c] = np.exp(-pow((int(image[i][j]) - int(image[i + m][j + n])), 2) / sigma2)
                    kernel[m + r][n + c] = skernel[m + r][n + r] * gkernel[m + r][n + r]
            sum_kernel = sum(sum(kernel))
            kernel = kernel / sum_kernel
            for m in range(-r, r):
                for n in range(-c, c):
                    bilater_image[i][j] = image[i + m][j + n] * kernel[m + r][n + c] + bilater_image[i][j]
    return bilater_image
##########################################################################################


im_gray = cv2.imread('../inputs/cat.png',0)

result_bf1 = bilateral_filter_gray(im_gray, 10, 30.0, 3.0)
result_bf2 = bilateral_filter_gray(im_gray, 10, 30.0, 30.0)
result_bf3 = bilateral_filter_gray(im_gray, 10, 100.0, 3.0)
result_bf4 = bilateral_filter_gray(im_gray, 10, 100.0, 30.0)
result_bf5 = bilateral_filter_gray(im_gray, 5, 100.0, 30.0)

result_bf1 = cv2.resize(result_bf1, (128, 128), interpolation=cv2.INTER_LINEAR)
result_bf2 = cv2.resize(result_bf2, (128, 128), interpolation=cv2.INTER_LINEAR)
result_bf3 = cv2.resize(result_bf3, (128, 128), interpolation=cv2.INTER_LINEAR)
result_bf4 = cv2.resize(result_bf4, (128, 128), interpolation=cv2.INTER_LINEAR)
result_bf5 = cv2.resize(result_bf5, (128, 128), interpolation=cv2.INTER_LINEAR)

result_bf1 = np.uint8(result_bf1)
result_bf2 = np.uint8(result_bf2)
result_bf3 = np.uint8(result_bf3)
result_bf4 = np.uint8(result_bf4)
result_bf5 = np.uint8(result_bf5)

cv2.imwrite('../results/ex2b_bf_10_30_3.jpg', result_bf1)
cv2.imwrite('../results/ex2b_bf_10_30_30.jpg', result_bf2)
cv2.imwrite('../results/ex2b_bf_10_100_3.jpg', result_bf3)
cv2.imwrite('../results/ex2b_bf_10_100_30.jpg', result_bf4)
cv2.imwrite('../results/ex2b_bf_5_100_30.jpg', result_bf5)

