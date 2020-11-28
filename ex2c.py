# Median filtering without OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE

def median_filter_gray(img, size):
    img_cp = np.copy(img)
    pad_num = int((size - 1)/2)
    img_cp = np.pad(img_cp, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = img_cp.shape
    output_img = np.copy(img_cp)
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_img[i, j] = np.median(img_cp[i - pad_num: i + pad_num + 1, j - pad_num: j + pad_num + 1])
    output_img = output_img[pad_num: m - pad_num, pad_num: n - pad_num]
    return output_img
##########################################################################################


im_gray = cv2.imread('../inputs/cat.png',0)
im_gray = cv2.resize(im_gray, (256,256))

gaussian_noise = np.zeros((im_gray.shape[0], im_gray.shape[1]),dtype=np.uint8)#
gaussian_noise = cv2.randn(gaussian_noise, 128, 20)

uniform_noise = np.zeros((im_gray.shape[0], im_gray.shape[1]),dtype=np.uint8)
uniform_noise = cv2.randu(uniform_noise,0,255)
ret, impulse_noise = cv2.threshold(uniform_noise,220,255,cv2.THRESH_BINARY)

gaussian_noise = (gaussian_noise*0.5).astype(np.uint8)
impulse_noise = impulse_noise.astype(np.uint8)

imnoise_gaussian = cv2.add(im_gray, gaussian_noise)
imnoise_impulse = cv2.add(im_gray, impulse_noise)

imnoise_gaussian = cv2.resize(imnoise_gaussian, (128, 128), interpolation=cv2.INTER_LINEAR)
imnoise_impulse = cv2.resize(imnoise_impulse, (128, 128), interpolation=cv2.INTER_LINEAR)

cv2.imwrite('../results/ex2c_gnoise.jpg', np.uint8(imnoise_gaussian))
cv2.imwrite('../results/ex2c_inoise.jpg', np.uint8(imnoise_impulse))

result_original_mf = median_filter_gray(im_gray, 5)
result_gaussian_mf = median_filter_gray(imnoise_gaussian, 5)
result_impulse_mf = median_filter_gray(imnoise_impulse, 5)

result_original_mf = cv2.resize(result_original_mf, (128, 128), interpolation=cv2.INTER_LINEAR)
result_gaussian_mf = cv2.resize(result_gaussian_mf, (128, 128), interpolation=cv2.INTER_LINEAR)
result_impulse_mf = cv2.resize(result_impulse_mf, (128, 128), interpolation=cv2.INTER_LINEAR)

cv2.imwrite('../results/ex2c_original_median_5.jpg', np.uint8(result_original_mf))
cv2.imwrite('../results/ex2c_gnoise_median_5.jpg', np.uint8(result_gaussian_mf))
cv2.imwrite('../results/ex2c_inoise_median_5.jpg', np.uint8(result_impulse_mf))

result_original_mf = median_filter_gray(im_gray, 11)
result_gaussian_mf = median_filter_gray(imnoise_gaussian, 11)
result_impulse_mf = median_filter_gray(imnoise_impulse, 11)

result_original_mf = cv2.resize(result_original_mf, (128, 128), interpolation=cv2.INTER_LINEAR)
result_gaussian_mf = cv2.resize(result_gaussian_mf, (128, 128), interpolation=cv2.INTER_LINEAR)
result_impulse_mf = cv2.resize(result_impulse_mf, (128, 128), interpolation=cv2.INTER_LINEAR)

cv2.imwrite('../results/ex2c_original_median_11.jpg', np.uint8(result_original_mf))
cv2.imwrite('../results/ex2c_gnoise_median_11.jpg', np.uint8(result_gaussian_mf))
cv2.imwrite('../results/ex2c_inoise_median_11.jpg', np.uint8(result_impulse_mf))

