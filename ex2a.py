# Gaussian filtering without OpenCV
import numpy as np
import cv2

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE


def gaussian_filter_gray(img, ksize, sigma):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    # zero padding
    padding = ksize//2
    result = np.zeros((H+padding*2, W+padding*2, C), dtype=np.float)
    result[padding: padding+H, padding: padding+W] = img.copy().astype(np.float)
    # prepare Kernel
    Kernel = np.zeros((ksize, ksize), dtype=np.float)
    for x in range(-padding, -padding+ksize):
        for y in range(-padding, -padding+ksize):
            Kernel[y+padding, x+padding] = np.exp(-(x**2+y**2) / (2*(sigma**2)))
    Kernel /= (2 * np.pi * sigma * sigma)
    Kernel /= Kernel.sum()
    tmp = result.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                result[padding + y, padding + x, c] = np.sum(Kernel * tmp[y: y + ksize, x: x + ksize, c])
    result = np.clip(result, 0, 255)
    result = result[padding: padding + H, padding: padding + W].astype(np.uint8)
    return result
##########################################################################################


im_gray = cv2.imread('../inputs/lena.jpg', 0)
im_gray = cv2.resize(im_gray, (256, 256))

result_gf1 = gaussian_filter_gray(im_gray, 5, 1.0)
result_gf2 = gaussian_filter_gray(im_gray, 5, 10.0)
result_gf3 = gaussian_filter_gray(im_gray, 10, 1.0)
result_gf4 = gaussian_filter_gray(im_gray, 10, 10.0)

result_gf1 = np.uint8(result_gf1)
result_gf2 = np.uint8(result_gf2)
result_gf3 = np.uint8(result_gf3)
result_gf4 = np.uint8(result_gf4)

cv2.imwrite('../results/ex2a_gf_5_1.jpg', result_gf1)
cv2.imwrite('../results/ex2a_gf_5_10.jpg', result_gf2)
cv2.imwrite('../results/ex2a_gf_10_1.jpg', result_gf3)
cv2.imwrite('../results/ex2a_gf_10_10.jpg', result_gf4)
