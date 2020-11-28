# K-Means Clustering
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im = cv2.imread('../inputs/baboon.jpg')
#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#


def seg_kmeans(img, n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, best_labels, certers = cv2.kmeans(data, n, None, criteria, 10, flags)
    centers = np.uint8(certers)
    res1 = certers[best_labels.flatten()]
    res2 = res1.reshape(img.shape)
    return res2


def indices_for(m, n):
    i, j =np.ogrid[:m, :n]
    v = np.empty((m, n, 2), dtype=np.float32)
    v[..., 0] = i
    v[..., 1] = j
    v.shape = (-1, 2)
    return v


def seg_kmeans_xy(img, n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    num = indices_for(256, 256)
    sigma = np.full([65536, 2], 1.2)
    coordinate = num / sigma
    data1 = img.reshape((-1, 3))
    data1 = np.float32(data1)
    data = np.hstack((data1, coordinate))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, best_labels, certers = cv2.kmeans(data, n, None, criteria, 10, flags)
    centers = np.uint8(certers)
    res1 = certers[best_labels.flatten()]
    res1 = np.delete(res1, [3, 4], axis=1)
    res2 = res1.reshape(img.shape)
    return res2


result_rgb_2 = seg_kmeans(im, 2)
result_rgb_4 = seg_kmeans(im, 4)
result_rgb_8 = seg_kmeans(im, 8)
result_rgbxy_2 = seg_kmeans_xy(im, 2)
result_rgbxy_4 = seg_kmeans_xy(im, 4)
result_rgbxy_8 = seg_kmeans_xy(im, 8)


##########################################################################################
cv2.imwrite('../results/ex3d_rgb_2.jpg', result_rgb_2)
cv2.imwrite('../results/ex3d_rgbxy_2.jpg', result_rgbxy_2)
cv2.imwrite('../results/ex3d_rgb_4.jpg', result_rgb_4)
cv2.imwrite('../results/ex3d_rgbxy_4.jpg', result_rgbxy_4)
cv2.imwrite('../results/ex3d_rgb_8.jpg', result_rgb_8)
cv2.imwrite('../results/ex3d_rgbxy_8.jpg', result_rgbxy_8)