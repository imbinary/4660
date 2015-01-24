__author__ = 'chris'
import numpy as np
import cv2


def translate(image, x, y):
    m = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    return shifted