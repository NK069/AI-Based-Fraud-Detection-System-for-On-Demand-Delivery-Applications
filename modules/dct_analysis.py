import cv2
import numpy as np
import os

def dct_analysis(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (256, 256))

    dct = cv2.dct(np.float32(img))
    dct_log = np.log(abs(dct) + 1)

    dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
    dct_norm = np.uint8(dct_norm)

    os.makedirs("static/results", exist_ok=True)
    cv2.imwrite("static/results/dct.jpg", dct_norm)

    return np.mean(np.abs(dct))