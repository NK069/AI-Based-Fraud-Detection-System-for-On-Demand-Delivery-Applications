import cv2
import numpy as np
import os

def prnu_analysis(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    noise = gray - blur

    prnu_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    prnu_norm = np.uint8(prnu_norm)

    os.makedirs("static/results", exist_ok=True)
    cv2.imwrite("static/results/prnu.jpg", prnu_norm)

    return np.std(noise)