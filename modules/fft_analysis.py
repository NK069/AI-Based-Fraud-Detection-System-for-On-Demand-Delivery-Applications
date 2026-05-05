import cv2
import numpy as np
import os

def fft_analysis(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (256, 256))

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    fft_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    fft_norm = np.uint8(fft_norm)

    os.makedirs("static/results", exist_ok=True)
    cv2.imwrite("static/results/fft.jpg", fft_norm)

    return np.mean(magnitude)