import cv2
import numpy as np


def extract_features(image_path):
    """
    Extracts a 22-dimensional feature vector from an image.
    Returns a FLAT LIST of floats — compatible with sklearn.
    Called by both train_model.py and app.py.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img  = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    features = []

    # ---- 1. DCT — block texture energy ----
    block_means, block_stds = [], []
    for i in range(0, 256 - 8, 8):
        for j in range(0, 256 - 8, 8):
            block = gray[i:i+8, j:j+8]
            dct   = cv2.dct(block)
            ac    = dct.flatten()[1:]
            block_means.append(float(np.mean(np.abs(ac))))
            block_stds.append(float(np.std(ac)))

    features.append(float(np.mean(block_means)))           # F1
    features.append(float(np.std(block_means)))            # F2
    features.append(float(np.mean(block_stds)))            # F3
    features.append(float(np.percentile(block_means, 95))) # F4

    # ---- 2. FFT — frequency spectrum ----
    f         = np.fft.fft2(gray)
    fshift    = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    features.append(float(np.mean(magnitude)))             # F5
    features.append(float(np.std(magnitude)))              # F6
    features.append(float(np.max(magnitude)))              # F7
    features.append(float(np.percentile(magnitude, 99)))   # F8

    h, w   = magnitude.shape
    cx, cy = h // 2, w // 2
    r      = 32
    centre = magnitude[cx-r:cx+r, cy-r:cy+r]
    total  = float(np.sum(magnitude))
    features.append((total - float(np.sum(centre))) / (total + 1e-9))  # F9

    # ---- 3. PRNU — sensor noise fingerprint ----
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blur

    features.append(float(np.std(noise)))                  # F10
    features.append(float(np.mean(np.abs(noise))))         # F11

    q_stds = [float(np.std(noise[r0:r0+128, c0:c0+128]))
              for r0 in (0, 128) for c0 in (0, 128)]
    features.append(float(np.std(q_stds)))                 # F12

    # ---- 4. LBP — micro-texture ----
    gray_u8  = np.uint8(np.clip(gray, 0, 255))
    lbp_hist = _lbp_histogram(gray_u8)
    features.append(float(np.std(lbp_hist)))               # F13
    features.append(float(np.max(lbp_hist)))               # F14
    features.append(float(_entropy(lbp_hist)))             # F15

    # ---- 5. Edges & gradients ----
    edges = cv2.Canny(gray_u8, 50, 150)
    features.append(float(np.sum(edges > 0)) / edges.size) # F16

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    features.append(float(np.mean(gm)))                    # F17
    features.append(float(np.std(gm)))                     # F18

    # ---- 6. Color statistics ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    features.append(float(np.std(hsv[:, :, 1])))           # F19
    features.append(float(np.mean(hsv[:, :, 1])))          # F20

    b, g_ch, r_ch = [img[:, :, i].flatten().astype(float) for i in range(3)]
    features.append(float(np.corrcoef(r_ch, g_ch)[0, 1]))  # F21

    # ---- 7. JPEG blocking artifact ----
    features.append(float(_jpeg_blocking_score(gray_u8)))  # F22

    return features  # 22 floats


def extract_features_with_steps(image_path):
    """
    Used by app.py for the live dashboard.
    Returns features + animated step data for charts.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img  = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # DCT blocks for heatmap
    dct_blocks = []
    for i in range(0, 256 - 8, 8):
        for j in range(0, 256 - 8, 8):
            block = gray[i:i+8, j:j+8]
            dct   = cv2.dct(block)
            ac    = dct.flatten()[1:]
            dct_blocks.append(float(np.mean(np.abs(ac))))

    # FFT peaks for chart
    f         = np.fft.fft2(gray)
    fshift    = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    fft_peaks = [float(x) for x in sorted(magnitude.flatten(), reverse=True)[:100]]

    # PRNU progress for chart
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blur
    prnu_progress = []
    step = max(1, len(noise) // 50)
    for i in range(step, len(noise) + 1, step):
        prnu_progress.append(float(np.mean(np.abs(noise[:i])) * 100))

    features = extract_features(image_path)

    return {
        "features"     : features,
        "dct_blocks"   : dct_blocks[:100],
        "fft_peaks"    : fft_peaks[:100],
        "prnu_progress": prnu_progress,
        "dct_mean"     : round(float(np.mean(dct_blocks)), 2),
        "fft_mean"     : round(float(np.mean(fft_peaks)),  2),
        "prnu_mean"    : round(float(np.mean(prnu_progress)), 2),
    }


# ---- helpers ----

def _lbp_histogram(gray, radius=1, n_points=8, bins=256):
    h, w    = gray.shape
    lbp_img = np.zeros((h, w), dtype=np.uint8)
    offsets = [
        (int(round(radius * np.sin(2 * np.pi * p / n_points))),
         int(round(radius * np.cos(2 * np.pi * p / n_points))))
        for p in range(n_points)
    ]
    padded = np.pad(gray, radius, mode='edge')
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
        lbp_img  += ((neighbour >= gray).astype(np.uint8)) << bit
    hist, _ = np.histogram(lbp_img.flatten(), bins=bins, range=(0, 255), density=True)
    return hist


def _entropy(hist):
    p = hist[hist > 0]
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _jpeg_blocking_score(gray):
    h, w   = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    crop   = gray[:h8, :w8].astype(float)
    boundary_vals, interior_vals = [], []
    for i in range(0, h8, 8):
        boundary_vals.extend(crop[i, :w8].tolist())
        if i + 7 < h8:
            boundary_vals.extend(crop[i+7, :w8].tolist())
        if i + 3 < h8:
            interior_vals.extend(crop[i+3, :w8].tolist())
    if not interior_vals or not boundary_vals:
        return 0.0
    return abs(float(np.std(boundary_vals)) - float(np.std(interior_vals)))