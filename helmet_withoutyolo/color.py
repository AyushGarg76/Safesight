# color.py
import numpy as np


# ── Core conversion ───────────────────────────────────────────────────

def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (H, W, 3) uint8 → HSV float32.
    H in [0, 360),  S in [0, 1],  V in [0, 1]
    Pure NumPy — no cv2 allowed.
    """
    img = image_rgb.astype(np.float32) / 255.0
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    Cmax  = np.maximum(np.maximum(R, G), B)
    Cmin  = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    H = np.zeros_like(R)
    mask_r = (Cmax == R) & (delta > 0)
    mask_g = (Cmax == G) & (delta > 0)
    mask_b = (Cmax == B) & (delta > 0)

    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)

    S = np.divide(delta, Cmax, out=np.zeros_like(delta), where=Cmax > 0)
    V = Cmax

    return np.stack([H, S, V], axis=-1).astype(np.float32)


# ── Helmet color detection ────────────────────────────────────────────

def helmet_color_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask (H, W) — True where pixel is likely a helmet color.
    Covers orange, yellow, white, and blue helmets.
    """
    hsv = rgb_to_hsv(image_rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    orange = (H >= 15)  & (H <= 40)  & (S > 0.50) & (V > 0.40)
    yellow = (H > 40)   & (H <= 70)  & (S > 0.50) & (V > 0.40)
    white  =                            (S < 0.20) & (V > 0.80)
    blue   = (H >= 200) & (H <= 240) & (S > 0.40) & (V > 0.30)

    return orange | yellow | white | blue


# ── Morphological cleanup ─────────────────────────────────────────────

def erode(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Shrinks blobs — removes isolated noise pixels."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad    = kernel_size // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    return sliding_window_view(padded, (kernel_size, kernel_size)).all(axis=(-2, -1))


def dilate(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Expands blobs — fills gaps inside helmet regions."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad    = kernel_size // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    return sliding_window_view(padded, (kernel_size, kernel_size)).any(axis=(-2, -1))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Erode to kill noise, then dilate to fill the helmet body."""
    mask = erode(mask,  kernel_size=3)
    mask = dilate(mask, kernel_size=9)
    return mask