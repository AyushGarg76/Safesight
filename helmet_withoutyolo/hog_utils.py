import numpy as np

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Weighted luminance formula — matches human eye sensitivity.
    Green carries the most perceived brightness information.
    Input:  (H, W, 3) uint8
    Output: (H, W)    float32, range [0, 255]
    """
    return (0.299 * image[..., 0] +
            0.587 * image[..., 1] +
            0.114 * image[..., 2]).astype(np.float32)

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2D convolution — no scipy, no cv2.
    Uses stride tricks for speed instead of nested loops.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad so output is same size as input
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # sliding_window_view gives us every (kh, kw) patch without copying data
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, (kh, kw))  # shape: (H, W, kh, kw)

    # Dot each patch with the kernel
    return (windows * kernel).sum(axis=(-2, -1))


def compute_gradients(gray: np.ndarray):
    """
    Sobel-X detects vertical edges (responds to horizontal change).
    Sobel-Y detects horizontal edges (responds to vertical change).

    Sobel-X:          Sobel-Y:
    [-1  0 +1]        [+1 +2 +1]
    [-2  0 +2]        [ 0  0  0]
    [-1  0 +1]        [-1 -2 -1]
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    Gx = convolve2d(gray, Kx)
    Gy = convolve2d(gray, Ky)

    magnitude   = np.sqrt(Gx**2 + Gy**2)
    orientation = np.degrees(np.arctan2(Gy, Gx))  # range: (-180, 180]

    # HOG uses *unsigned* orientations — a vertical edge looks the same
    # whether the dark side is left or right. Fold into [0, 180).
    orientation = orientation % 180

    return Gx, Gy, magnitude, orientation

def build_cell_histograms(magnitude: np.ndarray,
                          orientation: np.ndarray,
                          cell_size: int = 8,
                          n_bins: int = 9) -> np.ndarray:
    """
    Divide the image into non-overlapping cells of (cell_size × cell_size).
    Each pixel casts a weighted vote into 2 adjacent bins (bilinear interpolation).

    Returns: (n_cells_y, n_cells_x, n_bins) — one 9-dim histogram per cell.

    Bin boundaries: 0°, 20°, 40°, 60°, 80°, 100°, 120°, 140°, 160°
    Bin centres:    10°, 30°, 50°, 70°, 90°, 110°, 130°, 150°, 170°
    """
    H, W = magnitude.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    histograms = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)

    bin_width = 180.0 / n_bins  # = 20 degrees per bin

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            # Extract the 8×8 patch for this cell
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size

            cell_mag = magnitude[y0:y1, x0:x1].ravel()   # 64 values
            cell_ori = orientation[y0:y1, x0:x1].ravel() # 64 values

            # Each pixel votes into bins using bilinear interpolation.
            # This prevents sharp jumps when an orientation crosses a bin edge.
            bin_float = cell_ori / bin_width       # e.g. 35° → bin 1.75
            bin_lo    = np.floor(bin_float).astype(int) % n_bins
            bin_hi    = (bin_lo + 1) % n_bins
            weight_hi = bin_float - np.floor(bin_float)  # fractional part
            weight_lo = 1.0 - weight_hi

            # Accumulate weighted magnitudes into both adjacent bins
            np.add.at(histograms[cy, cx], bin_lo, cell_mag * weight_lo)
            np.add.at(histograms[cy, cx], bin_hi, cell_mag * weight_hi)

    return histograms

def normalize_blocks(histograms: np.ndarray,
                     block_size: int = 2,
                     epsilon: float = 1e-6) -> np.ndarray:
    """
    Slide a (block_size × block_size) window of cells across the histogram grid.
    Stride = 1 cell, so blocks overlap.

    For a 128×64 patch: 16×8 cells → 15×7 = 105 blocks → 105×4×9 = 3780 features.

    L2-norm:  v_normalized = v / sqrt(||v||² + ε²)
    The epsilon prevents division by zero in flat regions.
    """
    n_cells_y, n_cells_x, n_bins = histograms.shape
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    descriptor = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            # Concatenate the 4 cell histograms in this 2×2 block
            block = histograms[by:by+block_size, bx:bx+block_size, :]
            block_vec = block.ravel()  # 4 × 9 = 36 values

            # L2 normalization
            norm = np.sqrt(np.sum(block_vec**2) + epsilon**2)
            descriptor.append(block_vec / norm)

    return np.concatenate(descriptor)  # final 1D feature vector

def extract_hog(image_rgb: np.ndarray,
                cell_size: int = 8,
                block_size: int = 2,
                n_bins: int = 9,
                patch_size: tuple = (128, 64)) -> tuple:
    """
    Full HOG pipeline for a single image patch.

    patch_size: (height, width) — standard is 128×64 for pedestrian/person detection.
    For helmet detection on cropped head regions you might use 64×64.

    Returns:
        descriptor  — 1D feature vector (3780 values for 128×64)
        internals   — dict with gradient/histogram data for visualization
    """
    import cv2  # only for resize — all math stays ours

    # Resize patch to standard size
    patch = cv2.resize(image_rgb, (patch_size[1], patch_size[0]))

    gray                         = rgb_to_gray(patch)
    Gx, Gy, magnitude, orient    = compute_gradients(gray)
    histograms                   = build_cell_histograms(magnitude, orient,
                                                         cell_size, n_bins)
    descriptor                   = normalize_blocks(histograms, block_size)

    internals = {
        'patch':       patch,
        'gray':        gray,
        'Gx':          Gx,
        'Gy':          Gy,
        'magnitude':   magnitude,
        'orientation': orient,
        'histograms':  histograms,
        'cell_size':   cell_size,
        'n_bins':      n_bins,
    }

    return descriptor, internals

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_hog(internals: dict,
                  scale: float = 0.8,
                  arrow_color: str = '#00E5FF') -> None:
    """
    Draws HOG orientation arrows overlaid on the image patch.

    For each cell:
      1. Find the dominant bin (highest magnitude vote)
      2. Convert bin index → angle in degrees
      3. Draw a line from cell centre outward in that direction,
         scaled by the bin's total magnitude
    """
    patch      = internals['patch']
    histograms = internals['histograms']
    cell_size  = internals['cell_size']
    n_bins     = internals['n_bins']

    n_cells_y, n_cells_x, _ = histograms.shape
    bin_width = 180.0 / n_bins          # 20° per bin
    max_mag   = histograms.max()        # for normalizing arrow length

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.patch.set_facecolor('#0D0D0D')

    # ── Panel 1: original patch ──────────────────────────────────────
    axes[0].imshow(patch)
    axes[0].set_title('Input patch', color='white', fontsize=12, pad=8)
    axes[0].axis('off')

    # ── Panel 2: gradient magnitude ──────────────────────────────────
    axes[1].imshow(internals['magnitude'], cmap='hot', vmin=0, vmax=255)
    axes[1].set_title('Gradient magnitude', color='white', fontsize=12, pad=8)
    axes[1].axis('off')

    # ── Panel 3: HOG orientation arrows ─────────────────────────────
    axes[2].imshow(patch, alpha=0.35)
    axes[2].set_title('HOG orientations', color='white', fontsize=12, pad=8)
    axes[2].set_facecolor('#0D0D0D')

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            hist = histograms[cy, cx]

            # Cell centre in pixel coordinates
            centre_y = cy * cell_size + cell_size / 2
            centre_x = cx * cell_size + cell_size / 2

            for b in range(n_bins):
                if hist[b] < max_mag * 0.15:   # skip weak bins
                    continue

                # Bin centre angle (unsigned → symmetric arrow)
                angle_deg = b * bin_width + bin_width / 2   # e.g. bin 0 → 10°
                angle_rad = np.radians(angle_deg)

                # Arrow half-length proportional to bin strength
                half_len = (hist[b] / max_mag) * cell_size * scale

                dx =  np.cos(angle_rad) * half_len
                dy = -np.sin(angle_rad) * half_len  # image y-axis flipped

                # Draw symmetric line (unsigned orientation = no preferred direction)
                axes[2].annotate(
                    '', xy=(centre_x + dx, centre_y + dy),
                    xytext=(centre_x - dx, centre_y - dy),
                    arrowprops=dict(
                        arrowstyle='-',
                        color=arrow_color,
                        lw=0.9 + hist[b] / max_mag
                    )
                )

    axes[2].axis('off')
    axes[2].set_xlim(0, patch.shape[1])
    axes[2].set_ylim(patch.shape[0], 0)

    for ax in axes:
        ax.set_facecolor('#0D0D0D')

    plt.tight_layout(pad=1.5)
    plt.savefig('hog_visualization.png', dpi=150,
                bbox_inches='tight', facecolor='#0D0D0D')
    plt.show()
    print("Saved → hog_visualization.png")

if __name__ == "__main__":
    import cv2

    img = cv2.imread("construction.jpg")
    if img is None:
        raise FileNotFoundError("construction.jpg not found — add any image and rename it")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    descriptor, internals = extract_hog(img_rgb, patch_size=(128, 64))

    print(f"Feature vector length: {len(descriptor)}")
    print(f"Min: {descriptor.min():.4f}  Max: {descriptor.max():.4f}  Mean: {descriptor.mean():.4f}")

    visualize_hog(internals)

    # This descriptor is what you feed to your SVM in Phase 3
    # Shape: (3780,) — one row per training image