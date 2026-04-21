# Safesight 
A project under the course UCS532: Computer vision (3W13)

This `README.md` is designed to explain the mathematical logic behind your Projective Transformation code, specifically for an industrial safety context.

---


# Geometric Transformation Module: Bird's-Eye View Mapping

This module converts skewed, oblique CCTV footage into a top-down **Bird's-Eye View (BEV)**. This allows for accurate distance measurements and zone-monitoring in a factory environment.

## 1. Projective Transformation (Homography)
In a standard camera view, parallel lines (like floor markings) appear to converge at a vanishing point. To fix this, we use a **Homography Matrix ($H$)**. This is a $3 \times 3$ matrix that maps points from the source plane (Camera) to the destination plane (Ground Map).

### The Homography Matrix
The matrix $H$ has 8 degrees of freedom:

$$
H = \begin{bmatrix} 
h_{11} & h_{12} & h_{13} \\\\ 
h_{21} & h_{22} & h_{23} \\\\ 
h_{31} & h_{32} & 1 
\end{bmatrix}
$$

* **Top-left $2 \times 2$**: Handles rotation, scaling, and shearing.
* **Third Column**: Handles translation ($x, y$ shifts).
* **Bottom Row**: Handles the perspective warp (making lines parallel again).

## 2. Mathematical Calculations

### Step 1: Solving for $H$
The function `cv2.getPerspectiveTransform` takes 4 pairs of points. Each point provides two linear equations. Since there are 8 unknowns in the matrix, **4 points** are the mathematical minimum required to solve the system.

### Step 2: Point Transformation
To map a worker's detection (e.g., feet at $[x, y]$) to the ground, we use **Homogeneous Coordinates**. We add a third dimension $w=1$:

$$
\begin{bmatrix} x' \\\\ y' \\\\ w' \end{bmatrix} = H \cdot \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix}
$$

### Step 3: Perspective Division (Normalization)
The resulting $x'$ and $y'$ are in "projective space." To get the final pixel coordinates on your 2D map, the computer must divide by the scaling factor $w'$:

$$
\text{Map}_X = \frac{x'}{w'} \quad , \quad \text{Map}_Y = \frac{y'}{w'}
$$

## 3. Function Explanations

### `get_birdseye_view(frame)`
* **Input**: A raw CCTV frame.
* **Process**: Calculates the $H$ matrix and warps the entire image using **Bilinear Interpolation** (`INTER_LINEAR`).
* **Purpose**: Creates the visual "Map" of the factory floor.

### `map_detection_to_ground(coords, matrix)`
* **Input**: $[x, y]$ coordinates (ideally the bottom-center of a YOLO bounding box).
* **Process**: Applies the matrix multiplication and perspective division to that specific point.
* **Efficiency**: This is significantly faster than warping the whole image because it only calculates the transformation for a single pixel coordinate.

## 4. Why Use "Bottom-Center" for Coordinates?
When mapping detections to a ground plane, using the center of a bounding box (the person's waist) will result in a "floating" error. By using the **bottom-center** (the feet), we ensure the coordinate exists exactly on the $Z=0$ plane where the homography matrix is valid.


## Documentation and Articles

| Article | Link |
| :--- | :--- |
| The introduction | https://ayushgarg282800.substack.com/p/what-makes-a-computer-vision-project |
| The research methodology | https://shubhampathneja21.substack.com/p/the-closed-loop-workflow-a-better |
