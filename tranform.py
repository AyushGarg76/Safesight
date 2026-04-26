import cv2
import numpy as np

def get_birdseye_view(frame):
    src_pts = np.float32([
        [450, 310],  # Top-left
        [820, 310],  # Top-right
        [1150, 750], # Bottom-right
        [120, 750]   # Bottom-left
    ])

    width, height = 500, 600
    dst_pts = np.float32([
        [0, 0],              # Top-left
        [width, 0],          # Top-right
        [width, height],     # Bottom-right
        [0, height]          # Bottom-left
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_image = cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)

    return warped_image, matrix

def map_detection_to_ground(coords, matrix):
    pts = np.array([[coords]], dtype='float32')
    ground_pts = cv2.perspectiveTransform(pts, matrix)
    return ground_pts

if __name__ == "__main__":
    image = cv2.imread("image_enhancement_functions/image.png")
    if image is not None:
        warped_img, H_matrix = get_birdseye_view(image) 
        
        worker_feet = [600, 500]
        ground_coord = map_detection_to_ground(worker_feet, H_matrix)
        
        print(f"Worker is at ground position: {ground_coord}")
    else:
        print("Error: Could not read image.png")