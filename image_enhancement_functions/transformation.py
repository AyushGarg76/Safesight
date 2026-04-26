import cv2
import os
import argparse


def demo(image_path: str, out_dir: str = "output/transforms"):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is not None:
        print(f"Successfully loaded image '{image_path}' with shape: {img.shape}")
    else:
        print(f"Failed to load image '{image_path}'")
        return

    h, w = img.shape[:2]

    # Euclidean
    euc = euclidean_transform(img, angle_deg=15, tx=40, ty=20)
    cv2.imwrite(f"{out_dir}/euclidean.jpg", euc)
    print(f"Euclidean saved → {out_dir}/euclidean.jpg")

    # Affine (parametric)
    aff = affine_transform(img, scale_x=1.15, scale_y=0.85, shear=0.12, angle_deg=10)
    cv2.imwrite(f"{out_dir}/affine.jpg", aff)
    print(f"Affine saved    → {out_dir}/affine.jpg")

    # Affine from 3 points
    src3 = np.float32([[0,0],[w-1,0],[0,h-1]])
    dst3 = np.float32([[w*0.05,h*0.1],[w*0.9,h*0.05],[w*0.1,h*0.85]])
    aff3 = affine_from_points(img, src3, dst3)
    cv2.imwrite(f"{out_dir}/affine_3pts.jpg", aff3)
    print(f"Affine-3pt saved→ {out_dir}/affine_3pts.jpg")

    # Projective
    src4 = np.float32([[w*0.1,h*0.1],[w*0.9,h*0.05],[w*0.95,h*0.9],[w*0.05,h*0.85]])
    dst4 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    proj = projective_transform(img, src4, dst4)
    cv2.imwrite(f"{out_dir}/projective.jpg", proj)
    print(f"Projective saved→ {out_dir}/projective.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="image.png")
    parser.add_argument("--out",   type=str, default="output/transforms")
    args = parser.parse_args()
    demo(args.image, args.out)


