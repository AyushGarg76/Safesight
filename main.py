# import cv2
# import argparse
# import os
# from image_enhancement import apply_trigger
# from tranform import get_birdseye_view, map_detection_to_ground

# def main():
#     parser = argparse.ArgumentParser(description="Run complete computer vision pipeline")
#     parser.add_argument("--image", type=str, default="image.png", help="Path to input image")
#     parser.add_argument("--enhance", action="store_true", help="Apply image enhancement before transformations")
#     parser.add_argument("--outdir", type=str, default="output/pipeline", help="Output directory")
#     args = parser.parse_args()

#     # Load image
#     frame = cv2.imread(args.image)
#     if frame is None:
#         print(f"Error: Could not read image at {args.image}")
#         return

#     os.makedirs(args.outdir, exist_ok=True)
#     cv2.imwrite(os.path.join(args.outdir, "1_original.jpg"), frame)
#     print(f"Loaded image {args.image} with shape {frame.shape}")

#     # Step 1: Image Enhancement (Optional/Adaptive)
#     if args.enhance:
#         print("Applying image enhancements...")
#         # Since apply_trigger automatically detects the kind of enhancement to apply:
#         frame = apply_trigger(frame, verbose=True)
#         cv2.imwrite(os.path.join(args.outdir, "2_enhanced.jpg"), frame)
#     else:
#         print("Skipping enhancements (use --enhance to apply)")

#     # Step 2: Bird's Eye View Transformation
#     print("Generating Bird's Eye View...")
#     warped_img, H_matrix = get_birdseye_view(frame)
#     cv2.imwrite(os.path.join(args.outdir, "3_birdseye.jpg"), warped_img)
#     print(f"Homography Matrix calculated.")

#     # Step 3: Example Detection Mapping
#     worker_feet = [600, 500]
#     print(f"Mapping imaginary worker feet detection at {worker_feet} to ground...")
#     ground_coord = map_detection_to_ground(worker_feet, H_matrix)
#     print(f"Worker is at ground position: {ground_coord}")

#     print(f"\nPipeline finished! Check the '{args.outdir}' folder for output images.")

# if __name__ == "__main__":
#     main()
import cv2
import argparse
import os
import subprocess
from image_enhancement import apply_trigger
from tranform import get_birdseye_view, map_detection_to_ground


def run_pipeline(args):
    # Load image
    frame = cv2.imread(args.image)

    if frame is None:
        print(f"Error: Could not read image at {args.image}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    # Save original image
    cv2.imwrite(os.path.join(args.outdir, "1_original.jpg"), frame)
    print(f"Loaded image {args.image} with shape {frame.shape}")

    # Step 1: Enhancement
    if args.enhance:
        print("Applying image enhancements...")
        frame = apply_trigger(frame, verbose=True)
        cv2.imwrite(os.path.join(args.outdir, "2_enhanced.jpg"), frame)
    else:
        print("Skipping enhancements")

    # Step 2: Bird Eye View
    print("Generating Bird's Eye View...")
    warped_img, H_matrix = get_birdseye_view(frame)
    cv2.imwrite(os.path.join(args.outdir, "3_birdseye.jpg"), warped_img)

    print("Homography Matrix calculated.")

    # Step 3: Example Worker Detection Mapping
    worker_feet = [600, 500]
    print(f"Mapping worker feet detection at {worker_feet}")

    ground_coord = map_detection_to_ground(worker_feet, H_matrix)

    print(f"Worker ground position: {ground_coord}")

    print(f"\nPipeline finished! Check '{args.outdir}' folder.")


def main():
    parser = argparse.ArgumentParser(description="Run complete computer vision pipeline")

    parser.add_argument("--image", type=str, default="image.png", help="Path to input image")
    parser.add_argument("--enhance", action="store_true", help="Apply image enhancement")
    parser.add_argument("--outdir", type=str, default="output/pipeline", help="Output folder")
    parser.add_argument("--helmet", action="store_true", help="Run helmet detection project")
    parser.add_argument("--v2", action="store_true", help="Run V2")
    args = parser.parse_args()

    # Helmet Detection Mode
    if args.helmet:
        print("Launching Helmet Detection...")
        subprocess.run(["streamlit", "run", "helmet_withoutyolo/app.py"])

    # V2 Mode
    elif args.v2:
        print("Launching V2 Project...")
        subprocess.run(["python3", "V2/scripts/app.py"])
    # Normal Pipeline Mode
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()