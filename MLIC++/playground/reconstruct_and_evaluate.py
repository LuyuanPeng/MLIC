import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

# Configuration
RECONSTRUCTED_DIR = "experiments/mlicplus240ssim-tcoms19-diff/eval_results/tcoms719_image_diffs"  # Directory containing xxx_rec.png files
RENDERED_IMAGES_DIR = "/home/luyuan/Data/image_compression/NVSPrior/tcoms19_gsplat/test"  # Rendered images directory
CAMERA_IMAGES_DIR = "/home/luyuan/Data/image_compression/NVSPrior/tcoms19_camera/test"  # Ground truth images directory
OUTPUT_DIR = "/home/luyuan/Data/image_compression/NVSPrior/tcoms19_mlic_diff_reconstructed_evaluation"  # Directory to save evaluated images

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load an image
def load_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

# Function to save an image
def save_image(filepath, image):
    cv2.imwrite(filepath, image)

# Function to calculate PSNR
def calculate_psnr(image1, image2):
    return psnr(image1, image2, data_range=255)

# Iterate through files in all three directories in parallel
psnr_values = []

# Get sorted lists of files from each directory
# Find all _rec.png files in the reconstructed directory
reconstructed_files = [f for f in os.listdir(RECONSTRUCTED_DIR) if f.endswith("_rec.png")]
# Sort files to ensure matching order
reconstructed_files.sort()
print(reconstructed_files[:5])  # Print first 5 files for verification
rendered_files = sorted(os.listdir(RENDERED_IMAGES_DIR))
camera_files = sorted(os.listdir(CAMERA_IMAGES_DIR))

# Ensure all directories have the same number of files
min_file_count = min(len(reconstructed_files), len(rendered_files), len(camera_files))

for i in range(min_file_count):
    reconstructed_file = reconstructed_files[i]
    rendered_file = rendered_files[i]
    camera_file = camera_files[i]

    reconstructed_path = os.path.join(RECONSTRUCTED_DIR, reconstructed_file)
    rendered_path = os.path.join(RENDERED_IMAGES_DIR, rendered_file)
    camera_path = os.path.join(CAMERA_IMAGES_DIR, camera_file)

    # Debugging: Print constructed file paths
    print(f"Processing files: {reconstructed_file}, {rendered_file}, {camera_file}")
    print(f"Reconstructed path: {reconstructed_path}")
    print(f"Rendered path: {rendered_path}")
    print(f"Camera path: {camera_path}")

    # Check if files exist and are readable
    if not os.path.exists(rendered_path):
        print(f"Warning: Rendered image not found: {rendered_path}")
        continue
    if not os.path.exists(camera_path):
        print(f"Warning: Camera image not found: {camera_path}")
        continue

    # Load images
    reconstructed_image = load_image(reconstructed_path)
    rendered_image = load_image(rendered_path)
    camera_image = load_image(camera_path)

    if rendered_image is None:
        print(f"Warning: Failed to load rendered image: {rendered_path}")
        continue
    if camera_image is None:
        print(f"Warning: Failed to load camera image: {camera_path}")
        continue

    # Resize images to the same size if necessary
    if rendered_image.shape != reconstructed_image.shape:
        rendered_image = cv2.resize(rendered_image, (reconstructed_image.shape[1], reconstructed_image.shape[0]))
    if camera_image.shape != reconstructed_image.shape:
        camera_image = cv2.resize(camera_image, (reconstructed_image.shape[1], reconstructed_image.shape[0]))

    # Combine reconstructed and rendered images
    combined_image = cv2.addWeighted(reconstructed_image, 0.5, rendered_image, 0.5, 0)

    # Save combined image
    combined_output_path = os.path.join(OUTPUT_DIR, reconstructed_file.replace("_rec.png", "_reconstructed.png"))
    save_image(combined_output_path, combined_image)

    # Evaluate PSNR
    psnr_value = calculate_psnr(combined_image, camera_image)
    psnr_values.append(psnr_value)
    print(f"Processed {reconstructed_file}: PSNR = {psnr_value:.2f}")

# Save PSNR results
if psnr_values:
    psnr_results_path = os.path.join(OUTPUT_DIR, "psnr_results.txt")
    with open(psnr_results_path, "w") as f:
        for i, psnr_value in enumerate(psnr_values):
            f.write(f"File {i + 1}: PSNR = {psnr_value:.2f}\n")
    print(f"PSNR evaluation completed. Results saved to {psnr_results_path}")
