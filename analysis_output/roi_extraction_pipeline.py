import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Note: You will need to install segment-anything and torch for this to work
# pip install torch torchvision
# pip install git+https://github.com/facebookresearch/segment-anything.git
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Warning: segment_anything is not installed. Pseudo-labeling will not execute.")

def preprocess_dicom_to_image(dicom_path):
    """
    Reads a DICOM file and normalizes its pixel array into an 8-bit RGB image
    required by the Segment Anything Model (SAM).
    """
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    
    # Normalize the image to 0-255
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    
    # SAM requires 3-channel RGB images
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
        
    return img_rgb

def extract_vertebral_rois(image_path, model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
    """
    Uses SAM to extract L1-L4 vertebral ROIs from a preprocessed DICOM image.
    """
    if not os.path.exists(checkpoint_path):
        print(f"SAM checkpoint {checkpoint_path} not found. Please download it.")
        return None
        
    print(f"Loading SAM ({model_type})...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    
    # 1. Load and preprocess the image
    image = preprocess_dicom_to_image(image_path)
    predictor.set_image(image)
    
    print("Extracting L1-L4 ROIs using heuristic point prompts...")
    # 2. Define prompt points (Heuristic: estimating center of L1-L4 based on image dimensions)
    # In a real pipeline, a lightweight bounding box model (e.g., YOLO) would generate these points.
    h, w, _ = image.shape
    
    # Approximating positions (this assumes the spine is centered and vertical)
    input_points = np.array([
        [w // 2, int(h * 0.4)],  # L1 approx
        [w // 2, int(h * 0.5)],  # L2 approx
        [w // 2, int(h * 0.6)],  # L3 approx
        [w // 2, int(h * 0.7)]   # L4 approx
    ])
    input_labels = np.array([1, 1, 1, 1])  # 1 indicates a foreground point
    
    # 3. Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    # Select the mask with the highest confidence score
    best_mask = masks[np.argmax(scores)]
    
    return image, best_mask, input_points

def visualize_results(image, mask, points, save_path):
    """
    Visualizes the original image, the prompt points, and the generated SAM mask.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Overlay the mask
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
    
    # Plot prompt points
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='*', s=200, edgecolor='white')
    
    plt.title("SAM Vertebral Pseudo-labeling (L1-L4)")
    plt.axis('off')
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Example usage on the first available X-ray
    xray_dir = Path("d:/LUMOS/lumos_x")
    dicoms = list(xray_dir.glob('**/*.dcm'))
    
    if dicoms:
        test_dicom = dicoms[0]
        print(f"Testing ROI extraction on {test_dicom}")
        # To run this, you need the SAM weights downloaded locally.
        # extract_vertebral_rois(test_dicom)
        print("Pipeline script generated successfully.")
    else:
        print("No DICOM files found to test.")
