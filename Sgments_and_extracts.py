import cv2
import numpy as np
import json
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

# --- SETUP ---
image_path = "/Users/divyanshisharma/Downloads/segment-anything/sat_test.png"
checkpoint = "/Users/divyanshisharma/Downloads/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to("cuda" if torch.cuda.is_available() else "cpu")

# Initialize mask generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

# Generate masks
masks = mask_generator.generate(image_rgb)

# Find largest mask
largest_mask = max(masks, key=lambda m: np.sum(m['segmentation']))
binary_mask = largest_mask['segmentation'].astype(np.uint8) * 255

# Extract contour
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

# Format contour for GPT
contour_points = simplified_contour.squeeze().tolist()
if isinstance(contour_points[0], int):  # Edge case
    contour_points = [contour_points]

# GPT prompt structure
gpt_input = {
    "instruction": "You are an assistant that places simulation objects in satellite images based on regions.",
    "contour": contour_points,
    "object_property_rules": {
        "Fire": {
            "size": [50, 50, 50],
            "rule": "Only place on flammable regions"
        },
        "Aircraft": {
            "size": [40, 40, 20],
            "rule": "Only place on runways"
        },
        "Tank": {
            "size": [100, 20, 20],
            "rule": "Can be placed on flat ground"
        }
    },
    "example_output_format": [
        {
            "name": "Fire1",
            "position": {"x": 130, "y": 35, "z": -50},
            "rotation": {"h": 270, "p": 0, "r": 0},
            "scale": {"x": 1, "y": 1, "z": 1},
            "fxs": [
                {
                    "template": "fire",
                    "name": "Fire1",
                    "position": {"x": 0, "y": 0, "z": 50},
                    "rotation": {"h": 0, "p": 0, "r": 0},
                    "scale": {"x": 50, "y": 50, "z": 50}
                }
            ]
        }
    ]
}

# Save to JSON
with open("gpt_input_prompt.json", "w") as f:
    json.dump(gpt_input, f, indent=2)

# Optional: visualize the contour
cv2.drawContours(image_rgb, [simplified_contour], -1, (0, 255, 0), 2)
plt.imshow(image_rgb)
plt.title("Simplified Contour")
plt.axis("off")
plt.show()
