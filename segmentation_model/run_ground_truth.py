import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load the image
image_path = '../demo_test9/b59-15762303_9.png'
original_image = Image.open(image_path)
original_image_array = np.array(original_image)

# Load the JSON data
json_path = '../demo_test9/b59-15762303_9.json'
with open(json_path, 'r') as file:
    annotations = json.load(file)

# Create an empty mask with the same size as the original image
mask = np.zeros(original_image_array.shape[:2], dtype=np.uint8)

# Define colors for different classes (in RGB format)
colors = {
    'cancer': [255, 0, 0],   # Red
    'mix': [0, 255, 0],      # Green
    'warthin': [0, 0, 255]   # Blue
}

# Define unique values for each class
class_values = {
    'cancer': 1,
    'mix': 2,
    'warthin': 3
}

# Draw the polygons on the mask
for shape in annotations['shapes']:
    label = shape['label'].lower()
    if label in class_values:
        polygon = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], class_values[label])  # Using unique values for each class

# Create an RGBA mask with colors corresponding to the classes
mask_rgb = np.zeros((*original_image_array.shape, 4), dtype=np.uint8)
for label, value in class_values.items():
    mask_rgb[mask == value] = [*colors[label], 128]  # Applying the correct color based on class value

# Convert mask to PIL image for alpha compositing
mask_pil = Image.fromarray(mask_rgb, 'RGBA')

# Overlay the mask on the original image
original_image_pil = original_image.convert('RGBA')
overlayed_image = Image.alpha_composite(original_image_pil, mask_pil)

# Display the image with the overlay
plt.imshow(overlayed_image)
plt.axis('off')
plt.show()