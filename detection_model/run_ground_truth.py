from util import get_ground_truth_boxes, draw_boxes
import torch
from torchvision import transforms
from PIL import Image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


json_path = '../demo_test9/c230-11996869_12.json'
original_size = (512, 512)  
transformed_size = (256, 256)  
gt_boxes, gt_labels = get_ground_truth_boxes(json_path, original_size, transformed_size)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Convert labels to numeric values
label_mapping = {'right normal': 1, 'left normal': 2}
numeric_labels = [label_mapping.get(label, 0) for label in gt_labels]

# Convert boxes and labels to the format expected by draw_boxes
ground_truth = {'boxes': torch.tensor(gt_boxes), 'labels': torch.tensor(numeric_labels)}

# Draw ground truth boxes on the image
image_path = '../demo_test9/c230-11996869_12.png'
image = Image.open(image_path).convert('RGB')
transformed_image = transform(image).to(device)  # The image is already transformed here
draw_boxes(transformed_image, ground_truth)