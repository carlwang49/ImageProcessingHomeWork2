import numpy as np
from util import dice_coefficient, resize_prediction
from torchvision import transforms
from sklearn.model_selection import KFold
from util import get_model, ensemble_predict
import torch
from PIL import Image
import json
import cv2

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define the number of splits for cross-validation and the number of classes
n_splits = 4
num_classes = 4  # Including background as a class

# Initialize the KFold cross-validator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models for each fold
models = []
for fold in range(n_splits):
    model = get_model(num_classes).to(device)
    # 显式地将模型权重映射到CPU
    model_state_dict = torch.load(f'./Result/segmentation_model_fold_{fold}.pth', map_location='cpu')
    model.load_state_dict(model_state_dict)
    model.eval()
    models.append(model)


# Prediction and visualization for a new image
image_path = '../demo_test9/b59-15762303_9.png'  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict using the ensemble
ensemble_output = ensemble_predict(models, image_tensor)
prediction = torch.argmax(ensemble_output, dim=1).squeeze(0)

# Load the image
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

if prediction.shape[1:] != mask.shape:
    resized_prediction = resize_prediction(prediction, mask.shape)
else:
    resized_prediction = prediction

# 转换预测结果为整数类型
resized_prediction_int = resized_prediction.cpu().numpy().astype(np.int64)

# 确定预测结果中最常见的类别
predicted_class_value = np.argmax(np.bincount(resized_prediction_int.flatten(), minlength=num_classes))

# 如果最常见的类别是背景（0），则选择第二常见的类别
if predicted_class_value == 0:
    predicted_class_value = np.argmax(np.bincount(resized_prediction_int.flatten(), minlength=num_classes)[1:]) + 1

# 计算该类别的Dice Coefficient
pred_class_mask = (resized_prediction_int == predicted_class_value)
gt_class_mask = (mask == predicted_class_value)
dice_score = dice_coefficient(pred_class_mask, gt_class_mask)

# 将类别数值映射回类别名称
class_labels = {1: 'cancer', 2: 'mix', 3: 'warthin'}
predicted_class_label = class_labels.get(predicted_class_value, 'unknown')

print(f'Dice Coefficient for {predicted_class_label}: {dice_score}')
