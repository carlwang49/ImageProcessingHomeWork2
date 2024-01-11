from torchvision import transforms
from sklearn.model_selection import KFold
from util import get_model, ensemble_predict, visualize_prediction
import torch
from PIL import Image


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
    model_state_dict = torch.load(f'../segmentation_model_fold_{fold}.pth', map_location='cpu')
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

# Visualize the prediction
visualize_prediction(image_path, prediction, alpha=0.5)