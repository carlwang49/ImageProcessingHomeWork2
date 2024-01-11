import torch
from torchvision import transforms
from PIL import Image
from util import get_model, ensemble_predict_single_image, draw_boxes

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

num_folds = 4
models = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold in range(num_folds):
    model = get_model(num_classes=3).to(device)
    model_state_dict = torch.load(f'./Result/best_model_fold_{fold}.pth', map_location='cpu')
    model.load_state_dict(model_state_dict)
    model.eval()
    models.append(model)
    
image_path = '../demo_test9/c230-11996869_12.png'
image = Image.open(image_path).convert('RGB')

transformed_image = transform(image).to(device)  # The image is already transformed here
iou_threshold = 0.5  # Set a suitable IoU threshold value
ensemble_prediction = ensemble_predict_single_image(models, transformed_image, device, iou_threshold)
draw_boxes(transformed_image, ensemble_prediction, title='Prediction')