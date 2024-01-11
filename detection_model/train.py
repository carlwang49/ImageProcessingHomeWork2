import torch
import copy
from torchvision import transforms
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from sklearn.model_selection import KFold
from util import collate_fn, get_model, evaluate, train_one_epoch

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
dataset = CustomDataset(root_dir='../data_20240105', transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 2 classes + background
model = get_model(num_classes=3)

# Define the number of folds for cross-validation
num_folds = 4
num_epochs = 5
kf = KFold(n_splits=num_folds, shuffle=True)

# Perform cross-validation
fold_performance = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
best_models = []

# Open a text file to save the training progress
with open('training_progress.txt', 'w') as f:
    for fold, (train_ids, test_ids) in enumerate(kf.split(dataset)):
        # Split dataset into training and validation sets for the current fold
        train_subset = torch.utils.data.Subset(dataset, train_ids)
        val_subset = torch.utils.data.Subset(dataset, test_ids)

        # Prepare data loaders
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Initialize the model for this fold
        model = get_model(num_classes=3).to(device)
        optimizer = SGD(model.parameters(), lr=0.003, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

        best_iou = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        # Train the model
        for epoch in range(num_epochs):
            epoch_loss = train_one_epoch(model, optimizer, scheduler, train_loader, device)
            iou = evaluate(model, val_loader, device)
            progress_str = f"Fold: {fold}, Epoch: {epoch}, Loss: {epoch_loss}, IoU: {iou}"
            print(progress_str)
            
            # Write the training progress to the text file
            f.write(progress_str + '\n')

            # Early stopping and save best model
            if iou > best_iou:
                best_iou = iou
                best_model_wts = copy.deepcopy(model.state_dict())

        fold_performance.append(best_iou)
        best_models.append(best_model_wts)

    # Save the best model for each fold
    for fold, model_wts in enumerate(best_models):
        torch.save(model_wts, f'./Result/best_model_fold_{fold}.pth')

# Average performance
average_performance = sum(fold_performance) / len(fold_performance)
print(f"Average IoU over {num_folds} folds: {average_performance}")
