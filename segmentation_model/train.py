from torchvision import transforms
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader, random_split
from util import collate_fn, get_transform, get_model, train_and_validate, evaluate_model
from CustomDatasetWithAugmentation import CustomDatasetWithAugmentation
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from sklearn.model_selection import KFold
import numpy as np 

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# DataLoader
dataset = CustomDataset(root_dir='../data_20240105', transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
num_epochs = 25
dataset = CustomDatasetWithAugmentation('../data_20240105', transform=get_transform(train=True))

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Model, loss, optimizer
model = get_model(num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the number of splits for cross-validation
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Convert the CustomDatasetWithAugmentation into a list to enable indexing
dataset_list = list(dataset)

# Store the metrics for each fold
fold_metrics = []
with open('training_progress.txt', 'w') as f:
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_list)):
        print(f"Starting fold {fold + 1}/{n_splits}")
        
        # Subset the dataset based on the indices for this fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Create DataLoaders for training and validation subsets
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Initialize the model and optimizer for this fold
        model = get_model(num_classes=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train the model using the train_and_validate function you've defined
        trained_model = train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device)
        
        # Evaluate the model and store the metrics
        fold_eval_metrics = evaluate_model(trained_model, val_loader, criterion, device)
        
        # Print the metrics for the current fold
        progress_str = f"Fold {fold + 1}/{n_splits} - Loss: {fold_eval_metrics['loss']:.4f}, Accuracy: {fold_eval_metrics['accuracy']:.4f}"
        print(progress_str)
            
        # Write the training progress to the text file
        f.write(progress_str + '\n')
        
        fold_metrics.append(fold_eval_metrics)
        
        # Optionally, save the model for this fold
        torch.save(trained_model.state_dict(), f'./Result/segmentation_model_fold_{fold}.pth')

    # Calculate and print the average metrics across all folds
    average_metrics = {metric: np.mean([fm[metric] for fm in fold_metrics]) for metric in fold_metrics[0]}
    print("Average metrics across all folds:", average_metrics)