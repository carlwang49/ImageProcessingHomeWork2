import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision import transforms, models
from PIL import Image
import json
import os
import re
from torchvision.datasets import CocoDetection


# 假設 'data_dir' 是您所有資料的路徑
data_dir = './data_20240105'


class CustomCocoDetection(CocoDetection):
    def __init__(self, root, transforms=None, ids=None):
        super(CustomCocoDetection, self).__init__(root, annFile=None, transforms=transforms)
        self.ids = ids
        self.coco = None

    def load_coco_json(self, json_path, image_folder, ids=None):
        json_file_path = os.path.join(image_folder, json_path)
        with open(json_file_path) as json_file:
            self.coco = json.load(json_file)
        
        self.ids = ids or [0]  # 這裡我們設定一個默認的 ID 列表，因為您的 JSON 結構中可能沒有 ID，這樣就不會引發錯誤
        image_path = self.coco['imagePath']
        image_filename = os.path.basename(image_path)
        self.images = [os.path.join(image_folder, image_filename)]


    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]

        # 檢查是否有形狀資訊，若無則重新取樣一個索引
        while img_id not in coco or 'shapes' not in coco[img_id]:
            index = torch.randint(0, len(self), (1,)).item()
            img_id = self.ids[index]

        ann_ids = coco[img_id].get('shapes', [])
        
        target = {}
        if ann_ids:
            target['boxes'] = torch.as_tensor([shape['points'] for shape in ann_ids], dtype=torch.float32)
            target['labels'] = torch.as_tensor([categories.index(shape['label']) + 1 for shape in ann_ids], dtype=torch.int64)
        else:
            # 若無形狀資訊，可以設定一個預設的空標註
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)

        # 處理圖片路徑
        path = coco[img_id]['imagePath']
        path = os.path.join(self.root, re.sub(r"[^\w\s.]", "_", path))
        path = os.path.splitext(path)[0] + "_10.png"  # 將檔案副檔名改為 .png
        img = Image.open(path).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target



# 定義轉換函數
def get_transform(train):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    if train:
        # Add more training-specific transformations here
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform_list)


# 讀取所有類別的資料並創建資料集
categories = ['Cancer', 'Mix', 'Warthin']
# Load the datasets
datasets = {cat: CustomCocoDetection(root=os.path.join(data_dir, cat),
                                     transforms=get_transform(train=True))
            for cat in ['Cancer', 'Mix', 'Warthin']}

# Assuming each category has its own annotation file
for cat, dataset in datasets.items():
    json_files = [f for f in os.listdir(os.path.join(data_dir, cat)) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found in {os.path.join(data_dir, cat)}.")
    for json_file in json_files:
        json_path = json_file  # Remove redundant path join here
        image_folder = os.path.join(data_dir, cat)
        dataset.load_coco_json(json_path, image_folder)
        
# Combine all datasets into one large dataset
combined_dataset = ConcatDataset(list(datasets.values()))

# Create data loaders with more workers and pin_memory for faster data loading
batch_size = 1
num_workers = 4
pin_memory = True
train_loader = DataLoader(combined_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=pin_memory)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# Data loaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)


# Initialize the model
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
num_classes = 4  # 3 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Loss function and optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    print(epoch)
    model.train()
    running_loss = 0.0
    print('test')
    print("Length of train_loader:", len(train_loader))
    print("Length of combined_dataset:", len(combined_dataset))
    for images, targets in train_loader:
        print("Images:", images)
        print("Targets:", targets)
        break  # 只列印第一個批次
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
    
    # Update the learning rate
    lr_scheduler.step()

    # Validation phase
    model.eval()
    validation_loss = 0.0
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        validation_loss += losses.item()

    # Print the average losses after the epoch
    avg_train_loss = running_loss / len(train_loader)
    avg_validation_loss = validation_loss / len(test_loader)
    print(f"Epoch [{epoch}/{num_epochs}] - Train loss: {avg_train_loss:.4f}, Validation loss: {avg_validation_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "faster_rcnn.pth")