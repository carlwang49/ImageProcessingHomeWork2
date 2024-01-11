from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torch
import cv2
import numpy as np

class CustomDataset(Dataset):
    label_mapping = {
        'Cancer': 1,
        'Mix': 2,
        'Warthin': 3,
    }

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name in self.label_mapping.keys():
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_dir, img_name)
                    json_path = os.path.join(class_dir, img_name.replace('.png', '.json'))
                    self.samples.append((img_path, json_path, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        with open(json_path, 'r') as file:
            annotation = json.load(file)

        mask = self.create_mask_from_json(annotation, original_size, class_name)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()

        return image, mask, class_name

    def create_mask_from_json(self, annotation, image_shape, class_name):
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for shape in annotation['shapes']:
            if shape['label'].lower() == class_name.lower():
                # Convert points to a polygon
                polygon = np.array(shape['points'], dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))  # This reshaping is necessary for cv2.fillPoly
                cv2.fillPoly(mask, [polygon], self.label_mapping[class_name])

        # Resize the mask to match the transformed image size
        mask = cv2.resize(mask, (256, 256))

        return mask
