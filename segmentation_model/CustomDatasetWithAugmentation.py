from CustomDataset import CustomDataset
from PIL import Image
import torch
import json


class CustomDatasetWithAugmentation(CustomDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        super().__init__(root_dir, transform)
        self.augmentation = augmentation

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

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        return image, mask, class_name