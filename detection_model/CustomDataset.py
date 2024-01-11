import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    # Mapping of class labels to numerical values
    label_mapping = {
        'right normal': 1,
        'left normal': 2
    }

    def __init__(self, root_dir, transform=None):
        """
        CustomDataset constructor.

        Args:
            root_dir (str): Root directory containing subdirectories for each class.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load all image paths and corresponding json paths
        for class_name in ['Cancer', 'Mix', 'Warthin']:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_dir, img_name)
                    json_path = os.path.join(class_dir, img_name.replace('.png', '.json'))
                    self.samples.append((img_path, json_path, class_name))

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get the image and target data for a specific index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing the image and target data.
        """
        img_path, json_path, class_name = self.samples[idx]

        # Load image and apply transformations
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        if self.transform is not None:
            image = self.transform(image)
        new_size = image.size()[-2:]

        # Load JSON
        with open(json_path, 'r') as file:
            annotation = json.load(file)

        target = {}
        boxes = []
        labels = []

        for shape in annotation['shapes']:
            label = shape['label'].lower()
            # Only process 'left_normal' and 'right_normal'
            if label not in ['left normal', 'right normal']:
                continue  # Skip labels that are not 'left_normal' or 'right_normal'

            label_value = self.label_mapping.get(label, None)
            if label_value is None:
                continue  # Skip if label is not found in label_mapping

            points = shape['points']
            x_coordinates, y_coordinates = zip(*points)
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)

            # Scale the bounding box coordinates according to the image resize
            scale_x, scale_y = new_size[1] / original_size[0], new_size[0] / original_size[1]
            box = [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]
            boxes.append(box)
            labels.append(label_value)

        # Convert everything to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.nelement() != 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # Build the target dictionary
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
