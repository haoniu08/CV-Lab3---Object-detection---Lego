# lego_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as T

class LegoDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Load corresponding annotation (Pascal VOC XML format)
        ann_path = os.path.join(self.ann_dir, self.img_files[idx].replace('.jpg', '.xml'))
        boxes = []
        labels = []

        # Parse XML
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            label = obj.find('name').text
            if label == 'lego':  # Use only 'lego' label, class 1
                labels.append(1)

            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

        return img, target


# Example usage of transforms
transform = T.Compose([
    T.ToTensor(),  # Converts image to PyTorch tensor
    T.Resize((600, 600)),  # Resizes images to a fixed size
])