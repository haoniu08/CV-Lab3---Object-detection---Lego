import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class LegoDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        
        # Load image with OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Load label (assumes YOLO format)
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                x = x * self.img_size
                y = y * self.img_size
                w = w * self.img_size
                h = h * self.img_size
                boxes.append([cls, x, y, w, h])

        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.0  # Normalize and convert to tensor

        return image, boxes

# Usage example:
train_dataset = LegoDataset(image_dir="lego_dataset/train/images", label_dir="lego_dataset/train/annotations", img_size=640)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
