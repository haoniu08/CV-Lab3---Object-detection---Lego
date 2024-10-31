import torch
from model import get_model  # Import model definition
from model_training import train_model  # Import training function
from lego_counting import count_legos  # Import counting function
from lego_dataloader import get_dataloaders

# Set the device to CPU
device = torch.device("cpu")  # Use CPU since CUDA is not available

# Paths to your image and annotation directories
train_img_dir = "lego_dataset/train/images"
train_ann_dir = "lego_dataset/train/annotations"
val_img_dir = "lego_dataset/val/images"
val_ann_dir = "lego_dataset/val/annotations"
test_img_dir = "lego_dataset/test/images"
test_ann_dir = "lego_dataset/test/annotations"

# Hyperparameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# Get the model and move it to the CPU
model = get_model(num_classes=2).to(device)  # Move model to the appropriate device
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Get data loaders
train_loader, val_loader, test_loader = get_dataloaders(
    train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, test_img_dir, test_ann_dir, batch_size=batch_size
)

# Train the model
trained_model = train_model(model, train_loader, val_loader, optimizer, num_epochs, device)

# Detect and count in test images
trained_model.eval()  # Set the model to evaluation mode
for images, _ in test_loader:
    images = [img.to(device) for img in images]  # Ensure images are on the correct device
    detections = trained_model(images)  # Pass through the model
    for detection in detections:
        lego_count = count_legos(detection, confidence_threshold=0.5)
        print(f"Detected {lego_count} LEGO pieces.")