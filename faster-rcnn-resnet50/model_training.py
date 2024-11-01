import torch
from torch.utils.data import DataLoader
from model import get_model  # Import model definition


def train_model(model, data_loader, val_loader, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = [img.to(device) for img in images]  # Move images to device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            loss_dict = model(images, targets)
            # Sum of losses
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}")

    print("Training complete")
    return model