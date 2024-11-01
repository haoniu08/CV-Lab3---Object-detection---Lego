from torch.utils.data import DataLoader
from lego_dataset import LegoDataset
from torchvision.transforms import ToTensor, Resize, Compose

def get_dataloaders(train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, test_img_dir, test_ann_dir, batch_size=4):
    # Define the transformations
    transform = Compose([
        Resize((600, 600)),
        ToTensor()
    ])

    # Create dataset instances
    train_dataset = LegoDataset(train_img_dir, train_ann_dir, transforms=transform)
    val_dataset = LegoDataset(val_img_dir, val_ann_dir, transforms=transform)
    test_dataset = LegoDataset(test_img_dir, test_ann_dir, transforms=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader, test_loader