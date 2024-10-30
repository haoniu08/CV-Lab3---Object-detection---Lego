import os
import shutil
import random
from tqdm import tqdm


def split_dataset(img_dir, xml_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split the dataset into train, val, and test sets
    """
    # Create directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, 'annotations'), exist_ok=True)

    # Get all image files
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"Found {len(img_files)} images")

    # Shuffle the dataset
    random.seed(42)  # For reproducibility
    random.shuffle(img_files)

    # Calculate split sizes
    total_size = len(img_files)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Split the dataset
    train_files = img_files[:train_size]
    val_files = img_files[train_size:train_size + val_size]
    test_files = img_files[train_size + val_size:]

    # Function to copy files
    def copy_files(file_list, split_name):
        print(f"\nProcessing {split_name} split...")
        for img_file in tqdm(file_list):
            # Get corresponding XML file name
            xml_file = os.path.splitext(img_file)[0] + '.xml'

            # Copy image
            shutil.copy2(
                os.path.join(img_dir, img_file),
                os.path.join(output_base_dir, split_name, 'images', img_file)
            )

            # Copy XML
            shutil.copy2(
                os.path.join(xml_dir, xml_file),
                os.path.join(output_base_dir, split_name, 'annotations', xml_file)
            )

    # Process each split
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    # Print summary
    print("\nDataset splitting completed:")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")

    return {
        'train_size': len(train_files),
        'val_size': len(val_files),
        'test_size': len(test_files)
    }


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    source_img_dir = "imgs- 500 - manual/imgs"  # Your folder with 500 images
    source_xml_dir = "imgs- 500 - manual/annotations-modified"  # Your folder with 500 XML files
    output_dir = "lego_dataset"  # Where to save the split datasets

    stats = split_dataset(source_img_dir, source_xml_dir, output_dir)