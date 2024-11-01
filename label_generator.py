import os
import xml.etree.ElementTree as ET
import cv2

def convert_xml_to_yolo(xml_file, output_dir, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_data = []
    for obj in root.findall("object"):
        class_id = 0  # Use 0 for the "lego" class
        xml_box = obj.find("bndbox")
        xmin = int(xml_box.find("xmin").text)
        ymin = int(xml_box.find("ymin").text)
        xmax = int(xml_box.find("xmax").text)
        ymax = int(xml_box.find("ymax").text)

        # Convert to YOLO format (normalized)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Save YOLO formatted data
    txt_filename = os.path.join(output_dir, os.path.basename(xml_file).replace(".xml", ".txt"))
    with open(txt_filename, "w") as f:
        f.write("\n".join(yolo_data))

# Paths to your images and XML files
dataset_path = "/Users/tianyu/Documents/NEU/2024 Fall/CS5330/CV-Lab3---Object-detection---Lego/lego_dataset"
subdirs = ["train", "val", "test"]

for subdir in subdirs:
    image_dir = os.path.join(dataset_path, subdir, "images")
    annotation_dir = os.path.join(dataset_path, subdir, "annotations")
    label_dir = os.path.join(dataset_path, subdir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    # Process each XML file
    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(annotation_dir, xml_file)
            img_path = os.path.join(image_dir, xml_file.replace(".xml", ".jpg"))

            # Load image to get dimensions
            image = cv2.imread(img_path)
            if image is None:
                print(f"Image not found for {img_path}, skipping.")
                continue
            img_height, img_width = image.shape[:2]

            # Convert and save YOLO formatted labels
            convert_xml_to_yolo(xml_path, label_dir, img_width, img_height)
