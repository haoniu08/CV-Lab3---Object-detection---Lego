import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def simplify_xml_labels(input_file, output_file):
    """
    Read an XML file and replace all part numbers in <name> tags with 'lego'
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Find all <name> tags within <object> tags and replace content with 'lego'
    for obj in root.findall('.//object/name'):
        obj.text = 'lego'
    
    # Write the modified XML
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def process_xml_files(input_dir, output_dir, num_files=5):
    """
    Process a specified number of XML files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of XML files
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    # Process specified number of files
    for xml_file in tqdm(xml_files[:num_files]):
        input_path = os.path.join(input_dir, xml_file)
        output_path = os.path.join(output_dir, xml_file)
        
        simplify_xml_labels(input_path, output_path)
        
        # Print before and after for verification
        print(f"\nProcessed: {xml_file}")
        print("First few lines of the modified file:")
        with open(output_path, 'r') as f:
            print('\n'.join(f.readlines()[:15]))  # Print first 15 lines

# Test the script with a few files
input_dir = "/Users/haoniu/Desktop/CS5330 - Pattern Recognition/Lab3/imgs- 500 - manual/annotations"    # Replace with your input directory
output_dir = "/Users/haoniu/Desktop/CS5330 - Pattern Recognition/Lab3/imgs- 500 - manual/annotations-modified"  # Replace with your output directory

process_xml_files(input_dir, output_dir, num_files=500)