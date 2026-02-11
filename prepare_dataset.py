import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def process_dataset(base_dir, output_pos, output_neg):
    annotations_dir = os.path.join(base_dir, "Annotations")
    images_dir = os.path.join(base_dir, "JPEGImages")
    
    if not os.path.exists(annotations_dir) or not os.path.exists(images_dir):
        print(f"Skipping {base_dir} (missing Annotations or JPEGImages)")
        return

    os.makedirs(output_pos, exist_ok=True)
    os.makedirs(output_neg, exist_ok=True)

    xml_files = sorted(Path(annotations_dir).glob("*.xml"))
    print(f"Processing {len(xml_files)} files in {base_dir}...")

    count_pos = 0
    count_neg = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find("filename").text
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            # Try to find with different extension if original fails
            stem = xml_file.stem
            potential_files = list(Path(images_dir).glob(f"{stem}.*"))
            if potential_files:
                image_path = str(potential_files[0])
            else:
                print(f"Warning: Image not found for {xml_file.name}")
                continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        has_person = False
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name == "person":
                has_person = True
                bndbox = obj.find("bndbox")
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))
                
                # Crop
                # Ensure coordinates are within image bounds
                height, width = img.shape[:2]
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)
                
                if xmax > xmin and ymax > ymin:
                    crop = img[ymin:ymax, xmin:xmax]
                    if crop.size > 0:
                        # Resize to 64x128 for HOG
                        crop_resized = cv2.resize(crop, (64, 128))
                        output_filename = os.path.join(output_pos, f"pos_{count_pos:05d}.jpg")
                        cv2.imwrite(output_filename, crop_resized)
                        count_pos += 1

        if not has_person:
            # Save strictly negative images (no person at all)
            output_filename = os.path.join(output_neg, f"neg_{count_neg:05d}.jpg")
            cv2.imwrite(output_filename, img)
            count_neg += 1

    print(f"Finished {base_dir}: {count_pos} positive crops, {count_neg} negative images.")

def main():
    base_data = "data"
    train_dir = os.path.join(base_data, "Train")
    test_dir = os.path.join(base_data, "Test")

    print("Starting dataset preparation...")

    # Process Train
    process_dataset(train_dir, 
                   os.path.join(train_dir, "pos"), 
                   os.path.join(train_dir, "neg"))

    # Process Test
    process_dataset(test_dir, 
                   os.path.join(test_dir, "pos"), 
                   os.path.join(test_dir, "neg"))
                   
    print("\nDone! Dataset organized for HOG training.")
    print("Next step: Run ./build/train_hog_svm")

if __name__ == "__main__":
    main()
