import os
import cv2

def crop_and_classify_yolo(image_folder, text_folder, output_folder):
    """
    Reads images and YOLO annotations, crops the bounding boxes, and saves
    them into class-specific folders.
    """
    
    # Define and create output directories
    damaged_dir = os.path.join(output_folder, 'damaged')
    healthy_dir = os.path.join(output_folder, 'healthy')
    os.makedirs(damaged_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)
    
    # Map class IDs to their corresponding output directories
    class_dirs = {
        '0': damaged_dir,
        '1': healthy_dir
    }
    
    # Common image extensions to check
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # Iterate through all text files in the annotation folder
    for txt_filename in os.listdir(text_folder):
        if not txt_filename.endswith('.txt'):
            continue
            
        base_name = os.path.splitext(txt_filename)[0]
        
        # Find the matching image file
        image_path = None
        for ext in valid_extensions:
            temp_path = os.path.join(image_folder, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break
                
        if not image_path:
            print(f"Warning: Could not find matching image for {txt_filename}")
            continue
            
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Read the annotations
        txt_path = os.path.join(text_folder, txt_filename)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            
        # Process each bounding box in the file
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = parts[0]
            
            # Skip if it's not class 0 or 1
            if class_id not in class_dirs:
                print(f"Skipping unknown class {class_id} in {txt_filename}")
                continue
                
            # YOLO format is: class_id, center_x, center_y, width, height (normalized)
            x_center, y_center, w, h = map(float, parts[1:5])
            
            # Convert normalized YOLO coordinates to absolute pixel coordinates
            abs_w = int(w * img_w)
            abs_h = int(h * img_h)
            
            x_min = int((x_center * img_w) - (abs_w / 2))
            y_min = int((y_center * img_h) - (abs_h / 2))
            x_max = x_min + abs_w
            y_max = y_min + abs_h
            
            # Ensure coordinates don't fall outside image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            
            # Ensure the crop area is valid (width and height > 0)
            if x_max <= x_min or y_max <= y_min:
                continue
                
            # Crop the image (NumPy array slicing: img[y_start:y_end, x_start:x_end])
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Construct the output filename and save
            # Using idx to ensure unique names if there are multiple boxes per image
            out_filename = f"{base_name}_crop{idx}.jpg"
            out_filepath = os.path.join(class_dirs[class_id], out_filename)
            
            cv2.imwrite(out_filepath, cropped_img)
            
    print(f"Processing complete. Cropped images saved to {output_folder}")

# --- Example Usage ---
if __name__ == "__main__":
    INPUT_IMAGES = "D:/Swinburne/Jan_2026/COS40007_AI-Engineer/Project/code/data/input/classified-damaged-signs/train/images"
    INPUT_LABELS = "D:/Swinburne/Jan_2026/COS40007_AI-Engineer/Project/code/data/input/classified-damaged-signs/train/labels"
    OUTPUT_FOLDER = "D:/Swinburne/Jan_2026/COS40007_AI-Engineer/Project/code/data/output/classified_cropped_boxes"

crop_and_classify_yolo(INPUT_IMAGES, INPUT_LABELS, OUTPUT_FOLDER)