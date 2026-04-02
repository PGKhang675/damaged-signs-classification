import os
from pathlib import Path
import cv2

def count_total_boxes(label_folder: str) -> int:
    """
    Count total number of YOLO bounding boxes in all label files.

    Each line in a YOLO annotation file corresponds to one bounding box.
    """

    total_boxes = 0
    label_folder = Path(label_folder)

    for label_file in label_folder.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            total_boxes += len(lines)

    return total_boxes


def convert_yolo_to_pixels(x_center, y_center, width, height, img_w, img_h):
    """
    Convert normalized YOLO bbox to pixel coordinates.
    """

    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2


def extract_boxes(image_folder: str, label_folder: str, output_folder: str, min_size: int = 50):
    """
    Crop each bounding box into a separate image.
    Boxes with width or height < min_size pixels are excluded.

    Prints:
    - total boxes found
    - boxes excluded (too small)
    - boxes extracted
    """

    from pathlib import Path
    import cv2

    image_folder = Path(image_folder)
    label_folder = Path(label_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    total_boxes_found = 0
    small_boxes_excluded = 0
    boxes_extracted = 0

    for label_file in label_folder.glob("*.txt"):

        image_name = label_file.stem
        image_path = None

        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_folder / f"{image_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        with open(label_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for i, line in enumerate(lines):

            parts = line.split()
            if len(parts) != 5:
                continue

            total_boxes_found += 1

            cls, x, y, w, h = map(float, parts)

            x_center = x * img_w
            y_center = y * img_h
            width = w * img_w
            height = h * img_h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < min_size or box_h < min_size:
                small_boxes_excluded += 1
                continue

            crop = img[y1:y2, x1:x2]

            output_name = f"{image_name}_box{i}_class{int(cls)}.jpg"
            cv2.imwrite(str(output_folder / output_name), crop)

            boxes_extracted += 1

    print("Total boxes found:", total_boxes_found)
    print("Small boxes excluded:", small_boxes_excluded)
    print("Boxes extracted:", boxes_extracted)

# Example usage
if __name__ == "__main__":

    image_folder = "./data/signs-detected/train/images"
    label_folder = "./data/signs-detected/train/labels"
    output_folder = "./data/cropped_boxes"

    extract_boxes(image_folder, label_folder, output_folder)