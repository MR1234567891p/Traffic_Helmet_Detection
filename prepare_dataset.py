# DATASET PREPARATION SCRIPT
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# CONFIG
RAW_IMAGES_DIR = r"F:\7th sem lab\4709_HELMET_YOLOV8\images"
RAW_ANN_DIR = r"F:\7th sem lab\4709_HELMET_YOLOV8\annotations"
OUTPUT_DIR = r"F:\7th sem lab\4709_HELMET_YOLOV8\images\helmet_dataset"

FORCED_CLASSES = None 
# FORCED_CLASSES = ["helmet", "no_helmet", "motorcycle"]

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42


def normalize_class_name(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    return name


def get_image_size_from_xml(root):
    size = root.find("size")
    if size is None:
        return None, None
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    return width, height


def convert_bbox_to_yolo(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / w
    y_center = ((ymin + ymax) / 2.0) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    return x_center, y_center, bw, bh


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    width, height = get_image_size_from_xml(root)

    objects = []
    for obj in root.findall("object"):
        cls_name = normalize_class_name(obj.find("name").text)

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        objects.append({
            "class_name": cls_name,
            "bbox": (xmin, ymin, xmax, ymax)
        })

    return filename, width, height, objects


def discover_classes(annotation_files):
    class_names = set()

    for xml_file in tqdm(annotation_files, desc="Discovering classes"):
        _, _, _, objects = parse_xml(xml_file)
        for obj in objects:
            class_names.add(obj["class_name"])

    return sorted(list(class_names))


def make_dirs(base_dir):
    for split in ["train", "test"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)


def find_image_file(images_dir, filename_from_xml):
    xml_path = os.path.join(images_dir, filename_from_xml)
    if os.path.exists(xml_path):
        return xml_path

    stem = Path(filename_from_xml).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate

    return None


def write_label_file(label_path, width, height, objects, class_to_id):
    lines = []
    for obj in objects:
        cls_name = obj["class_name"]
        if cls_name not in class_to_id:
            continue

        x_center, y_center, bw, bh = convert_bbox_to_yolo(
            (width, height),
            obj["bbox"]
        )
        class_id = class_to_id[cls_name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def main():
    random.seed(RANDOM_SEED)

    annotation_files = sorted([
        os.path.join(RAW_ANN_DIR, f)
        for f in os.listdir(RAW_ANN_DIR)
        if f.endswith(".xml")
    ])

    if not annotation_files:
        raise FileNotFoundError(f"No XML files found in {RAW_ANN_DIR}")

    if FORCED_CLASSES is not None:
        class_names = [normalize_class_name(c) for c in FORCED_CLASSES]
    else:
        class_names = discover_classes(annotation_files)

    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    print("\nDetected classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    make_dirs(OUTPUT_DIR)

    train_files, test_files = train_test_split(
        annotation_files,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    splits = {
        "train": train_files,
        "test": test_files
    }

    for split_name, files in splits.items():
        print(f"\nProcessing {split_name}: {len(files)} files")

        for xml_file in tqdm(files, desc=f"{split_name} split"):
            filename, width, height, objects = parse_xml(xml_file)

            image_path = find_image_file(RAW_IMAGES_DIR, filename)
            if image_path is None:
                print(f"[WARNING] Image not found for XML: {xml_file}")
                continue

            img_name = Path(image_path).name
            img_stem = Path(image_path).stem

            dst_img = os.path.join(OUTPUT_DIR, "images", split_name, img_name)
            dst_lbl = os.path.join(OUTPUT_DIR, "labels", split_name, img_stem + ".txt")

            shutil.copy2(image_path, dst_img)
            write_label_file(dst_lbl, width, height, objects, class_to_id)

    # Use test set as val too, so YOLO can validate during training
    data_yaml = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("\nDone.")
    print(f"YOLO dataset created at: {OUTPUT_DIR}")
    print(f"data.yaml saved at: {yaml_path}")
    print("\nSplit summary:")
    print(f"Train: {len(train_files)}")
    print(f"Test : {len(test_files)}")


if __name__ == "__main__":
    main()