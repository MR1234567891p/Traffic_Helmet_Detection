# 5) DATASET INSPECTOR 
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


# CONFIG
RAW_IMAGES_DIR = r"F:\7th sem lab\4709_HELMET_YOLOV8\images"
RAW_ANN_DIR = r"F:\7th sem lab\4709_HELMET_YOLOV8\annotations"
YOLO_DATASET_DIR = r"F:\7th sem lab\4709_HELMET_YOLOV8\images\helmet_dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DIVIDER = "─" * 55


def normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_tag = root.find("filename")
        filename = filename_tag.text.strip() if filename_tag is not None else ""

        size = root.find("size")
        width = int(size.find("width").text) if size is not None else 0
        height = int(size.find("height").text) if size is not None else 0

        objects = []
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            if name_tag is None:
                continue
            cls = normalize(name_tag.text)

            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                bw = xmax - xmin
                bh = ymax - ymin
            else:
                bw = bh = 0

            objects.append({"class": cls, "bw": bw, "bh": bh})

        return filename, width, height, objects, None

    except Exception as e:
        return "", 0, 0, [], str(e)


def inspect_raw():
    print(f"\n{'═'*55}")
    print("  RAW DATASET INSPECTION")
    print(f"{'═'*55}")

    all_images = [
        f for f in os.listdir(RAW_IMAGES_DIR)
        if Path(f).suffix.lower() in IMAGE_EXTS
    ]
    image_stems = {Path(f).stem for f in all_images}
    ext_counts = Counter(Path(f).suffix.lower() for f in all_images)

    print(f"\n📁 IMAGE FOLDER: {RAW_IMAGES_DIR}")
    print(f"{DIVIDER}")
    print(f"  Total images       : {len(all_images)}")
    for ext, cnt in sorted(ext_counts.items()):
        print(f"  {ext:<10} : {cnt}")

    all_xmls = [
        f for f in os.listdir(RAW_ANN_DIR)
        if f.endswith(".xml")
    ]
    xml_stems = {Path(f).stem for f in all_xmls}

    print(f"\n📄 ANNOTATION FOLDER: {RAW_ANN_DIR}")
    print(f"{DIVIDER}")
    print(f"  Total XML files    : {len(all_xmls)}")

    imgs_with_ann = image_stems & xml_stems
    imgs_no_ann = image_stems - xml_stems
    anns_no_image = xml_stems - image_stems

    print(f"\n🔗 CROSS-CHECK")
    print(f"{DIVIDER}")
    print(f"  Images with annotation   : {len(imgs_with_ann)}")
    print(f"  Images WITHOUT annotation: {len(imgs_no_ann)}")
    print(f"  Annotations without image: {len(anns_no_image)}")

    class_counter = Counter()
    objects_per_image = []
    images_per_class = defaultdict(set)
    parse_errors = []
    empty_annotations = []

    for xml_file in sorted(os.listdir(RAW_ANN_DIR)):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(RAW_ANN_DIR, xml_file)
        _, _, _, objects, err = parse_xml(xml_path)

        if err:
            parse_errors.append((xml_file, err))
            continue

        if not objects:
            empty_annotations.append(xml_file)
        else:
            objects_per_image.append(len(objects))
            for obj in objects:
                class_counter[obj["class"]] += 1
                images_per_class[obj["class"]].add(xml_file)

    total_objects = sum(class_counter.values())

    print(f"\n📦 OBJECT / CLASS ANALYSIS")
    print(f"{DIVIDER}")
    print(f"  Total annotated objects  : {total_objects}")
    print(f"  Unique classes           : {len(class_counter)}")

    if objects_per_image:
        print(f"  Avg objects/image        : {sum(objects_per_image)/len(objects_per_image):.1f}")
        print(f"  Max objects in one image : {max(objects_per_image)}")
        print(f"  Min objects in one image : {min(objects_per_image)}")

    print(f"\n  CLASS BREAKDOWN:")
    print(f"  {'Class':<20} {'Objects':>8} {'Images':>8}")
    print(f"  {'-'*40}")
    for cls, cnt in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"  {cls:<20} {cnt:>8} {len(images_per_class[cls]):>8}")


def inspect_yolo():
    if not os.path.exists(YOLO_DATASET_DIR):
        print(f"\n⚠️ YOLO_DATASET_DIR not found: {YOLO_DATASET_DIR}")
        return

    print(f"\n\n{'═'*55}")
    print("  PREPARED YOLO DATASET INSPECTION")
    print(f"{'═'*55}")
    print(f"  Path: {YOLO_DATASET_DIR}\n")

    for split in ["train", "test"]:
        img_dir = os.path.join(YOLO_DATASET_DIR, "images", split)
        lbl_dir = os.path.join(YOLO_DATASET_DIR, "labels", split)

        if not os.path.exists(img_dir):
            continue

        images = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMAGE_EXTS]
        labels = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")] if os.path.exists(lbl_dir) else []

        print(f"{split.upper()} -> Images: {len(images)}, Labels: {len(labels)}")

    yaml_path = os.path.join(YOLO_DATASET_DIR, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"\ndata.yaml found: {yaml_path}")
    else:
        print("\ndata.yaml not found.")


if __name__ == "__main__":
    inspect_raw()
    inspect_yolo()