import os
import shutil
import random

raw_dir = "data/raw/train"
val_dir = "data/raw/val"

split_ratio = 0.2  # 20% validation

for class_name in os.listdir(raw_dir):
    class_path = os.path.join(raw_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    val_count = int(len(images) * split_ratio)

    val_images = images[:val_count]

    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_dir, img)
        shutil.move(src, dst)

    print(f"{class_name}: moved {val_count} images to val")