import os

data_dir = "data/raw"

for split in ["train", "val"]:
    path = os.path.join(data_dir, split)
    print("\n", split.upper())

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            print(folder, ":", len(os.listdir(folder_path)))