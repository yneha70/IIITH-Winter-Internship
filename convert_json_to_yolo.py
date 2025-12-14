import os

# Paths
annotations_dir = "annotations_json"  # contains 'train' and 'val'
images_dir = "dataset/images"          # contains 'train' and 'val' images
labels_dir = "labels"                   # output YOLO txt files

# Ensure output folders exist
os.makedirs(os.path.join(labels_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)

def convert_json_to_yolo(json_folder, image_folder, label_folder):
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            # Check if corresponding image exists
            img_extensions = [".jpg", ".jpeg", ".png"]
            image_found = False
            for ext in img_extensions:
                if os.path.exists(os.path.join(image_folder, base_name + ext)):
                    image_found = True
                    break
            if not image_found:
                print(f"Skipping {filename}: no corresponding image found")
                continue

            txt_path = os.path.join(label_folder, base_name + ".txt")

            # Read JSON content (line by line)
            lines_written = 0
            with open(json_path, "r", encoding="utf-8") as f, open(txt_path, "w", encoding="utf-8") as out_f:
                for line in f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
                        lines_written += 1

            # Remove empty files
            if lines_written == 0:
                os.remove(txt_path)

# Convert training
convert_json_to_yolo(
    os.path.join(annotations_dir, "train"),
    os.path.join(images_dir, "train"),
    os.path.join(labels_dir, "train")
)

# Convert validation
convert_json_to_yolo(
    os.path.join(annotations_dir, "val"),
    os.path.join(images_dir, "val"),
    os.path.join(labels_dir, "val")
)

print("All YOLO .txt files created successfully!")
