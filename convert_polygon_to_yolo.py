import os

# Set the paths
images_dir = "dataset/images"   # Your image folders: train/val
labels_dir = "labels"           # Your current polygon txt files: train/val
output_dir = "labels_yolo"      # New folder to store YOLO format txt files

# Create output folders if they don't exist
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

def polygon_to_yolo(polygon_coords, image_width, image_height):
    """
    Convert polygon coordinates (x1 y1 x2 y2 x3 y3 x4 y4) 
    to YOLO bbox format: x_center y_center width height
    All values normalized (0-1).
    """
    xs = polygon_coords[0::2]
    ys = polygon_coords[1::2]
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return x_center, y_center, width, height

# Example image dimensions (replace with actual if images differ)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

for split in ["train", "val"]:
    split_labels_dir = os.path.join(labels_dir, split)
    split_output_dir = os.path.join(output_dir, split)
    
    for filename in os.listdir(split_labels_dir):
        if not filename.endswith(".txt"):
            continue
        
        input_path = os.path.join(split_labels_dir, filename)
        output_path = os.path.join(split_output_dir, filename)
        
        with open(input_path, "r") as f:
            lines = f.readlines()
        
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 9:
                continue  # Skip lines that don't have class + 8 coordinates
            
            class_id = parts[0]
            coords = list(map(float, parts[1:]))
            
            x_center, y_center, width, height = polygon_to_yolo(coords, IMAGE_WIDTH, IMAGE_HEIGHT)
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write YOLO-formatted file
        with open(output_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("Conversion to YOLO format complete!")
