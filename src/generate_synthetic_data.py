import cv2
import numpy as np
import os
import random
import yaml

# Setup folders
base_dir = 'datasets/syntethic_balls'
if os.path.exists(base_dir):
  import shutil
  shutil.rmtree(base_dir) # Make sure its empty

os.makedirs(f'{base_dir}/train/images', exist_ok=True)
os.makedirs(f'{base_dir}/train/labels', exist_ok=True)
os.makedirs(f'{base_dir}/test/images', exist_ok=True)
os.makedirs(f'{base_dir}/test/labels', exist_ok=True)

# Ball generating function
def create_dataset(subset, count=50):
  print(f"Generating {count} images for {subset}...")

  for i in range(count):
    # Create black canvas for background
    img_size = 640
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Create random balls/circles
    # Random radius between 20-100px
    radius = random.randint(20,100)

    # Calculate center of circle
    # Make sure it stays in the image
    center_x = random.randint(radius, img_size - radius)
    center_y = random.randint(radius, img_size - radius)

    # Draw RED circle
    cv2.circle(img, (center_x, center_y), radius, (0,0,255), -1)

    # Save the image
    img_filename = f'{base_dir}/{subset}/images/ball_{i}.jpg'
    cv2.imwrite(img_filename, img)

    # Calculate YOLO label
    # Format: class_id x_center y_center width height
    # Class ID 0 = "Red Ball"
    norm_x = center_x / img_size
    norm_y = center_y / img_size
    norm_w = (radius * 2) / img_size
    norm_h = (radius * 2) / img_size

    # Save label
    lbl_filename = f'{base_dir}/{subset}/labels/ball_{i}.txt'
    with open(lbl_filename, 'w') as f:
      f.write(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

def main_loop(num_samples):
  # Generate images
    create_dataset('train', samples*0.7)
    create_dataset('test', samples*0.3)

    # Create YAML config file
    data_yaml = {
        'path': os.path.abspath(base_dir),
        'train': 'train/images',
        'val': 'test/images',
        'names': {0: 'Red Ball'}
    }

    with open(f'{base_dir}/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    print("Dataset created successfully!")
    print(f"Config saved at: {base_dir}/data.yaml")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for YOLOv8.")

    parser.add_argument(
        "--samples", 
        type=int,           
        default=1000,       
        help="The number of synthetic images to generate."
    )

    args = parser.parse_args()

    main_loop(num_samples=args.samples)