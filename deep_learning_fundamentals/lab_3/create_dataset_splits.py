import os
import csv
import random
from os.path import join


# set random seed for reproducibility
random.seed(42)

# Define the classes (same as in the combination script)
final_classes = {
    'organic': 0,
    'battery': 1,
    'glass': 2,
    'metal': 3,
    'paper': 4,
    'cardboard': 5,
    'plastic': 6,
    'textiles': 7,
    'trash': 8,
}


# Paths
dataset_root = './datasets/combined_waste_dataset'

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure ratios add up to 1
assert train_ratio + val_ratio + test_ratio == 1.0


def create_csv_files(out_root):
    """Create CSV files for train, val, test splits."""
    os.makedirs(out_root, exist_ok=True)
    for split in ['train', 'val', 'test']:
        csv_path = join(out_root, f'{split}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'class'])


def split_and_write_csvs(comb_root, out_root, classes,
                         train_r, val_r):
    """Split images and write to CSV files."""
    csv_files = {}
    for split in ['train', 'val', 'test']:
        csv_path = join(out_root, f'{split}.csv')
        csv_files[split] = open(csv_path, 'a', newline='', encoding='utf-8')
    
    writers = {split: csv.writer(f) for split, f in csv_files.items()}
    
    try:
        for cls in classes.keys():
            cls_dir = join(comb_root, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Class directory {cls_dir} does not exist. "
                      "Skipping.")
                continue
            
            # Get all image files
            images = [f for f in os.listdir(cls_dir)
                      if os.path.isfile(join(cls_dir, f))]
            if not images:
                print(f"Warning: No images found in {cls_dir}. Skipping.")
                continue
            
            # Shuffle the images
            random.shuffle(images)
            n = len(images)
            
            # Calculate split indices
            train_end = int(train_r * n)
            val_end = int((train_r + val_r) * n)
            
            train_imgs = images[:train_end]
            val_imgs = images[train_end:val_end]
            test_imgs = images[val_end:]
            
            # Write to CSVs
            splits_data = [('train', train_imgs), ('val', val_imgs),
                           ('test', test_imgs)]
            for split, imgs in splits_data:
                for img in imgs:
                    rel_path = join(cls, img)
                    writers[split].writerow([rel_path, cls])
            
            print(f"Class {cls}: {len(train_imgs)} train, "
                  f"{len(val_imgs)} val, {len(test_imgs)} test images.")
    finally:
        for f in csv_files.values():
            f.close()


if __name__ == "__main__":
    # Create CSV files
    create_csv_files(dataset_root)
    
    # Split and write
    split_and_write_csvs(dataset_root, dataset_root, final_classes,
                         train_ratio, val_ratio)
    
    print("Dataset splitting completed!")
