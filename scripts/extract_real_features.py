"""
Extract image features from the real dataset in /dataset folder.
Processes images from dataset/{class}/ and generates feature_extraction.csv.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl

# Get workspace root
workspace_root = Path(__file__).parent.parent
dataset_path = workspace_root / "dataset"
output_dir = workspace_root / "data" / "tabular"
output_dir.mkdir(parents=True, exist_ok=True)

# Distribution parameters for synthetic weight/size
dist_params = {
    "banana": {"weight": (120, 15), "size": (18, 2)},
    "carrot": {"weight": (60, 10), "size": (15, 2.5)},
    "cucumber": {"weight": (300, 40), "size": (20, 3)},
    "mandarina": {"weight": (80, 12), "size": (6.5, 0.8)},
    "tomato": {"weight": (100, 15), "size": (7, 1)}
}


def extract_features(image, image_class):
    """Extract features from a single image."""
    features = {}
    
    # Extract BGR channel statistics
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    
    features["blue_mean"] = float(np.mean(blue))
    features["blue_std"] = float(np.std(blue))
    features["green_mean"] = float(np.mean(green))
    features["green_std"] = float(np.std(green))
    features["red_mean"] = float(np.mean(red))
    features["red_std"] = float(np.std(red))
    
    # Extract 8x8 grayscale features
    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_flat = gray_small.reshape(-1).astype("float32")
    for i, val in enumerate(gray_flat):
        features[f"gray_{i:03d}"] = float(val)
    
    # Sample weight and size from class distribution
    params = dist_params[image_class]
    features["weight"] = float(np.random.normal(params["weight"][0], params["weight"][1]))
    features["size"] = float(np.random.normal(params["size"][0], params["size"][1]))
    
    features["class"] = image_class
    return features


def main():
    """Main extraction pipeline."""
    # Get list of class directories
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"Error: No class directories found in {dataset_path}")
        sys.exit(1)
    
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    rows = []
    total_images = 0
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([
            f for f in class_dir.iterdir() 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        print(f"Processing {class_name}: {len(image_files)} images", flush=True)
        
        for i, image_path in enumerate(image_files):
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"  Warning: Failed to read {image_path.name}")
                    continue
                
                features = extract_features(img, class_name)
                rows.append(features)
                total_images += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(image_files)}", flush=True)
            
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
                continue
        
        print(f"  Complete: {total_images} total images extracted\n", flush=True)
    
    # Create DataFrame and save
    if rows:
        df = pl.DataFrame(rows)
        output_path = output_dir / "feature_extraction.csv"
        df.write_csv(str(output_path))
        print(f"\n✓ Saved {len(rows)} feature vectors to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
    else:
        print("Error: No features extracted!")
        sys.exit(1)


if __name__ == "__main__":
    main()
