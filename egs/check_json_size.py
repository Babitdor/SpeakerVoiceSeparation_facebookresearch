import os
import json

current = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(current, "mydataset")

for split in ["cv", "tr", "tt"]:
    split_dir = os.path.join(json_dir, split)
    if not os.path.isdir(split_dir):
        print(f"Directory not found: {split_dir}")
        continue
    print(f"\nChecking JSON files in: {split_dir}")
    for fname in os.listdir(split_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(split_dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                print(f"{fname}: {len(data)} items")
            except Exception as e:
                print(f"{fname}: Error reading file ({e})")
