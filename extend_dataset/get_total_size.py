import os
import math
from tqdm import tqdm

def get_total_size(root_dir):
    """Calculate the total size of a directory and its subdirectories."""
    total_size = 0

    for dirpath, _, filenames in tqdm(os.walk(root_dir), desc="Calculating size"):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
    return total_size

# root_directory = "/scratch/allclear/dataset_v3/dataset_30k_v4"
root_directory = "/home/hz477/allclear_dataset_final_version"
total_size_bytes = get_total_size(root_directory)

def convert_size(size_bytes):
    """Convert bytes to a human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

print(f"Total size: {convert_size(total_size_bytes)}")
