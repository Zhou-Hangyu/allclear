import os
import subprocess
import concurrent.futures
from tqdm import tqdm

def rsync_file(src_file):
    # Replace the base source path with the destination path
    dest_file = src_file.replace(
        "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4",
        "/share/hariharan/cloud_removal/MultiSensor/allclear_dataset"
    )

    # Create the destination directory if it doesn't exist
    dest_dir = os.path.dirname(dest_file)
    os.makedirs(dest_dir, exist_ok=True)

    # Perform rsync for the individual file
    subprocess.run(["rsync", "-ah", "--delete", src_file, dest_dir], check=True)

def collect_file_paths(root_dir):
    """Recursively collect all file paths from the given directory."""
    file_paths = []
    for dirpath, _, filenames in tqdm(os.walk(root_dir)):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths

# Source directory
src_dir = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4"

# Collect all files from the source directory
file_paths = collect_file_paths(src_dir)

# Number of parallel jobs
num_jobs = 16

# Use ProcessPoolExecutor for parallel rsync
with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
    list(tqdm(executor.map(rsync_file, file_paths), total=len(file_paths)))
