import os
import subprocess
import concurrent.futures
from tqdm import tqdm

def rsync_dir(dir_name):
    src_dir = f"/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/{dir_name}"
    dest_dir = f"/scratch/allclear/dataset_v3/dataset_30k_v4/"
    # dest_dir = f"/home/hz477/allclear_backup/dataset_v3/dataset_30k_v4/"
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["rsync", "-ah", "--delete", src_dir, dest_dir], check=True)

SELECTED_ROIS_FNAME = "test_4k.txt"
# SELECTED_ROIS_FNAME = "dataset_500.txt"
# SELECTED_ROIS_FNAME = "train_2k.txt"
# SELECTED_ROIS_FNAME = "train_9k.txt"
with open(f"/share/hariharan/cloud_removal/metadata/v3/{SELECTED_ROIS_FNAME}") as f:
    SELECTED_ROIS = f.read().splitlines()

# SELECTED_ROIS = os.listdir("/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4")

num_jobs = 16
# num_jobs = 32
# num_jobs = 96

with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
    list(tqdm(executor.map(rsync_dir, SELECTED_ROIS), total=len(SELECTED_ROIS)))