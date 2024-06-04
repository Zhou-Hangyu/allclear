import os
import subprocess
import concurrent.futures
from tqdm import tqdm
import json

def rsync_file(fpath):
    src_file = fpath.replace("/scratch/allclear/dataset_v3/dataset_30k_v4", "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4")
    dest_dir = os.path.dirname(fpath)
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["rsync", "-ah", "--delete", src_file, dest_dir], check=True)

def extract_file_paths(dataset):
    file_paths = []
    for data in dataset.values():
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[1], str):
                        file_paths.append(item[1])
                        if "s2_toa" in item[1]:
                            file_paths.append(item[1].replace("s2_toa", "cld_shdw"))
    return file_paths

SELECTED_FILES_FNAME = "s2p_tx3_train_19k_d10k_v3.json"

with open(f"/share/hariharan/cloud_removal/metadata/v4/{SELECTED_FILES_FNAME}") as f:
    file_paths = extract_file_paths(json.load(f))

num_jobs = 16
# num_jobs = 32
# num_jobs = 96

with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
    list(tqdm(executor.map(rsync_file, file_paths), total=len(file_paths)))