import os
import math
from tqdm import tqdm
from multiprocessing import Pool

def get_total_size(root_dir):
    """Calculate the total size of a directory and its subdirectories."""
    total_size = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
    return total_size

def convert_size(size_bytes):
    """Convert bytes to a human-readable format."""
    print(size_bytes)
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def process_chunk(chunk):
    """Process a chunk of dirs."""
    for dir in tqdm(chunk):
        total_size_bytes = get_total_size(dir)
        # write to a file
        with open(f"data/metadata/allclear_dataset_total_size_{dir.replace('/', '_')}.txt", "a") as f:
            # wipe the file
            f.truncate(0)
            f.write(f"{total_size_bytes}")

def split_list(input_list, n_chunks):
    """Split the input list into n_chunks roughly equal parts."""
    chunk_size = len(input_list) // n_chunks + 1
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def batch_process(dirs, n_cores=None):
    """Batch process dirs using multiple processes with a tqdm progress bar."""
    n_cores = n_cores
    chunks = split_list(dirs, n_cores)

    with Pool(processes=n_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_chunk, chunks), total=len(chunks), desc="Overall Progress"):
            pass

if __name__ == "__main__":
    # root_directory = "/scratch/allclear/dataset_v3/dataset_30k_v4"
    root_directory = "/home/hz477/allclear_dataset_final_version"
    dirs = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    num_workers = 200

    batch_process(dirs, n_cores=num_workers)

    # read from all the files and sum up the total size
    total_size = 0
    for dir in tqdm(dirs, desc="Aggregating total size"):
        with open(f"data/metadata/allclear_dataset_total_size_{dir.replace('/', '_')}.txt", "r") as f:
            content = f.read()
            total_size += int(content)
    print(f"Total size: {convert_size(total_size)}")