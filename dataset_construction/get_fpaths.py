import os
from tqdm import tqdm
from multiprocessing import Pool


def get_file_lists(root_dir):
    """Recursively find .tif and .csv files in nested directories."""
    tif_files = []
    csv_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            fpath = os.path.join(dirpath, filename)
            if filename.endswith(".tif"):
                tif_files.append(fpath)
            elif filename.endswith(".csv"):
                csv_files.append(fpath)

    return tif_files, csv_files


def process_chunk(chunk):
    """Process a chunk of dirs."""
    for dir in tqdm(chunk):
        tif_files, csv_files = get_file_lists(dir)
        # write to a file
        with open(f"data/metadata/allclear_dataset_file_list_tif_{dir.replace('/', '_')}.txt", "a") as f:
            # wipe the file
            f.truncate(0)
            f.write("\n".join(tif_files))
        with open(f"data/metadata/allclear_dataset_file_list_csv_{dir.replace('/', '_')}.txt", "a") as f:
            # wipe the file
            f.truncate(0)
            f.write("\n".join(csv_files))

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
    num_workers = 50
    batch_process(dirs, n_cores=num_workers)

    # read from all the files and aggregate the file lists
    tif_files = []
    csv_files = []
    for dir in dirs:
        with open(f"data/metadata/allclear_dataset_file_list_tif_{dir.replace('/', '_')}.txt", "r") as f:
            tif_files.extend(f.read().splitlines())
        with open(f"data/metadata/allclear_dataset_file_list_csv_{dir.replace('/', '_')}.txt", "r") as f:
            csv_files.extend(f.read().splitlines())
    # write to a file
    with open("data/metadata/allclear_tif_files_path_list_final_version.txt", "w") as f:
        f.write("\n".join(tif_files))
    with open("data/metadata/allclear_csv_files_path_list_final_version.txt", "w") as f:
        f.write("\n".join(csv_files))

    # remove the intermediate files
    for dir in dirs:
        os.remove(f"data/metadata/allclear_dataset_file_list_tif_{dir.replace('/', '_')}.txt")
        os.remove(f"data/metadata/allclear_dataset_file_list_csv_{dir.replace('/', '_')}.txt")

    print(f"Found {len(tif_files)} .tif files and {len(csv_files)} .csv files in {root_directory}.")