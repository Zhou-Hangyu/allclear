import os
import subprocess
import concurrent.futures
import argparse
from tqdm import tqdm

def sync_directories(source_path, dest_path, num_workers=16):
    """
    Synchronize files from source directory to destination directory using parallel rsync.
    
    Args:
        source_path (str): Path to source directory
        dest_path (str): Path to destination directory
        num_workers (int, optional): Number of parallel workers. Defaults to 16.
    """
    def rsync_file(src_file):
        # Replace source path with destination path to maintain structure
        dest_file = src_file.replace(source_path, dest_path)
        dest_dir = os.path.dirname(dest_file)
        os.makedirs(dest_dir, exist_ok=True)
        subprocess.run(["rsync", "-ah", "--delete", src_file, dest_dir], check=True)
    
    def collect_file_paths(root_dir):
        """Recursively collect all file paths from the given directory."""
        file_paths = []
        print(f"Collecting files from {root_dir}...")
        for dirpath, _, filenames in tqdm(os.walk(root_dir)):
            for filename in filenames:
                file_paths.append(os.path.join(dirpath, filename))
        return file_paths

    # Validate paths
    if not os.path.exists(source_path):
        raise ValueError(f"Source path does not exist: {source_path}")
    
    # Collect files
    file_paths = collect_file_paths(source_path)
    
    if not file_paths:
        print("No files found in source directory.")
        return
    
    print(f"Starting sync of {len(file_paths)} files using {num_workers} workers...")
    
    # Run parallel sync
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(rsync_file, file_paths), total=len(file_paths)))
    
    print("Sync completed successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel rsync for dataset transfer")
    parser.add_argument("--source", "-s", required=True, help="Source directory path")
    parser.add_argument("--dest", "-d", required=True, help="Destination directory path")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of parallel workers (default: 16)")
    return parser.parse_args()

if __name__ == "__main__":
    """
    Example usage:
        python parallel_data_sync.py -s /path/to/source -d /path/to/dest -w 16
    """
    args = parse_args()
    sync_directories(
        source_path=args.source,
        dest_path=args.dest,
        num_workers=args.workers
    )