import requests
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import time
import argparse

# Configuration
BASE_URL = "http://allclear.cs.cornell.edu/dataset/allclear"
CHUNK_SIZE = 8192

def download_file(url, dest_path, show_progress=True):
    """Download a file with progress bar and return success status"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            return False

        total_size = int(response.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f:
            if show_progress:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def verify_file(file_path):
    """Verify if file is complete by trying to open it"""
    try:
        if file_path.suffix == '.gz':
            import gzip
            with gzip.open(file_path, 'rb') as f:
                f.read(1)  # Try reading first byte
        return True
    except:
        return False

def download_metadata():
    """Download metadata files"""
    metadata_dir = Path("metadata")
    metadata_dir.mkdir(exist_ok=True)
    filename = "metadata.tar.gz"
    url = f"{BASE_URL}/{filename}"
    dest_path = metadata_dir / filename
    print(f"Downloading metadata from {url} to {dest_path}")
    success = download_file(url, dest_path)

    if success and verify_file(dest_path):
        # Extract the tar.gz file
        try:
            import tarfile
            import shutil

            with tarfile.open(dest_path, 'r:gz') as tar:
                tar.extractall(path=metadata_dir)

            nested_dir = metadata_dir / "metadata"
            if nested_dir.exists() and nested_dir.is_dir():
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(metadata_dir))
                nested_dir.rmdir()
            print(f"Successfully downloaded and extracted {filename}")
            # Remove the tar.gz file after extraction
            dest_path.unlink()
        except Exception as e:
            print(f"Error extracting {filename}: {e}")
            if dest_path.exists():
                dest_path.unlink()
    elif success:
        print(f"Downloaded {filename} but verification failed")
        dest_path.unlink()
    else:
        print(f"Skipping {filename} - not found on server")


def load_roi_list():
    """Load and combine all ROI IDs from metadata files"""
    metadata_dir = Path("metadata")
    roi_ids = set()
    
    for filename in ["test_rois_3k.txt", "train_rois_19k.txt", "val_rois_1k.txt"]:
        file_path = metadata_dir / "rois" / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found")
            continue
            
        with open(file_path, 'r') as f:
            roi_ids.update(line.strip() for line in f)
    
    return sorted(list(roi_ids))

def download_roi_worker(roi_batch):
    """Worker function for parallel ROI downloads"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for roi_id in roi_batch:
        filename = f"{roi_id}.tar.gz"
        dest_path = data_dir / filename
        url = f"{BASE_URL}/data/{filename}"
        
        # Skip if already downloaded, verified, and extracted
        if (data_dir / roi_id).exists():
            continue
            
        # Remove if exists but invalid
        if dest_path.exists():
            dest_path.unlink()
            
        # Download file
        success = download_file(url, dest_path, show_progress=False)
        time.sleep(0.1)
        
        if success and verify_file(dest_path):
            # Extract the tar.gz file
            try:
                import tarfile
                with tarfile.open(dest_path, 'r:gz') as tar:
                    tar.extractall(path=data_dir)
                print(f"Successfully downloaded and extracted {filename}")
                # Remove the tar.gz file after extraction
                dest_path.unlink()
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                if dest_path.exists():
                    dest_path.unlink()
        elif success:
            print(f"Downloaded {filename} but verification failed")
            dest_path.unlink()
        else:
            print(f"Skipping {filename} - not found on server")

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Download dataset with configurable CPU cores')
    parser.add_argument('--cpus', type=int, default=8,
                       help='Number of CPU cores to use (default: 8)')
    args = parser.parse_args()
    
    # Calculate N_CORES using args.cpus
    n_cores = max(1, args.cpus - 1)  # Leave one core free
    
    # Download metadata files
    print("Downloading metadata files...")
    download_metadata()
    
    # Load ROI IDs
    print("\nLoading ROI IDs from metadata...")
    roi_ids = load_roi_list()
    print(f"Found {len(roi_ids)} unique ROI IDs")
    
    # Split ROIs into chunks for parallel processing
    chunk_size = len(roi_ids) // n_cores + 1
    roi_chunks = [roi_ids[i:i + chunk_size] for i in range(0, len(roi_ids), chunk_size)]
    
    # Download ROIs in parallel
    print(f"\nDownloading ROIs using {n_cores} processes...")
    with mp.Pool(n_cores) as pool:
        pool.map(download_roi_worker, roi_chunks)
    
    print("\nDownload completed!")

if __name__ == "__main__":
    main()