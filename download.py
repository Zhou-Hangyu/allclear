import requests
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import hashlib
import os

# Configuration
BASE_URL = "http://allclear.cs.cornell.edu/dataset/allclear"
METADATA_FILES = ["test_rois_3k.txt", "train_rois_19k.txt", "val_rois_1k.txt"]
CHUNK_SIZE = 8192
CPUS = 8
N_CORES = max(1, CPUS - 1)  # Leave one core free

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
    
    for filename in METADATA_FILES:
        dest_path = metadata_dir / filename
        url = f"{BASE_URL}/metadata/{filename}"
        
        if dest_path.exists():
            print(f"Metadata file {filename} already exists")
            continue
            
        print(f"Downloading metadata: {filename}")
        download_file(url, dest_path)

def load_roi_list():
    """Load and combine all ROI IDs from metadata files"""
    metadata_dir = Path("metadata")
    roi_ids = set()
    
    for filename in METADATA_FILES:
        file_path = metadata_dir / filename
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
        filename = f"roi{roi_id}.tar.gz"
        dest_path = data_dir / filename
        url = f"{BASE_URL}/data/{filename}"
        
        # Skip if already downloaded and verified
        if dest_path.exists() and verify_file(dest_path):
            continue
            
        # Remove if exists but invalid
        if dest_path.exists():
            dest_path.unlink()
            
        # Download file
        success = download_file(url, dest_path, show_progress=False)
        if success and verify_file(dest_path):
            print(f"Successfully downloaded {filename}")
        elif success:
            print(f"Downloaded {filename} but verification failed")
            dest_path.unlink()
        else:
            print(f"Skipping {filename} - not found on server")

def main():
    # Download metadata files
    print("Downloading metadata files...")
    download_metadata()
    
    # Load ROI IDs
    print("\nLoading ROI IDs from metadata...")
    roi_ids = load_roi_list()
    print(f"Found {len(roi_ids)} unique ROI IDs")
    
    # Split ROIs into chunks for parallel processing
    chunk_size = len(roi_ids) // N_CORES + 1
    roi_chunks = [roi_ids[i:i + chunk_size] for i in range(0, len(roi_ids), chunk_size)]
    
    # Download ROIs in parallel
    print(f"\nDownloading ROIs using {N_CORES} processes...")
    with mp.Pool(N_CORES) as pool:
        pool.map(download_roi_worker, roi_chunks)
    
    print("\nDownload completed!")

if __name__ == "__main__":
    main()