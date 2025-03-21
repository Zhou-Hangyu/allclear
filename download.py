import requests
from pathlib import Path
import string
import subprocess
from tqdm import tqdm
import os

# Configuration
base_url = "http://allclear.cs.cornell.edu/dataset/allclear"
file_prefix = "allclear_data.tar.part"
download_dir = Path("downloaded_parts")
final_file = "allclear_data.tar"


def download_file(url, dest_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    if response.status_code == 404:
        return False

    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True


def main():
    # Create download directory
    download_dir.mkdir(exist_ok=True)

    # Download all parts
    print("Downloading parts...")
    for suffix in string.ascii_lowercase:  # aa, ab, ac, ...
        part_name = f"{file_prefix}{suffix}{suffix}"
        url = f"{base_url}/{part_name}"
        dest_path = download_dir / part_name

        # Skip if already downloaded
        if dest_path.exists():
            print(f"Skipping {part_name} (already exists)")
            continue

        print(f"Downloading {url}")
        if not download_file(url, dest_path):
            print(f"No more parts found after {part_name}")
            break

    # Concatenate parts
    print("\nConcatenating parts...")
    cat_command = f"cat {download_dir}/{file_prefix}* > {final_file}"
    subprocess.run(cat_command, shell=True, check=True)

    # Extract the tar file
    print("\nExtracting tar file...")
    extract_command = f"tar xf {final_file}"
    subprocess.run(extract_command, shell=True, check=True)

    # Optional: Remove downloaded parts and tar file
    response = input("\nRemove downloaded parts and tar file? (yes/no): ")
    if response.lower() == "yes":
        for part in download_dir.glob(f"{file_prefix}*"):
            part.unlink()
        download_dir.rmdir()
        Path(final_file).unlink()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
