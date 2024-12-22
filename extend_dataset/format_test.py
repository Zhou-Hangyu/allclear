import os
import rasterio as rs
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import warnings

def open_test(output_path):
  with rs.open(output_path) as src:
    print(src.profile)
    print(src.width, src.height)

def center_crop(input_path, output_path, crop_size=(256, 256)):
  """Load the COG and center crop it to the specified size."""
  temp_path = f"{output_path}.tmp.tif"

  with rs.open(input_path) as src:
    width, height = src.width, src.height
    center_x, center_y = width // 2, height // 2
    offset_x = center_x - crop_size[0] // 2
    offset_y = center_y - crop_size[1] // 2
    window = Window(offset_x, offset_y, crop_size[0], crop_size[1])

    data = src.read(window=window)
    profile = src.profile.copy()
    profile.update({
      "width": crop_size[0],
      "height": crop_size[1],
      "transform": src.window_transform(window),
      "compress": "zstd",
    })
    
    with rs.open(temp_path, "w", **profile) as dst:
      dst.write(data)

  # Convert the cropped image to COG
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      cog_translate(
        source=temp_path,
        dst_path=output_path,
        dst_kwargs=cog_profiles.get("zstd"),
        quiet=True
      )
  except Exception as e:
    print(f"Error converting to COG: {e}")
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)

def process_chunk(chunk, source_prefix, target_prefix, crop_size):
    """Process a chunk of files."""
    for input_path in chunk:
        output_path = input_path.replace(source_prefix, target_prefix)
        if os.path.exists(output_path):
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
          center_crop(input_path, output_path, crop_size)
        except Exception as e:
          print(f"Error processing {input_path}: {e}")

def split_list(input_list, n_chunks):
    """Split the input list into n_chunks roughly equal parts."""
    chunk_size = len(input_list) // n_chunks + 1
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def batch_process(files, source_prefix, target_prefix, crop_size=(256, 256), n_cores=None):
    """Batch process files using multiple processes with a tqdm progress bar."""
    n_cores = n_cores
    chunks = split_list(files, n_cores)

    process_chunk_partial = partial(process_chunk, source_prefix=source_prefix, target_prefix=target_prefix, crop_size=crop_size)
    with Pool(processes=n_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_chunk_partial, chunks), total=len(chunks), desc="Overall Progress"):
            pass

if __name__ == "__main__":
    source_prefix = "/scratch/allclear/dataset_v3/dataset_30k_v4"
    target_prefix = "/home/hz477/allclear_dataset_final_version"
    # target_prefix = "/scratch/allclear/allclear_dataset_final_version"
    num_workers = 20

    # with open("data/metadata/allclear_tif_files_path_list.txt", "r") as f:
    #     files = f.read().splitlines()
    with open("data/metadata/residual_files.txt", "r") as f:
      files = f.read().splitlines()

    batch_process(files, source_prefix, target_prefix, crop_size=(256, 256), n_cores=num_workers)