import pandas as pd
import rasterio as rs
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import glob
import datetime
import copy
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

# Path
RAW_DATA_PATH = "/share/hariharan/cloud_removal/MultiSensor/dataset_temp"
DATA_PATH = "/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2"
DATA_PATH_SPT_PATCH = "/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2/spatio_temporal_v46"
DATA_PATH_PATCHINFO = "/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2/patch_info"

# Customized ROIs
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', type=int, default=0, help='an integer for the start')
parser.add_argument('--end', type=int, default=28, help='an integer for the end')
parser.add_argument('--mode', type=str, default="beta", help='an integer for the end')
args = parser.parse_args()

# Load ROI information
logging.info(f"Loading ROI information from ROI {args.start} to ROI {args.end}")
ROIs_TRAIN =  []
roi_meta_df = pd.read_csv("/share/hariharan/cloud_removal/metadata/tile/roi_central_coordinates_sen12mscrts_sen12mscr.csv")

for idx, row in roi_meta_df.iterrows():

    if row["Split"] == "train" and args.start <= row["ROI_ID"] <= args.end:
        ROIs_TRAIN += [{
                "ROI_ID": f"""roi{row["ROI_ID"]}""",
                "LAT":    row["Center Y (EPSG:4326)"],
                "LONG":   row["Center X (EPSG:4326)"],
                }]
ROIs_TRAIN = pd.DataFrame.from_dict(ROIs_TRAIN)

# Get temporal information for each ROI
logging.info("Get temporal information for each ROI")
starting_date = datetime.datetime(2022, 1, 1)
train_list = {"roi_id": [], 
              "latitude": [],
              "longtitude": [],              
              "daydelta": [], 
              "month": [],
              "date": [],
              "path_s2": [],
              "path_s1": [],
              "path_cloud": [],
              "path_meta": []
             }
for idx, (roi_id, lat, long) in ROIs_TRAIN.iterrows():
    
    days = 365 if args.mode == "beta" else 100
    for daydelta in range(0, days):

        date = starting_date + datetime.timedelta(days=daydelta)
        ym = date.strftime('%Y_%-m')
        m = date.strftime('%-m')
        ymd = date.strftime('%Y_%-m_%-d')
        s2_path = os.path.join(RAW_DATA_PATH, roi_id, ym, 's2', f'{roi_id}_s2_{ymd}_median.tif')
        s1_path = os.path.join(RAW_DATA_PATH, roi_id, ym, 's1', f'{roi_id}_s1_{ymd}_median.tif')
        cloud_path = os.path.join(RAW_DATA_PATH, roi_id, ym, 'cld_shdw_new', f'{roi_id}_cld_shdw_new_{ymd}_median.tif')
        meta_path = os.path.join(RAW_DATA_PATH, roi_id, ym, 's2', f'{roi_id}_s2_{ymd}_median_metadata.csv')

        if not (os.path.exists(s2_path) and os.path.exists(cloud_path)) and not os.path.exists(s1_path):
            continue

        train_list["roi_id"] += [roi_id]
        train_list["latitude"] += [lat]
        train_list["longtitude"] += [long]
        train_list["daydelta"] += [daydelta]
        train_list["month"] += [m]
        train_list["date"] += [ymd]

        if os.path.exists(s2_path):
            train_list["path_s2"] += [s2_path]
            train_list["path_meta"] += [meta_path]
        else:
            train_list["path_s2"] += [""]
            train_list["path_meta"] += [""]  

        if os.path.exists(s1_path):
            train_list["path_s1"] += [s1_path]
        else:
            train_list["path_s1"] += [""]
            

        if os.path.exists(cloud_path):
            train_list["path_cloud"] += [cloud_path]
        else:
            train_list["path_cloud"] += [""]

roi_path_list = pd.DataFrame.from_dict(train_list)
roi_path_list.to_csv(os.path.join(DATA_PATH, "ROI_datas_list.csv"))
print(f"Size of good tiles: {len(roi_path_list)}")




# # Check and get Patch information from loading single band from s1 and s2.
# logging.info("Check and get Patch information from loading single band from s1 and s2.")
# patch_size = 256
# data = []
# previous_roi = "roi-1"
# for idx, row in tqdm(roi_path_list.iterrows(), total=len(roi_path_list)):
        
#     roi_id = row["roi_id"]
#     path_s2 = row["path_s2"]
#     path_meta = row["path_meta"]
#     path_s1 = row["path_s1"]
#     path_cloud = row["path_cloud"]
    
#     if previous_roi != roi_id and idx != 0:
#         logging.info(f"Complete {previous_roi}")
#         df_temp = pd.DataFrame(data)
#         SAVE_PATH = os.path.join(DATA_PATH_PATCHINFO, f"{previous_roi}.csv")
#         df_temp.to_csv(SAVE_PATH)
#         data = []
        
#     previous_roi = roi_id
    
#     # Check and read S2
#     if path_s2 is not None and len(path_s2) > 1:
#         with rs.open(path_s2) as src:
#             s2_vis = np.transpose(src.read([4,3,2]), (1,2,0))
#             s2_data = src.read([2])[0]
#     else:
#         s2_data = None
        
#     if path_s1 is not None and len(path_s1) > 1:
#         with rs.open(path_s1) as src:
#             s1_data = src.read([1])[0]
#     else:
#         s1_data = None

#     if path_cloud is not None and len(path_cloud) > 1:
#         with rs.open(path_cloud) as src:
#             cloud_data = src.read([1,2,3,4,5])
#     else:
#         cloud_data = None

#     # Check and read Meta
#     if path_meta is not None and len(path_meta) > 1:
#         meta = pd.read_csv(path_meta)
#     else:
#         meta = None

#     # Iterate over patches
#     for idx_j, j in enumerate(range(0, 1024, patch_size)):
#         for idx_i, i in enumerate(range(0, 1024, patch_size)):
#             patch_id = idx_j * 4 + idx_i
#             row_data = {
#                 'roi': roi_id,
#                 'daydelta': row["daydelta"],
#                 'month': row["month"],
#                 'date': row["date"],
#                 "patch_id": patch_id,
#                 'offset_x': i,
#                 'offset_y': j,
#                 "latitude": row["latitude"], 
#                 "longtitude": row["longtitude"], 
#                 "path_s2": row["path_s2"],
#                 "path_meta": row["path_meta"],
#                 "path_s1": row["path_s1"],
#                 "path_cloud": row["path_cloud"],
#             }

#             s2_count_nan = np.nansum(s2_data[i:i+patch_size, j:j+patch_size] < 0) if s2_data is not None else -1
#             s2_count_zero = np.nansum(s2_data[i:i+patch_size, j:j+patch_size] < 1)  if s2_data is not None else -1
#             s2_percentange_3K = np.nansum(s2_data[i:i+patch_size, j:j+patch_size] > 3_000) / (patch_size ** 2) if s2_data is not None else -1
#             s2_percentange_5K = np.nansum(s2_data[i:i+patch_size, j:j+patch_size] > 5_000) / (patch_size ** 2) if s2_data is not None else -1
            
#             s1_count_nan = np.nansum(s1_data[i:i+patch_size, j:j+patch_size] < -40) if s1_data is not None else -1

#             cloud_count_nan = np.nansum(cloud_data[0,i:i+patch_size, j:j+patch_size] < 0) if cloud_data is not None else -1
#             cloud_percentange_30 = np.nansum(cloud_data[1,i:i+patch_size, j:j+patch_size] == 1) / (patch_size ** 2) if cloud_data is not None else -1
#             shadow_percentage_20 = np.nansum(cloud_data[2,i:i+patch_size, j:j+patch_size] == 1) / (patch_size ** 2) if cloud_data is not None else -1
#             shadow_percentage_25 = np.nansum(cloud_data[3,i:i+patch_size, j:j+patch_size] == 1) / (patch_size ** 2) if cloud_data is not None else -1
#             shadow_percentage_30 = np.nansum(cloud_data[4,i:i+patch_size, j:j+patch_size] == 1) / (patch_size ** 2) if cloud_data is not None else -1
            
#             row_data.update({
#                 's2_count_nan': s2_count_nan,
#                 's2_count_zero': s2_count_zero,
#                 's2_percentange_3K': s2_percentange_3K,
#                 's2_percentange_5K': s2_percentange_5K,
#                 's1_count_nan': s1_count_nan,
#                 'cloud_count_nan': cloud_count_nan,
#                 'cloud_percentange_30': cloud_percentange_30,
#                 'shadow_percentage_20': shadow_percentage_20,
#                 'shadow_percentage_25': shadow_percentage_25, 
#                 'shadow_percentage_30': shadow_percentage_30, 
#                 'SENSOR_AZIMUTH': meta["sensor_azimuth"].values[0] if meta is not None else None,
#                 'SENSOR_ZENITH': meta["sensor_zenith"].values[0] if meta is not None else None,
#                 'SUN_AZIMUTH': meta["sun_azimuth"].values[0] if meta is not None else None,
#                 'SUN_ELEVATION': meta["sun_elevation"].values[0] if meta is not None else None,
#             })
            
#             data += [row_data]

#             time.sleep(0.01)

# logging.info(f"Complete {previous_roi}")
# df_temp = pd.DataFrame(data)
# df_temp.to_csv(os.path.join(DATA_PATH_PATCHINFO, f"{previous_roi}.csv"))
# logging.info("Complete patch information")


# Save band images
def save_as_cog(img, file_name, method="zstd"):
    
    assert len(img.shape) == 3
 
    format_configs = {
        # "LZW": {"dtype": "float32"},
        # "deflate": {"dtype": "float32"},
        # "jpeg": {"dtype": "int8"},
        "zstd": {"dtype": "int16"},
    }
        
    # img = img.int()
    
    src_profile = {
        "driver": "GTiff",
        "tiled": True,
        "dtype": format_configs[method]["dtype"],
        "count": img.shape[0],
        "height": img.shape[1],
        "width": img.shape[2],
        "BLOCKXSIZE": 256,
        "BLOCKYSIZE": 256,
        "crs": "epsg:4326",  # Assuming EPSG:4326 for the CRS
    }

    # Define the COG profile with default settings
    dst_profile = cog_profiles.get(method)

    # Translate the input array to a COG
    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as mem:
            # Populate the input file with numpy array
            mem.write(img)

            # Translate the in-memory file to a COG
            cog_translate(
                mem,
                file_name,
                dst_profile,
                in_memory=True,
                quiet=True,
            )
            
            
sat_configs = {
    "s1":     {"num_channels":  2, "band_list": [1,2], "vmin": -40, "vmax": 20},
    "s2":     {"num_channels": 12, "band_list": [1,2,3,4,5,6,7,8,9,10,11,12], "vmin": 0, "vmax": 10_000},    
    "cloud":  {"num_channels":  5, "band_list": [1,2,3,4,5], "vmin": 0, "vmax": 2},
}

# S2 bands correspondance: "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "MSK_CLDPRB", "QA60
"""
1: B1: Aerosol retrieval
2: B2: Blue
3: B3: Green
4: B4: Red
5: B5: Vegetation red edge
6: B6: Vegetation red edge
7: B7: Vegetation red edge
8: B8: NIR
9: B8A: Narrow NIR
10: B9: Water vapor
11: B11: SWIR
12: B12: SWIR
"""

# new s2 channels: 4,3,2,5,6,7,8,9,11,12
"""
0: B4 - Red
1: B3 - Green
2: B2 - Blue
3: B5 - Vegetation red edge
4: B6 - Vegetation red edge
5: B7 - Vegetation red edge
6: B8 - NIR
7: B8A - Narrow NIR
8: B11 - SWIR
9: B12 - SWIR
"""


logging.info("Save band images")
for idx, row in ROIs_TRAIN.iterrows():
    
    roi = row["ROI_ID"]
    patch_info_df = pd.read_csv(os.path.join(DATA_PATH_PATCHINFO, f"{roi}.csv"))
    
    print(f"Processing ROI {roi}")
    
    check_book = {"roi_id": [], 
                  "patch_id": [], 
                  "dates": [],
                  "day_count": [],
                  "latitude": [],
                  "longtitude": [],
                  "sun_azimuth": [],
                  "sun_elevation": [],
                  "month": [], 
                 }
    
    patch_list = patch_info_df[patch_info_df["patch_id"] == 0]
    s1_patch_list = patch_list[patch_list["s1_count_nan"] != -1]
    s2_patch_list = patch_list[patch_list["s2_count_nan"] != -1]
    cloud_patch_list = patch_list[patch_list["cloud_count_nan"] != -1]

    window = rs.windows.Window(0, 0, 1024, 1024)
    dates = patch_list["date"].values
    daycounts = patch_list["daydelta"].values
    
    buffer = np.zeros((13+2+3, len(patch_list), 1024, 1024), dtype=np.float16)
    buffer[15:] = 1

    for row_idx, date in tqdm(enumerate(patch_list["date"].unique()), total=len(patch_list)):

        time_patch_info = patch_list[patch_list["date"]==date]
        path_s2 = time_patch_info["path_s2"].item()
        path_s1 = time_patch_info["path_s1"].item()
        path_cloud = time_patch_info["path_cloud"].item()

        if isinstance(path_s2, str) and len(path_s2) > 1 and isinstance(path_cloud, str) and len(path_cloud) > 1:
            
            with rs.open(path_s2) as src:
                msi = src.read(sat_configs["s2"]["band_list"], window=window)
            msi[msi<-1] = -1
            msi = np.nan_to_num(msi, nan=-1)
            msi = msi.clip(-1, 10_000, out=msi)
            buffer[:10, row_idx] = msi[:10]
            buffer[11:13, row_idx] = msi[10:12]
            
            msi_cloud_mask = (msi[1] <= -1) & (msi[1] >= 10_000)

            with rs.open(path_cloud) as src:
                cloud = src.read(sat_configs["cloud"]["band_list"], window=window)
            cloud[0] = np.nan_to_num(cloud[0], nan=100)
            cloud[0,msi_cloud_mask] = 100
            cloud[0][cloud[0]<0] = 100
            
            cloud[1:5] = np.nan_to_num(cloud[1:5], nan=1)
            cloud[1:5,msi_cloud_mask] = 1
            cloud[1:5][cloud[1:5]<0] = 1
            
            buffer[15:18, row_idx] = cloud[[0,1,3]]
            time.sleep(0.2)
            
        if isinstance(path_s1, str) and len(path_s1) > 1:
            with rs.open(path_s1) as src:
                sar = src.read(sat_configs["s1"]["band_list"], window=window)
            sar[sar<-40] = -40
            sar[0] = np.clip(sar[0] + 25, 0, 25)
            sar[1] = np.clip(sar[1] + 32.5, 0, 32.5)
            sar = np.nan_to_num(sar, nan=-1)
            
            buffer[13:15, row_idx] = sar * 1_000

            time.sleep(0.2)

    for patch_row_idx in tqdm(range(4)):
        for patch_col_idx in range(4):

            row_start_index = patch_row_idx * 256
            col_start_index = patch_col_idx * 256
            patch_id = patch_row_idx * 4 + patch_col_idx

            SAVE_PATH = os.path.join(DATA_PATH_SPT_PATCH, f"{roi}_patch{patch_id}.cog")
            img = buffer[:,:,row_start_index:row_start_index+256,col_start_index:col_start_index+256]
            
            # s1 mask:
            # - if s1 is all zeros
            # - if > 30 % of image is nan (aka 0 or -1)
            invalid_s1_mask = (np.max(img[13:15], axis=(0,2,3), keepdims=False) == 0) | (np.mean((img[13:15] <= 0 ), axis=(0,2,3)) > 0.3)
            # s2_mask:
            # - if all the s2 iamge is nan (aka -1)
            # - if all the RGB values are 10000
            # - if > 30 % of image is nan (aka 0 or -1)
            invalid_s2_mask = (np.max(img[1:4], axis=(0,2,3), keepdims=False) == -1) | (np.mean((img[1:4]==10000), axis=(0,2,3)) == 1) | (np.mean(img[1:4]<=0, axis=(0,2,3), keepdims=False) > 0.3)
            invalid_mask = invalid_s1_mask & invalid_s2_mask
            print(f"Removing {invalid_mask.sum()} images from {len(invalid_mask)} images")
            img = img[:,~invalid_mask]
            
            img = img.reshape(18,-1,256)
            save_as_cog(img, SAVE_PATH, method="zstd")

            check_book["roi_id"] += [roi]
            check_book["patch_id"] += [patch_id]
            check_book["dates"] += [patch_list["date"].values[~invalid_mask]]
            check_book["day_count"] += [patch_list["daydelta"].values[~invalid_mask]]
            check_book["latitude"] += [patch_list["latitude"].values[0]]
            check_book["longtitude"] += [patch_list["longtitude"].values[0]]
            check_book["sun_azimuth"] += [patch_list["SENSOR_AZIMUTH"].values[0]]
            check_book["sun_elevation"] += [patch_list["SUN_ELEVATION"].values[0]]
            check_book["month"] += [patch_list["month"].values[0]]

            logging.info(f"Saved: {SAVE_PATH}")

            time.sleep(0.2)

    check_book = pd.DataFrame.from_dict(check_book)
    check_book.to_csv(os.path.join(DATA_PATH_SPT_PATCH, f"{roi}.csv"))
    
    del buffer