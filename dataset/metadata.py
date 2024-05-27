import pandas as pd
from tqdm import tqdm
import os
import rasterio as rs
import numpy as np
from multiprocessing import Pool


def nan_percentage(patch_info, channels=None):
    """Calculate the NaN percentage for each patch."""
    data = center_crop(patch_info["image_file_path"], channels=channels, size=(256, 256))
    nan_percentage = np.isnan(data).mean() * 100
    return {"nan_percentage": nan_percentage}

def center_crop(fpath, channels=None, size=(256, 256)):
    """Center crop an image."""
    with rs.open(fpath) as src:
        width, height = src.width, src.height
        center_x, center_y = width // 2, height // 2
        offset_x = center_x - size[0] // 2
        offset_y = center_y - size[1] // 2
        window = rs.windows.Window(offset_x, offset_y, size[0], size[1])
        if channels is None:
            data = src.read(window=window)
        else:
            data = src.read(channels, window=window)
    return data


def cloud_shadow_percentage(patch_info):
    shadow_cloud_file_path = patch_info["cloud_shadow_file_path"]
    if not os.path.exists(shadow_cloud_file_path):
        if VERBOSE:
            print(f"Shadow cloud file not found: {shadow_cloud_file_path}")
        data = {
            "cloud_percentage_0": -1,
            "cloud_percentage_30": -1,
            "shadow_percentage_30": -1,
        }
        return data
    # Channels: "s2_cld_prb_s2cloudless", "clouds_30", "shadows_thres_20", "shadows_thres_25", "shadows_thres_30"
    cld_shd = center_crop(shadow_cloud_file_path, size=(256, 256))
    cloud_mask_0 = cld_shd[0] > 0
    cloud_mask_30 = cld_shd[1]
    shd_mask_30 = cld_shd[-1]
    # compute percentage without nan
    cloud_percent_0 = np.nanmean(cloud_mask_0) * 100
    cloud_percent_30 = np.nanmean(cloud_mask_30) * 100
    shd_percent_30 = np.nanmean(shd_mask_30) * 100
    data = {
        "cloud_percentage_0": cloud_percent_0,
        "cloud_percentage_30": cloud_percent_30,
        "shadow_percentage_30": shd_percent_30,
    }
    return data


def process_general_data(df):
    tqdm.pandas(desc="Processing General Satellite Patches")
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            result = nan_percentage(row)
            results.append(pd.Series(result, name=index))
        except Exception as e:
            print(f"Error: {e}. Skipping patch {row['image_file_path']}")
            df.drop(index, inplace=True)
            continue

    results_df = pd.concat(results, axis=1).transpose()
    return df.merge(results_df, left_index=True, right_index=True, how="inner")


def process_dw_data(df):
    tqdm.pandas(desc="Processing Land Cover Satellite Patches")
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        pass

        try: # TODO: fix corrupted images during error handling
            result = nan_percentage(row)
            results.append(pd.Series(result, name=index))
        except Exception as e:
            print(f"Error: {e}. Skipping patch {row['image_file_path']}")
            df.drop(index, inplace=True)
            continue

    results_df = pd.concat(results, axis=1).transpose()
    return df.merge(results_df, left_index=True, right_index=True, how="inner")

def process_s2_data(df):
    tqdm.pandas(desc="Processing Sentinel-2 Patches")
    df["cloud_shadow_file_path"] = df["image_file_path"].replace("s2_toa", "cld_shdw", regex=True)
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            result1 = nan_percentage(row, channels=[1])
            result2 = cloud_shadow_percentage(row)
            result = {**result1, **result2}
            results.append(pd.Series(result, name=index))
        except Exception as e:
            print(f"Error: {e}. Skipping patch {row['image_file_path']}")
            df.drop(index, inplace=True)
            continue

    results_df = pd.concat(results, axis=1).transpose()
    return df.merge(results_df, left_index=True, right_index=True, how="inner")


def process_chunk(args):
    df, satellite = args
    if satellite == "s2_toa":
        return process_s2_data(df)
    elif satellite == "dw":
        return process_dw_data(df)
    else:
        return process_general_data(df)

def parallel_process(df, satellite, num_cores):
    print(f"Acquiring all metadata for {satellite} satellite patches...")
    # Split dataframe into chunks
    num_cores = num_cores
    chunk_size = len(df) // num_cores + 1
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, [(chunk, satellite) for chunk in chunks])
    return pd.concat(results, ignore_index=True)


def find_all_tiles(root: str, rois: [], date_range: [], satellites: [], rois_metadata: pd.DataFrame):
    data_entries = []
    for roi in tqdm(rois, desc='Processing ROIs'):
        roi_id = int(roi.split("roi")[1])  # "roixxxx" -> xxxx
        roi_metadata = rois_metadata[rois_metadata['index'] == roi_id]
        lat = roi_metadata['latitude'].values[0]
        lon = roi_metadata['longitude'].values[0]
        roi_path = os.path.join(root, roi)
        if os.path.isdir(roi_path):
            for date in date_range:
                date_path = os.path.join(roi_path, date)
                if os.path.isdir(date_path):
                    for satellite in satellites:
                        s_path = os.path.join(date_path, satellite)
                        if os.path.isdir(s_path):
                            images = [f for f in os.listdir(s_path) if f.endswith('.tif')]
                            metadata_files = {f: os.path.join(s_path, f) for f in os.listdir(s_path) if f.endswith('_metadata.csv')}
                            for image in images:
                                image_file_path = os.path.join(s_path, image)
                                metadata_file = image.replace('.tif', '_metadata.csv')
                                metadata_file_path = metadata_files.get(metadata_file, "")
                                data_entries.append({
                                    'roi': roi,
                                    'latitude': lat,
                                    'longitude': lon,
                                    'date_y_m': date,
                                    'satellite': satellite,
                                    'image_file_path': image_file_path,
                                    'metadata_file_path': metadata_file_path,
                                })
    return pd.DataFrame(data_entries)

def sanity_checks_tiles(df):
    """Perform sanity checks on the initial tile metadata df."""
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values in the DataFrame.")
    duplicate_entries = df.duplicated().sum()
    if duplicate_entries > 0:
        print(f"Warning: Found {duplicate_entries} duplicate entries in the DataFrame.")


def get_init_metadata(df):
    """Process and split the DataFrame by satellites and extract metadata for each patch."""
    satellites = df['satellite'].unique()
    results = {}

    for satellite in satellites:
        satellite_df = df[df['satellite'] == satellite].copy()
        metadata = []
        for index, row in tqdm(satellite_df.iterrows(), total=satellite_df.shape[0],
                               desc=f'Processing patches for satellite {satellite}'):
            try:
                metadata_df = pd.read_csv(row['metadata_file_path'])
                metadata_dict = metadata_df.to_dict(orient='records')[0] if not metadata_df.empty else {}
                data = row.to_dict()
                data.update(metadata_dict)
                metadata.append(data)
            except Exception as e:
                print(f"Error: {e}. Skipping patch {row['image_file_path']}")
                continue
        processed_df = pd.DataFrame(metadata)
        primary_columns = ['roi', 'latitude', 'longitude', 'date_y_m', 'satellite', 'image_file_path', 'metadata_file_path']
        metadata_columns = [col for col in processed_df.columns if col not in primary_columns]
        final_columns = primary_columns + sorted(metadata_columns)
        results[satellite] = processed_df[final_columns]
        # make capture_date datetime object, and add patch uid
        results[satellite]['capture_date'] = pd.to_datetime(results[satellite]['capture_date'], format="%Y-%m-%d %H:%M:%S")
        results[satellite]['uid'] = (results[satellite]['roi'] + '_' +
                                     results[satellite]['capture_date'].dt.strftime('%Y%m%d%H%M') + '_' +
                                     results[satellite]['satellite'])
        cols = results[satellite].columns.tolist()
        cols.insert(0, cols.pop(cols.index('uid')))
        results[satellite] = results[satellite][cols]
    return results

if __name__ == "__main__":
    # Command: python -m dataset.metadata
    VERBOSE = True
    # DATA_PATH = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4"
    DATA_PATH = "/scratch/allclear/dataset_v3/dataset_30k_v4"
    ROIS_METADATA = pd.read_csv("/share/hariharan/cloud_removal/allclear/experimental_scripts/data_prep/v3_distribution_train_20Ksamples.csv")
    # ROIS_METADATA = pd.read_csv("/share/hariharan/cloud_removal/allclear/experimental_scripts/data_prep/v3_distribution_test_4Ksamples.csv")
    ROIS = [f"roi{roi_id}" for roi_id in ROIS_METADATA['index'].tolist()]
    # SELECTED_ROIS_FNAME = "test_4k_full.txt"
    SELECTED_ROIS_FNAME = "train_2k.txt"
    # SELECTED_ROIS_FNAME = "dataset_500.txt"
    # SELECTED_ROIS_FNAME = "train_9k.txt"
    # SELECTED_ROIS = ROIS
    with open(f"/share/hariharan/cloud_removal/metadata/v3/{SELECTED_ROIS_FNAME}") as f:
        SELECTED_ROIS = f.read().splitlines()
    DATE_RANGE = [f'2022_{i}' for i in range(1, 13)]
    # SATS = ['s1', 's2_toa', 'cld_shdw', 'dw', 'landsat8', 'landsat9']
    # SATS = ['s2_toa']
    # SATS = ['s2_toa', 's1', 'cld_shdw', 'dw']
    SATS = ['s2_toa', 's1']
    # SATS = ['cld_shdw']
    WORKERS=32
    # WORKERS=96

    # Find all tiles
    metadata = find_all_tiles(DATA_PATH, SELECTED_ROIS, DATE_RANGE, SATS, ROIS_METADATA)
    sanity_checks_tiles(metadata)
    # metadata.to_csv(
    #     f"/share/hariharan/cloud_removal/metadata/v3/{SELECTED_ROIS_FNAME.split('.')[0]}_alltiles_metadata.csv",
    #     index=False)

    # Split up satellites and populate the original metadata
    metadata_dict = get_init_metadata(metadata)

    # Parallel I/O to get more metadata
    for satellite in metadata_dict:
        metadata_dict[satellite] = parallel_process(metadata_dict[satellite], satellite, WORKERS)
        # metadata_dict[satellite].to_csv(f"/scratch/allclear/metadata/v3/{satellite}_{SELECTED_ROIS_FNAME.split('.')[0]}_metadata.csv", index=False)
        metadata_dict[satellite].to_csv(f"/share/hariharan/cloud_removal/metadata/v3/{satellite}_{SELECTED_ROIS_FNAME.split('.')[0]}_metadata.csv", index=False)
    print("DataFrames are processed.")