import argparse
import pandas as pd
from tqdm import tqdm
import json

def construct_dataset(sensors: dict, main_sensor='s2_toa', tx=3, mode='s2p'):
    """
    Constructs a dataset for cloud removal from satellite images. The function pairs
    each clear image with surrounding cloudy images for testing or generates sets
    of images for training. It ensures no cloudy image is reused across different groups.

    Args:
        sensors (dict of pd.DataFrame): Dict of DataFrames containing metadata for all sensors.
        main_sensor (str): Name of the main sensor.
        tx (int): The number of images to find surrounding each clear image for testing, or in each set for training. Default is 3.
        mode (str): Mode of operation, either 'testing' or 'training'. Default is 'testing'.

    Returns:
        dict: A dictionary containing grouped image information. Each key is an ID, and the value is a dictionary with 'roi' and sensor data.
              For testing:
              {
                  ID1: {'roi': ('roixxxx', (latitude, longitude)), 'target': [(timestamp, image_file_path)],
                        'main_sensor': [(timestamp, image_file_path), (timestamp, image_file_path)...], ...},
                  ID2: {...}, ...
              }
              For training:
              {
                  ID1: {'roi': ('roixxxx', (latitude, longitude)),
                        'main_sensor': [(timestamp, image_file_path), (timestamp, image_file_path)...], ...},
                  ID2: {...}, ...
              }

    Raises:
        ValueError: If provided tx is less than or equal to 0.
                    If mode is not 'testing' or 'training'.

    Note:
        The input DataFrame must contain the following columns: 'roi', 'capture_date',
        'cloud_percentage_30', 'shadow_percentage', and any other necessary columns to
        calculate the cloud and shadow coverage. The function expects the 'capture_date'
        to be convertible to a pandas datetime format, and it relies on chronological
        sorting of these dates to function correctly. It is also assumed that each row
        in the DataFrame has a unique combination of 'roi', 'capture_date', and 'image_file_path'.
    """
    if tx <= 0:
        raise ValueError("tx must be greater than 0")
    def preprocess(sensor_df):
        sensor_df = sensor_df[sensor_df.nan_percentage == 0]
        sensor_df = sensor_df.drop_duplicates(subset=["capture_date", "roi"])
        sensor_df['capture_date'] = pd.to_datetime(sensor_df['capture_date'], format="%Y-%m-%d %H:%M:%S")
        sensor_df.sort_values(by='capture_date', inplace=True)
        return sensor_df

    main_sensor_df = sensors[main_sensor].copy()
    main_sensor_df = preprocess(main_sensor_df)
    if main_sensor == 's2_toa':
        # filter out images without cloud and shadow information
        main_sensor_df = main_sensor_df[main_sensor_df['cloud_percentage_30'] != -1]
    main_sensor_df['total_cloud_shadow'] = main_sensor_df['cloud_percentage_30'] + main_sensor_df['shadow_percentage_30']
    main_sensor_df['clear_image_flag'] = main_sensor_df['total_cloud_shadow'] < 10

    output_dict = {}
    id_counter = 0

    if mode == 's2p':
        for roi, main_sensor_per_roi_df in tqdm(main_sensor_df.groupby('roi'), desc=f"Processing ROIs ({mode})"):
            N = len(main_sensor_per_roi_df)
            used_patch_ids = set()
            for df_idx in range(N - tx):
                if df_idx in used_patch_ids:
                    continue
                sub_sequence = main_sensor_per_roi_df.iloc[df_idx:df_idx + tx + 1]
                if any(idx in used_patch_ids for idx in sub_sequence.index):
                    continue
                if 1 not in sub_sequence['clear_image_flag'].values[1:-1]:  # make sure there is clear image within the time series
                    continue
                # If there is multiple clear images in the middle, randomly select one
                target = sub_sequence[sub_sequence['clear_image_flag'] == 1].sample(1)
                input_sequence = sub_sequence[sub_sequence['capture_date'] != target['capture_date'].values[0]]
                assert not any(idx in used_patch_ids for idx in input_sequence.index), print(input_sequence.index, used_patch_ids)
                assert not any(idx in used_patch_ids for idx in target.index)
                used_patch_ids.update(input_sequence.index)
                used_patch_ids.update(target.index)
                assert target['capture_date'].values[0] not in input_sequence['capture_date'].values
                assert len(input_sequence) == tx
                entry = {
                    'roi': (roi, (target['latitude'].values[0], target['longitude'].values[0])),
                    'target': [(target['capture_date'].dt.strftime('%Y-%m-%d').values[0],
                                target['image_file_path'].values[0])],
                    main_sensor: [(date.strftime('%Y-%m-%d'), path) for date, path in
                                  zip(input_sequence['capture_date'], input_sequence['image_file_path'])]
                }
                tx_min, tx_max = input_sequence['capture_date'].min(), input_sequence['capture_date'].max()
                for sensor_name, sensor_df in sensors.items():
                    if sensor_name == main_sensor:
                        continue
                    sensor_roi_df = sensor_df[sensor_df['roi'] == roi].copy()
                    sensor_roi_df = preprocess(sensor_roi_df)
                    sensor_roi_df = sensor_roi_df[
                        (sensor_roi_df['capture_date'] >= tx_min) & (sensor_roi_df['capture_date'] <= tx_max)
                        ]
                    entry[sensor_name] = [
                        (date.strftime('%Y-%m-%d'), path) for date, path in
                        zip(sensor_roi_df['capture_date'], sensor_roi_df['image_file_path'])
                    ]
                output_dict[id_counter] = entry
                id_counter += 1
    elif mode == 's2s':
        for roi, roi_df in tqdm(main_sensor_df.groupby('roi'), desc=f"Processing ROIs ({mode})"):
            sets_count = len(roi_df) // tx
            for i in range(sets_count):
                set_main_sensor_df = roi_df.iloc[i * tx:(i + 1) * tx]
                entry = {
                    'roi': (roi, (set_main_sensor_df.iloc[0]['latitude'], set_main_sensor_df.iloc[0]['longitude'])),
                    main_sensor: [(date.strftime('%Y-%m-%d'), path) for date, path in
                                  zip(set_main_sensor_df['capture_date'], set_main_sensor_df['image_file_path'])]
                }
                tx_min, tx_max = set_main_sensor_df['capture_date'].min(), set_main_sensor_df['capture_date'].max()
                for sensor_name, sensor_df in sensors.items():
                    if sensor_name == main_sensor:
                        continue
                    sensor_roi_df = sensor_df[sensor_df['roi'] == roi].copy()
                    sensor_roi_df = preprocess(sensor_roi_df)
                    sensor_roi_df = sensor_roi_df[
                        (sensor_roi_df['capture_date'] >= tx_min) & (sensor_roi_df['capture_date'] <= tx_max)
                        ]
                    entry[sensor_name] = [
                        (date.strftime('%Y-%m-%d'), path) for date, path in
                        zip(sensor_roi_df['capture_date'], sensor_roi_df['image_file_path'])
                    ]
                output_dict[id_counter] = entry
                id_counter += 1

    return output_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Construct AllClear Datasets")
    parser.add_argument("--mode", type=str, help="Type of dataset", required=True, choices=['s2p', 's2s'])
    parser.add_argument("--tx", type=int, help="Number of main sensor images in each time series input", default=3)
    parser.add_argument("--main-sensor", type=str, help="Main sensor to use for constructing the dataset", required=True, default='s2_toa')
    parser.add_argument("--main-sensor-metadata", type=str, help="Path to the main sensor metadata file", required=True)
    parser.add_argument("--auxiliary-sensors", type=str, nargs='+', help="List of auxiliary sensors", required=True, default=['s1', 'landsat8', 'landsat9'])
    parser.add_argument("--auxiliary-sensor-metadata", type=str, nargs='+', help="Path to the metadata files for auxiliary sensors", required=True)
    parser.add_argument("--output-dir", type=str, help="Output path", required=True, default='/scratch/allclear/metadata/v3/')
    parser.add_argument("--version", type=str, help="Version of the dataset", required=True, default='v3')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main_sensor_metadata = pd.read_csv(args.main_sensor_metadata)
    auxiliary_sensor_metadata = {sensor: pd.read_csv(metadata) for sensor, metadata in zip(args.auxiliary_sensors, args.auxiliary_sensor_metadata)}
    sensors = {args.main_sensor: main_sensor_metadata, **auxiliary_sensor_metadata}
    dataset = construct_dataset(sensors, main_sensor=args.main_sensor, tx=args.tx, mode=args.mode)
    with open(f'{args.output_dir}/{args.mode}_tx{str(args.tx)}_{args.version}.json', 'w') as f:
        json.dump(dataset, f)
    intermediate_names = args.main_sensor_metadata.split("_metadata.csv")[0].split("_")
    dataset_string = f"{intermediate_names[-2]}_{intermediate_names[-1]}"
    print(f"Found {len(dataset)} entries in the dataset")
    print(f"Dataset saved to {args.output_dir}/{args.mode}_tx{str(args.tx)}_{dataset_string}_{args.version}.json")
