import os
import copy
from datetime import datetime
import numpy as np
import rasterio as rs
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob


def temporal_align_aux_sensors(main_sensor_timestamps, aux_sensor_timestamp, max_diff=2):
    differences = [abs(dt - aux_sensor_timestamp) for dt in main_sensor_timestamps]
    if min(differences).days > max_diff:
        return None
    else:
        return differences.index(min(differences))


class AllClearDataset(Dataset):
    """
        Loading and preprocessing satellite imagery data, including main sensor data
        and auxiliary sensor data. This class supports various sensors, cloud and shadow masks, and different
        temporal alignments.

        Attributes:
            dataset (dict): A dictionary containing dataset information.
            selected_rois (list or str): Regions of interest to be selected from the dataset. Use "all" to select all ROIs.
            main_sensor (str): The main sensor used in the dataset. Default is "s2_toa".
            aux_sensors (list): List of auxiliary sensors to include. Default includes "s1", "landsat8", "landsat9".
            aux_data (list): List of auxiliary data types. Default includes 'cld_shdw' and 'dw'.
            tx (int): Temporal length (number of time steps) of the input sequence.
            center_crop_size (tuple): Size of the center crop applied to the images. Default is (256, 256).
            clds_shdws (torch.Tensor): Cloud and shadow masks.
            format (str): Format of the dataset. Only "stp" (spatio-temporal patch) is supported.
            target (str): Target type. Can be "s2s" (seq2seq) or "s2p" (seq2point).
            s2_toa_channels (list): List of channels to use from the main sensor. Default uses all available channels.
            max_diff (int): Maximum temporal difference for aligning auxiliary sensors with the main sensor.
            channels (dict): Dictionary mapping sensors to their respective channels.

        Methods:
            __len__(): Returns the number of samples in the dataset.
            load_and_center_crop(fpath, channels=None, size=(256, 256)): Loads and center crops an image from the given file path.
            preprocess(image, sensor_name): Preprocesses the image according to the sensor type.
            __getitem__(idx): Retrieves the sample at the given index.

        Example:
        data = {
            1: {"roi": ("roi1234", (lat, long)), "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value1"},
            2: {"roi": ("roi5678", (lat, long)), "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value2"},
            3: {"roi": ("roi9101", (lat, long)), "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value3"}
        }
        dataset = AllClearDataset('dataset.json', ['roi1', 'roi2'], 3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    """

    def __init__(self,
                 dataset,
                 selected_rois,
                 main_sensor="s2_toa",
                 aux_sensors=None,
                 aux_data=None,
                 tx=3,
                 center_crop_size=(256, 256),
                 format="stp",
                 target_mode="s2s",
                 s2_toa_channels=None,
                 max_diff=2,
                 s1_preprocess_mode="default",
                 ):
        if aux_sensors is None:
            aux_sensors = []
        if aux_data is None:
            aux_data = ['cld_shdw', 'dw']
        if selected_rois == "all":
            self.dataset = dataset
        else:
            self.dataset = {ID: info for ID, info in dataset.items() if info["roi"][0] in selected_rois}
        self.main_sensor = main_sensor
        self.aux_sensors = aux_sensors
        self.sensors = [main_sensor] + aux_sensors
        self.aux_data = aux_data
        self.tx = tx
        self.center_crop_size = center_crop_size
        self.format = format
        self.target_mode = target_mode
        self.max_diff = max_diff
        self.s1_preprocess_mode = s1_preprocess_mode
        if self.format != "stp":
            raise ValueError("The format is not supported.")

        if s2_toa_channels is None:
            self.channels = {
                "s2_toa": list(range(1, 14)),
                "s1": [1, 2],
                "landsat8": list(range(1, 12)),
                "landsat9": list(range(1, 12)),
                "cld_shdw": [2, 5],
                "dw": [1],
            }
        else:
            self.channels = {
                "s2_toa": s2_toa_channels,
                "s1": [1, 2],
                "landsat8": list(range(1, 12)),
                "landsat9": list(range(1, 12)),
                "cld_shdw": [2, 5],
                "dw": [1],
            }
        self.dataset_ids = list(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def load_and_center_crop(fpath, channels=None, size=(256, 256)):
        """Load and center crop an image."""
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
        data = torch.from_numpy(data).float()
        return data

    @staticmethod
    def rescale(date, limits):
        return (date - limits[0]) / (limits[1] - limits[0])

    def preprocess(self, image, sensor_name):
        """Preprocess the image."""
        if sensor_name in ["s2_toa", "landsat8", "landsat9"]:
            image = torch.clip(image, 0, 10000) / 10000
            image = torch.nan_to_num(image, nan=0)
        elif sensor_name == "s1":
            if self.s1_preprocess_mode == "default":
                image[image < -40] = -40
                image[0] = torch.clip(image[0] + 25, 0, 25) / 25
                image[1] = torch.clip(image[1] + 32.5, 0, 32.5) / 32.5
                image = torch.nan_to_num(image, nan=-1)
            elif self.s1_preprocess_mode == "uncrtaints":
                dB_min, dB_max = -25, 0
                image = np.clip(image, dB_min, dB_max)
                image = self.rescale(image, [dB_min, dB_max])

        elif sensor_name == "cld_shdw":
            image = torch.nan_to_num(image, nan=1)
        elif sensor_name in ["dw"]:
            image = image
        else:  # TODO: Implement preprocessing for other sensors
            print(f'Preprocessing steps for {sensor_name} has not been implemented yet.')
            image = image
        return image

    @staticmethod
    def extract_date(path):
        parts = path.split('_')
        date_str = parts[-4] + '-' + parts[-3] + '-' + parts[-2]
        return datetime.strptime(date_str, '%Y-%m-%d')

    def __getitem__(self, idx):
        data_id = self.dataset_ids[idx]
        sample = self.dataset[data_id]
        roi = sample["roi"][0]
        latlong = sample["roi"][1]
        inputs_cld_shdw = None

        # load in images as lists of (timestamp, image) pairs (also load in cloud and shadow masks for main sensor)
        inputs = {sensor: [] for sensor in
                  self.sensors + ["input_cld_shdw", "input_dw", "target_cld_shdw", "target_dw"]}
        timestamps = []

        for sensor in self.sensors:
            sensor_inputs = sample[sensor]
            for sensor_input in sensor_inputs:
                timestamp, fpath = sensor_input
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                if not os.path.exists(
                        fpath):  # Add this line to handle the case when the script is not running on Sun / Bala's server
                    fpath = fpath.replace("/scratch/allclear/dataset_v3/",
                                          "/share/hariharan/cloud_removal/MultiSensor/")
                image = self.load_and_center_crop(fpath, self.channels[sensor], self.center_crop_size)
                image = self.preprocess(image, sensor)
                inputs[sensor].append((timestamp, image))
                if sensor == self.main_sensor:
                    timestamps.append(timestamp)
                    if "cld_shdw" in self.aux_data:
                        cld_shdw_fpath = fpath.replace("s2_toa", "cld_shdw")
                        if os.path.exists(cld_shdw_fpath):
                            cld_shdw = self.load_and_center_crop(cld_shdw_fpath, self.channels["cld_shdw"],
                                                                 self.center_crop_size)
                            cld_shdw = self.preprocess(cld_shdw, "cld_shdw")
                        else:
                            raise ValueError(f"Cloud shadow file not found: {cld_shdw_fpath}")
                        inputs["input_cld_shdw"].append((timestamp, cld_shdw))
                    if "dw" in self.aux_data:
                        dw_fpath = fpath.replace("s2_toa", "dw")
                        if os.path.exists(dw_fpath):
                            dw = self.load_and_center_crop(dw_fpath, self.channels["dw"], self.center_crop_size)
                            dw = self.preprocess(dw, "dw")
                            inputs["input_dw"].append((timestamp, dw))
                        else:
                            inputs["input_dw"].append((timestamp, torch.ones(len(self.channels['dw']),
                                                                             *self.center_crop_size)))  # placeholder
            inputs[sensor] = sorted(inputs[sensor], key=lambda x: x[0])
        timestamps = sorted(set(timestamps))
        timestamps_main_sensor = [timestamp for timestamp, _ in inputs[self.main_sensor]]
        start_date, end_date = timestamps[0], timestamps[-1]
        time_differences = [round((timestamp - start_date).total_seconds() / (24 * 3600)) for timestamp in
                            timestamps]
        timestamps = [timestamp.timestamp() for timestamp in timestamps]

        # organize the data into desired format
        if self.format == "stp":  # TODO: refactor this code once more format is added.
            inputs_main_sensor = torch.stack([image for _, image in inputs[self.main_sensor]])
            if "cld_shdw" in self.aux_data:
                inputs_cld_shdw = torch.stack([cld_shdw for _, cld_shdw in inputs["input_cld_shdw"]])
            else:
                inputs_cld_shdw = None
            if "dw" in self.aux_data:
                inputs_dw = torch.stack([dw for _, dw in inputs["input_dw"]])
            else:
                inputs_dw = None

        # load in targets. If seq2seq (s2s), target use syncld; if seq2point (s2p), target use predefined target clear image.
        if self.target_mode == "s2p":
            if "target" not in sample.keys():
                raise ValueError("Target is not available in the sample.")
            timestamp, fpath = sample["target"][0]
            if not os.path.exists(
                    fpath):  # Add this line to handle the case when the script is not running on Sun / Bala's server
                fpath = fpath.replace("/scratch/allclear/dataset_v3/", "/share/hariharan/cloud_removal/MultiSensor/")
            image = self.load_and_center_crop(fpath, self.channels[self.main_sensor], self.center_crop_size)
            image = self.preprocess(image, self.main_sensor)  # target by default is the main sensor
            inputs["target"] = [(timestamp, image)]
            if "cld_shdw" in self.aux_data:
                cld_shdw_fpath = fpath.replace("s2_toa", "cld_shdw")
                if os.path.exists(cld_shdw_fpath):
                    cld_shdw = self.load_and_center_crop(cld_shdw_fpath, self.channels["cld_shdw"],
                                                         self.center_crop_size)
                    cld_shdw = self.preprocess(cld_shdw, "cld_shdw")
                else:
                    raise ValueError(f"Cloud shadow file not found: {cld_shdw_fpath}")
                inputs["target_cld_shdw"] = cld_shdw.unsqueeze(0)
            else:
                inputs["target_cld_shdw"] = None
            if "dw" in self.aux_data and fpath.endswith("tif"):
                dw_fpath = fpath.replace("s2_toa", "dw")
                if not os.path.exists(dw_fpath):
                    dw_fpath = fpath.rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[0] + "/2022_*/dw/*.tif"
                    dw_fpaths = glob.glob(dw_fpath)
                    given_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") if isinstance(timestamp, str) else timestamp
                    dw_fpath = min(dw_fpaths, key=lambda path: abs(self.extract_date(path) - given_date))
                if not dw_fpath.endswith("tif"):
                    print("not ending with tif")
                    dw_fpath = dw_fpath + "/*.tif"
                dw = self.load_and_center_crop(dw_fpath, self.channels["dw"], self.center_crop_size)
                dw = self.preprocess(dw, "dw")
                inputs["target_dw"] = dw.unsqueeze(0)
            else:
                inputs["target_dw"] = None

        # final data organization
        # When a day is missing, insert 1s. (shape: (T, C, H, W) -> (C, T, H, W))
        # for spatial-temporal patch (stp) format, we align aux images with the main one if they are within the max_diff
        # and discard the rest of them to reduce the input sparsity.
        if self.format != "stp":
            raise ValueError(f"The {self.format} format is not supported.")
        else:
            output_sensors = self.sensors
            if "landsat8" in self.sensors and "landsat9" in self.sensors: # avoid adding landsat8 and landsat9 separately
                stp_sensors = [sensor for sensor in self.sensors if sensor != "landsat8"]
            else:
                stp_sensors = output_sensors
            sample_stp = []

            for sensor in stp_sensors:
                sample_stp.append(torch.ones((self.tx, len(self.channels[sensor]), *self.center_crop_size)))
            sample_stp = torch.concat(sample_stp, dim=1)

            # merge landsat8 and landsat9 into one sensor for less sparse input.
            channel_start_index = 0
            channel_start_indices = {}
            for sensor in output_sensors:
                if sensor == "landsat8" and "landsat9" in channel_start_indices:
                    channel_start_indices[sensor] = channel_start_indices["landsat9"]
                elif sensor == "landsat9" and "landsat8" in channel_start_indices:
                    channel_start_indices[sensor] = channel_start_indices["landsat8"]
                else:
                    channel_start_indices[sensor] = channel_start_index
                    channel_start_index += len(self.channels[sensor])

            for sensor in output_sensors:
                if sensor == self.main_sensor:
                    day_index = 0
                    for timestamp, image in inputs[sensor]:
                        image_channel_size = image.shape[0]
                        channel_start_index_sensor = channel_start_indices[sensor]
                        sample_stp[day_index,
                        channel_start_index_sensor:channel_start_index_sensor + image_channel_size, ...] = image
                        day_index += 1
                elif sensor in self.aux_sensors:
                    for timestamp, image in inputs[sensor]:
                        day_index = temporal_align_aux_sensors(timestamps_main_sensor, timestamp,
                                                               max_diff=self.max_diff)
                        if day_index == None:
                            continue
                        else:
                            image_channel_size = image.shape[0]
                            channel_start_index_sensor = channel_start_indices[sensor]
                            sample_stp[day_index,
                            channel_start_index_sensor:channel_start_index_sensor + image_channel_size, ...] = image

            sample_stp = sample_stp.permute(1, 0, 2, 3)
            if self.target_mode == "s2p":
                target_image = inputs["target"][0][1]
                target_image = target_image.unsqueeze(1)
                target_timestamps = datetime.strptime(inputs["target"][0][0], "%Y-%m-%d %H:%M:%S").timestamp()
            elif self.target_mode == "s2s":
                target_image = inputs_main_sensor.permute(1, 0, 2, 3)
                target_timestamps = timestamps

            item_dict = {
                "data_id": data_id if data_id is not None else None,
                "input_images": sample_stp,  # Shape: (C1(main+aux), T, H, W)
                "target": target_image,  # Shape: (C2(main_sensor), T, H, W)
                "input_cld_shdw": inputs_cld_shdw.permute(1, 0, 2, 3) if inputs_cld_shdw is not None else None,
                # Shape: (2, T, H, W)
                "target_cld_shdw": inputs["target_cld_shdw"].permute(1, 0, 2, 3) if inputs[
                                                                                        "target_cld_shdw"] is not None else None,
                # Shape: (1, T, H, W)
                "input_dw": inputs_dw.permute(1, 0, 2, 3) if inputs_dw is not None else None,  # Shape: (C, T, H, W)
                "target_dw": inputs["target_dw"].permute(1, 0, 2, 3) if inputs["target_dw"] is not None else None,
                "timestamps": torch.tensor(timestamps),  # Shape: (T)
                "target_timestamps": torch.tensor(target_timestamps),
                "time_differences": torch.Tensor(time_differences),
                "roi": roi,
                "latlong": latlong,
            }
            item_dict = {k: v for k, v in item_dict.items() if v is not None}

            return item_dict