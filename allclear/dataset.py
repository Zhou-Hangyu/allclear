import os
import copy
from datetime import datetime
import numpy as np
import rasterio as rs
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F



def square_cld_shdw_maxpool(image_size=256):
    """Generate square cloud & shadow masks."""
    mask_scale = np.random.choice([16, 32, 64])
    threshold = np.random.uniform(low=0.1, high=0.25)
    cld = (torch.rand((1, image_size, image_size)) < threshold / (mask_scale * 2) ** 2).float()
    shdw = (torch.rand((1, image_size, image_size)) < threshold / (mask_scale * 2) ** 2).float()
    square_cld = F.max_pool2d(cld, mask_scale + 1, stride=1, padding=int((mask_scale + 1) // 2))
    square_shdw = F.max_pool2d(shdw, mask_scale + 1, stride=1, padding=int((mask_scale + 1) // 2))
    square_cld_shdw = torch.cat([square_cld, square_shdw], dim=0)
    return square_cld_shdw


def square_cld_shdw(image_size=256):
    """Generate square cloud & shadow masks."""
    mask_scale = np.random.choice([16, 32, 64])
    threshold = np.random.uniform(low=0.1, high=0.25)
    num_squares = int(threshold * image_size * image_size / (mask_scale * mask_scale))

    cld = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    shdw = torch.zeros((1, image_size, image_size), dtype=torch.float32)

    for _ in range(num_squares):
        x = np.random.randint(0, image_size - mask_scale)
        y = np.random.randint(0, image_size - mask_scale)
        cld[:, y:y + mask_scale, x:x + mask_scale] = 1

    for _ in range(num_squares):
        x = np.random.randint(0, image_size - mask_scale)
        y = np.random.randint(0, image_size - mask_scale)
        shdw[:, y:y + mask_scale, x:x + mask_scale] = 1

    square_cld_shdw = torch.cat([cld, shdw], dim=0)
    return square_cld_shdw


def random_opacity():
    """Generate a random opacity value."""
    return torch.rand(1).item() * 0.5


def temporal_align_aux_sensors(main_sensor_timestamps, aux_sensor_timestamp, max_diff=2):
    differences = [abs(dt - aux_sensor_timestamp) for dt in main_sensor_timestamps]
    if min(differences).days > max_diff:
        return None
    else:
        return differences.index(min(differences))


class CRDataset(Dataset):
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
            1: {"roi": "roi1234", "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value1"},
            2: {"roi": "roi5678", "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value2"},
            3: {"roi": "roi9101", "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value3"}
        }
        dataset = CRDataset('dataset.json', ['roi1', 'roi2'], 3)
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
                 cld_shdw_fpaths=None,
                 do_preprocess=True,):
        if aux_sensors is None:
            aux_sensors = []
        if aux_data is None:
            aux_data = ['cld_shdw', 'dw']
        if selected_rois == "all":
            self.dataset = dataset
        else:
            self.dataset = {ID: info for ID, info in dataset.items() if info["roi"][0] in selected_rois}
            self.dataset = {str(i): self.dataset[ID] for i, ID in enumerate(self.dataset.keys())} # reindex the dataset
        self.main_sensor = main_sensor
        self.aux_sensors = aux_sensors
        self.sensors = [main_sensor] + aux_sensors
        self.aux_data = aux_data
        self.tx = tx
        self.center_crop_size = center_crop_size
        self.cld_shdw_fpaths = cld_shdw_fpaths
        self.format = format
        self.target_mode = target_mode
        self.max_diff = max_diff
        self.do_preprocess = do_preprocess
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

    def __len__(self):
        return len(self.dataset)

    def sample_cld_shdw(self):
        """Randomly sample clouds from existing cloud masks in the dataset.
        cld_shdw_fpaths: list of file path to all cloud and shadow masks in the dataset,
        where cloud in channel 1, shadow in channel 2."""
        while True: # Retry until a valid cloud shadow mask is loaded
            try:
                idx = torch.randint(0, len(self.cld_shdw_fpaths), (1,)).item()
                cld_shdw_fpath = self.cld_shdw_fpaths[idx]
                cld_shdw = self.load_and_center_crop(cld_shdw_fpath, self.channels["cld_shdw"], self.center_crop_size)
                cld_shdw = self.preprocess(cld_shdw, "cld_shdw", do_preprocess=self.do_preprocess)
                break
            except Exception as e:
                print(e)
                continue
        if torch.rand(1).item() > 0.5:
            cld_shdw = torch.flip(cld_shdw, dims=[1])
        if torch.rand(1).item() > 0.5:
            cld_shdw = torch.flip(cld_shdw, dims=[2])
        if torch.rand(1).item() > 0.5:
            cld_shdw = torch.rot90(cld_shdw, k=1, dims=[1, 2])
        return cld_shdw

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
    def preprocess(image, sensor_name, do_preprocess=True):
        """Set do_preprocess to False to skip preprocessing for benchmarking purposes."""
        if not do_preprocess:
            return image
        else:
            if sensor_name == "s2_toa":
                image = torch.clip(image, 0, 10000) / 10000
                image = torch.nan_to_num(image, nan=0)
            elif sensor_name == "s1":
                image[image < -40] = -40
                image[0] = torch.clip(image[0] + 25, 0, 25) / 25
                image[1] = torch.clip(image[1] + 32.5, 0, 32.5) / 32.5
                image = torch.nan_to_num(image, nan=-1)
            elif sensor_name == "cld_shdw":
                # mask = torch.isnan(image[0]) | (image[0] < 0.0001)
                # image[0][mask] = 1
                # image[1][mask] = 0
                image = torch.nan_to_num(image, nan=1)
                pass
            elif sensor_name in ["dw"]:
                image = image
            else:  # TODO: Implement preprocessing for other sensors
                print(f'Preprocessing steps for {sensor_name} has not been implemented yet.')
                image = image
            return image

    def __getitem__(self, idx):
        # with Profile() as prof:
        sample = self.dataset[str(idx)]
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
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
                image = self.load_and_center_crop(fpath, self.channels[sensor], self.center_crop_size)
                image = self.preprocess(image, sensor, do_preprocess=self.do_preprocess)
                inputs[sensor].append((timestamp, image))
                if sensor == self.main_sensor:
                    timestamps.append(timestamp)
                    if "cld_shdw" in self.aux_data:
                        cld_shdw_fpath = fpath.replace("s2_toa", "cld_shdw")
                        if os.path.exists(cld_shdw_fpath):
                            cld_shdw = self.load_and_center_crop(cld_shdw_fpath, self.channels["cld_shdw"],
                                                                 self.center_crop_size)
                            cld_shdw = self.preprocess(cld_shdw, "cld_shdw", do_preprocess=self.do_preprocess)
                        else:
                            raise ValueError(f"Cloud shadow file not found: {cld_shdw_fpath}")
                        inputs["input_cld_shdw"].append((timestamp, cld_shdw))
                    if "dw" in self.aux_data:
                        dw_fpath = fpath.replace("s2_toa", "dw")
                        if os.path.exists(dw_fpath):
                            dw = self.load_and_center_crop(dw_fpath, self.channels["dw"], self.center_crop_size)
                            dw = self.preprocess(dw, "dw", do_preprocess=self.do_preprocess)
                            inputs["input_dw"].append((timestamp, dw))
                        else:
                            inputs["input_dw"].append((timestamp, torch.ones(len(self.channels['dw']), *self.center_crop_size)))  # placeholder
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
            image = self.load_and_center_crop(fpath, self.channels[self.main_sensor], self.center_crop_size)
            image = self.preprocess(image, self.main_sensor, do_preprocess=self.do_preprocess)  # target by default is the main sensor
            inputs["target"] = [(timestamp, image)]
            if "cld_shdw" in self.aux_data:
                cld_shdw_fpath = fpath.replace("s2_toa", "cld_shdw")
                if os.path.exists(cld_shdw_fpath):
                    cld_shdw = self.load_and_center_crop(cld_shdw_fpath, self.channels["cld_shdw"],
                                                         self.center_crop_size)
                    cld_shdw = self.preprocess(cld_shdw, "cld_shdw", do_preprocess=self.do_preprocess)
                else:
                    raise ValueError(f"Cloud shadow file not found: {cld_shdw_fpath}")
                inputs["target_cld_shdw"] = cld_shdw.unsqueeze(0)
            else:
                inputs["target_cld_shdw"] = None
            if "dw" in self.aux_data:
                dw_fpath = fpath.replace("s2_toa", "dw")
                if not os.path.exists(dw_fpath):
                    # find a nearby dw image
                    dw_fpaths = os.listdir(os.path.dirname(dw_fpath))
                    while len(dw_fpaths) == 0:
                        if timestamp.month >= 1 and timestamp.month < 12:
                            dw_fpath = dw_fpath.replace(f"{timestamp.month}", f"{timestamp.month + 1}")
                        else:
                            dw_fpath = dw_fpath.replace(f"{timestamp.month}", f"{timestamp.month - 1}")
                        dw_fpaths = os.listdir(os.path.dirname(dw_fpath))
                    dw_fpath = os.path.join(os.path.dirname(dw_fpath), dw_fpaths[0])
                dw = self.load_and_center_crop(dw_fpath, self.channels["dw"], self.center_crop_size)
                dw = self.preprocess(dw, "dw", do_preprocess=self.do_preprocess)
                inputs["target_dw"] = dw.unsqueeze(0)
            else:
                inputs["target_dw"] = None

        elif self.target_mode == "s2s":
            if self.cld_shdw_fpaths is None:
                raise ValueError("Cloud and shadow masks are not available.")
            synthetic_inputs_main_sensor = copy.deepcopy(inputs_main_sensor)
            synthetic_clds_shdws = torch.zeros_like(inputs_cld_shdw, dtype=torch.float16)
            for i in range(inputs_cld_shdw.shape[0]):
                # sampled_cld_shdw = erode_dilate_cld_shdw(self.sample_cld_shdw()) * random_opacity()
                sampled_cld_shdw = self.sample_cld_shdw() * random_opacity()
                squared_cld_shdw = square_cld_shdw() * random_opacity()
                synthetic_cld_shdw = torch.max(sampled_cld_shdw, squared_cld_shdw)
                synthetic_cld_shdw[1] *= (synthetic_cld_shdw[0] > 0)  # no shdw on cld
                synthetic_clds_shdws[i] = synthetic_cld_shdw  # Shape: (T, 2, H, W)
            synthetic_inputs_main_sensor += synthetic_clds_shdws[:, 0, ...].unsqueeze(1)
            synthetic_inputs_main_sensor -= synthetic_clds_shdws[:, 1, ...].unsqueeze(1)
            inputs['target'] = inputs[self.main_sensor]
            inputs[self.main_sensor] = [(timestamp, syncld_main_sensor) for timestamp, syncld_main_sensor in
                                        zip(timestamps_main_sensor, synthetic_inputs_main_sensor)]
            inputs["target_cld_shdw"] = inputs_cld_shdw if "cld_shdw" in self.aux_data else None
            inputs["input_cld_shdw"] = torch.max(inputs_cld_shdw, synthetic_clds_shdws)
            inputs["input_cld_shdw"][1] *= (inputs["input_cld_shdw"][0] > 0)  # no shdw on cld
            inputs["target_dw"] = inputs_dw if "dw" in self.aux_data else None

        # final data organization
        # When a day is missing, insert 1s. (shape: (T, C, H, W) -> (C, T, H, W))
        # for spatial-temporal patch (stp) format, we align aux images with the main one if they are within the max_diff
        # and discard the rest of them to reduce the input sparsity.
        if self.format != "stp":
            raise ValueError(f"The {self.format} format is not supported.")
        else:
            output_sensors = self.sensors
            sample_stp = torch.ones((self.tx,
                                     sum([len(self.channels[sensor]) for sensor in output_sensors]),
                                     self.center_crop_size[0],
                                     self.center_crop_size[1]))

            # merge landsat8 and landsat9 into one sensor for less sparse input.
            channel_start_index = 0
            channel_start_indices = {}
            for sensor in output_sensors:
                if sensor == "landsat8" and "landsat9" in output_sensors:
                    channel_start_indices[sensor] = channel_start_indices["landsat9"]
                elif sensor == "landsat9" and "landsat8" in output_sensors:
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
                        if day_index == None: continue
                        image_channel_size = image.shape[0]
                        channel_start_index_sensor = channel_start_indices[sensor]
                        sample_stp[day_index,
                        channel_start_index_sensor:channel_start_index_sensor + image_channel_size, ...] = image

            sample_stp = sample_stp.permute(1, 0, 2, 3)
            if self.target_mode == "s2p":
                target_image = inputs["target"][0][1]
                target_image = target_image.unsqueeze(1)
            elif self.target_mode == "s2s":
                target_image = inputs_main_sensor.permute(1, 0, 2, 3)

            # (Stats(prof).strip_dirs().sort_stats(SortKey.TIME).print_stats())
            item_dict = {
                "input_images": sample_stp,  # Shape: (C1(main+aux), T, H, W)
                "target": target_image,  # Shape: (C2(main_sensor), T, H, W)
                "input_cld_shdw": inputs_cld_shdw.permute(1, 0, 2, 3) if inputs_cld_shdw is not None else None,
                # Shape: (2, T, H, W)
                "target_cld_shdw": inputs["target_cld_shdw"].permute(1, 0, 2, 3) if inputs[
                                                                                        "target_cld_shdw"] is not None else None,
                # Shape: (1, T, H, W)
                "input_dw": inputs_dw.permute(1, 0, 2, 3) if inputs_dw is not None else None,  # Shape: (C, T, H, W)
                "target_dw": inputs["target_dw"].permute(1, 0, 2, 3) if inputs["target_dw"] is not None else None,
                # Shape: (T, H, W)
                "timestamps": torch.tensor(timestamps),  # TODO: implement this correctly.
                "time_differences": torch.Tensor(time_differences),  # TODO: implement this correctly.
                "latlong": latlong,
            }
            item_dict = {k: v for k, v in item_dict.items() if v is not None}

            return item_dict

        # return {
        #     "input_images": input_images,  # Shape: (T, C, H, W)
        #     "target": target_image,  # Shape: (T, C, H, W)
        #     "target_cloud_mask": target_cloud_mask,  # Shape: (1, 1, H, W)
        #     "target_shadow_mask": target_shadow_mask,  # Shape: (1, 1, H, W)
        #     "target_lulc_label": target_lulc_label,  # Shape: (T)
        #     "target_lulc_map": target_lulc_map,  # Shape: (1, 1, H, W)
        #     "timestamps": timestamps,  # Shape: (T)
        #     "input_cloud_masks": input_cloud_masks,  # Shape: (T, 1, H, W)
        #     "input_shadow_masks": input_shadow_masks,  # Shape: (T, 1, H, W)
        # }

# TODO: Example usage
