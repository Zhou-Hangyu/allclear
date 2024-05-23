import copy
from datetime import datetime, timedelta
import json
import numpy as np
import torch
import rasterio as rs
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import center_crop

from cProfile import Profile
from pstats import SortKey, Stats


def sample_cld_shdw(clds_shdws):
    """Randomly sample clouds from existing cloud masks in the dataset."""
    idx = torch.randint(0, len(clds_shdws), (1,)).item()
    cld_shdw = clds_shdws[idx]
    if torch.rand(1).item() > 0.5:
        cld_shdw = torch.flip(cld_shdw, dims=[1])
    if torch.rand(1).item() > 0.5:
        cld_shdw = torch.flip(cld_shdw, dims=[2])
    if torch.rand(1).item() > 0.5:
        cld_shdw = torch.rot90(cld_shdw, k=1, dims=[1, 2])
    return cld_shdw


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

def erode_dilate_cld_shdw(cld_shdw, mask_dilation_kernel=7, blur_kernel_size=3, blur_sigma=1):
    """
    Apply erosion and dilation to cloud and shadow masks.

    Args:
        cld_shdw (torch.tensor): A torch tensor of shape (2, h, w) containing cloud and shadow masks.
        mask_dilation_kernel (int): The kernel size for the dilation operation.
        blur_kernel_size (int): The kernel size for the Gaussian blur.
        blur_sigma (float): The sigma for the Gaussian blur.

    Returns:
        torch.Tensor: The processed masks after erosion and dilation.
    """
    blur_kernel = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)

    def process_mask(mask):
        eroded_mask = -F.max_pool2d(mask * -1, kernel_size=3, stride=1, padding=1)  # Erosion
        eroded_mask = blur_kernel(eroded_mask)
        dilated_mask = F.max_pool2d(eroded_mask, kernel_size=mask_dilation_kernel, stride=1,
                                    padding=mask_dilation_kernel // 2)  # Dilation
        dilated_mask = blur_kernel(dilated_mask)
        return dilated_mask

    processed_cld = process_mask(cld_shdw[0])
    processed_shdw = process_mask(cld_shdw[1])

    return torch.cat([processed_cld, processed_shdw], dim=0)


def random_opacity():
    """Generate a random opacity value."""
    return torch.rand(1).item() * 0.5


def temporal_align_aux_sensors(main_sensor_timestamps, aux_sensor_timestamp, max_diff=2):
    differences = [abs(dt - aux_sensor_timestamp) for dt in main_sensor_timestamps]
    if min(differences).days > max_diff:
        return None
    else:
        return differences.index(min(differences))


def load_image(fpath, channels=None, center_crop_size=(256, 256)):
    with rs.open(fpath) as src:
        if channels is None:
            channels = list(range(src.count)) + 1
        image = src.read(channels)
    image = torch.from_numpy(image).float()
    try:
        image = F.center_crop(image, center_crop_size)
    except:
        image = center_crop(image, center_crop_size)
    return image


def preprocess(image, sensor_name):
    if sensor_name == "s2_toa":
        image = torch.clip(image, 0, 10000) / 10000
    elif sensor_name == "s1":
        image[image < -40] = -40
        image[0] = torch.clip(image[0] + 25, 0, 25) / 25
        image[1] = torch.clip(image[1] + 32.5, 0, 32.5) / 32.5
        image = torch.nan_to_num(image, nan=-1)
    elif sensor_name in ["cld_shdw", "dw"]:
        image = image
    else:  # TODO: Implement preprocessing for other sensors
        print(f'Preprocessing steps for {sensor_name} has not been implemented yet.')
        image = image
    return image


class CRDataset(Dataset):
    """
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Retrieves the target and input images, masks, and timestamps for a given index.

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
                 clds_shdws=None,
                 format="stp",
                 target="s2s",
                 max_diff=2):
        if aux_sensors is None:
            aux_sensors = ["s1", "landsat8", "landsat9"]
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
        self.clds_shdws = clds_shdws
        self.format = format
        self.target = target
        self.max_diff = max_diff
        if self.format != "stp":
            raise ValueError("The format is not supported.")
        self.channels = {
            "s2_toa": list(range(1, 14)),
            "s1": [1, 2],
            "landsat8": list(range(1, 12)),
            "landsat9": list(range(1, 12)),
            "cld_shdw": [2, 5]
        }

    def __len__(self):
        return len(self.dataset.keys())

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
                image = load_image(fpath, self.channels[sensor], self.center_crop_size)
                image = preprocess(image, sensor)
                inputs[sensor].append((timestamp, image))
                if sensor == self.main_sensor:
                    timestamps.append(timestamp)
                    if "cld_shdw" in self.aux_data:
                        cld_shdw_fpath = fpath.replace("s2_toa", "cld_shdw")
                        cld_shdw = load_image(cld_shdw_fpath, self.channels["cld_shdw"], self.center_crop_size)
                        cld_shdw = preprocess(cld_shdw, "cld_shdw")
                        inputs["input_cld_shdw"].append((timestamp, cld_shdw))
                    if "dw" in self.aux_data:
                        dw_fpath = fpath.replace("s2_toa", "dw")
                        dw = load_image(dw_fpath, self.channels["dw"], self.center_crop_size)
                        dw = preprocess(dw, "dw")
                        inputs["input_dw"].append((timestamp, dw))
            inputs[sensor] = sorted(inputs[sensor], key=lambda x: x[0])
        timestamps = sorted(set(timestamps))
        timestamps_main_sensor = [timestamp for timestamp, _ in inputs[self.main_sensor]]
        start_date, end_date = timestamps[0], timestamps[-1]
        time_differences = [round((timestamp - start_date).total_seconds() / (24 * 3600)) for timestamp in
                            timestamps]
        
        if self.format == "stp":  # TODO: refactor this code once more format is added.
            inputs_main_sensor = torch.stack([image for _, image in inputs[self.main_sensor]])
            if "cld_shdw" in self.aux_data:
                # inputs_cld_shdw = torch.stack([erode_dilate_cld_shdw(cld_shdw) for cld_shdw in inputs["input_cld_shdw"]])
                inputs_cld_shdw = torch.stack([cld_shdw for _, cld_shdw in inputs["input_cld_shdw"]])
            else:
                inputs_cld_shdw = None
            if "dw" in self.aux_data:
                inputs_dw = torch.stack([dw for _, dw in inputs["input_dw"]])
            else:
                inputs_dw = None

        # load in targets. If seq2seq (s2s), target use syncld; if seq2point (s2p), target use predefined target clear image.
        if self.target == "s2p":
            if "target" not in sample.keys():
                raise ValueError("Target is not available in the sample.")
            timestamp, fpath = sample["target"][0]
            image = load_image(fpath, self.channels[self.main_sensor], self.center_crop_size)
            image = preprocess(image, self.main_sensor)  # target by default is the main sensor
            inputs["target"] = [(timestamp, image)]
            if "cld_shdw" in self.aux_data:
                cld_shdw_fpath = fpath.replace("s2_toa", "cld_shdw")
                cld_shdw = load_image(cld_shdw_fpath, self.channels["cld_shdw"], self.center_crop_size)
                cld_shdw = preprocess(cld_shdw, "cld_shdw")
                inputs["target_cld_shdw"] = cld_shdw.unsqueeze(0)
            else:
                inputs["target_cld_shdw"] = None
            if "dw" in self.aux_data:
                dw_fpath = fpath.replace("s2_toa", "dw")
                dw = load_image(dw_fpath, self.channels["dw"], self.center_crop_size)
                dw = preprocess(dw, "dw")
                inputs["target_dw"] = dw.unsqueeze(0)
            else:
                inputs["target_dw"] = None

        elif self.target == "s2s":
            if self.clds_shdws is None:
                raise ValueError("Cloud and shadow masks are not available.")
            synthetic_inputs_main_sensor = copy.deepcopy(inputs_main_sensor)
            synthetic_clds_shdws = torch.zeros_like(inputs_cld_shdw, dtype=torch.float16)
            for i in range(inputs_cld_shdw.shape[0]):
                # sampled_cld_shdw = erode_dilate_cld_shdw(sample_cld_shdw(self.clds_shdws)) * random_opacity()
                sampled_cld_shdw = sample_cld_shdw(self.clds_shdws) * random_opacity()
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

        # load in inputs.
        # format the sample (When a day is missing, insert 1s) (shape: (T, C, H, W) -> (C, T, H, W))
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
            if self.target == "s2p":
                target_image = inputs["target"][0][1]
                target_image = target_image.unsqueeze(1)
            elif self.target == "s2s":
                target_image = inputs_main_sensor.permute(1, 0, 2, 3)
            # (Stats(prof).strip_dirs().sort_stats(SortKey.TIME).print_stats())

            # Create the dictionary with potential None values

            item_dict = {
                "input_images": sample_stp,  # Shape: (C, T, H, W)
                "target": target_image,  # Shape: (C, T, H, W)
                "input_cld_shdw": inputs_cld_shdw.permute(1, 0, 2, 3) if inputs_cld_shdw is not None else None,
                # Shape: (2, T, H, W)
                "target_cld_shdw": inputs["target_cld_shdw"].permute(1, 0, 2, 3) if inputs["target_cld_shdw"] is not None else None,
                # Shape: (2, T, H, W)
                "input_dw": inputs_dw.permute(1, 0, 2, 3) if inputs_dw is not None else None,  # Shape: (C, T, H, W)
                "target_dw": inputs["target_dw"].permute(1, 0, 2, 3) if inputs["target_dw"] is not None else None,
                # Shape: (C, T, H, W)
                "timestamps": None,  # TODO: implement this correctly.
                "time_differences": torch.Tensor(time_differences),  # TODO: implement this correctly.
                "latlong": latlong,
            }

            # Filter out None values
            item_dict = {k: v for k, v in item_dict.items() if v is not None}

            return item_dict

# to use the dataloader, please run the following code
# import json
# with open('/share/hariharan/ck696/allclear_0520/experiments/dataset_curation_0520_NoNan/test_500_tx3.json') as f:
#     metadata = json.load(f)
# dataset = CRDataset(metadata, None, 3, None, format="seq2point")
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
# for sample, target, time in dataloader: break