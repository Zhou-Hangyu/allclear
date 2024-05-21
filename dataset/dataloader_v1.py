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


def sample_cld_shd(clds_shds):
    """Randomly sample clouds from existing cloud masks in the dataset."""
    idx = torch.randint(0, len(clds_shds), (1,)).item()
    cld_shd = clds_shds[idx]
    if torch.rand(1).item() > 0.5:
        cld_shd = torch.flip(cld_shd, dims=[1])
    if torch.rand(1).item() > 0.5:
        cld_shd = torch.flip(cld_shd, dims=[2])
    if torch.rand(1).item() > 0.5:
        cld_shd = torch.rot90(cld_shd, k=1, dims=[1, 2])
    return cld_shd

def square_cld_shd(image_size=256):
    """Generate square cloud & shadow masks."""
    mask_scale = np.random.choice([16, 32, 64])
    threshold = np.random.uniform(low=0.1, high=0.25)
    cld = (torch.rand((1, image_size, image_size)) < threshold / (mask_scale * 2) ** 2).float()
    shd = (torch.rand((1, image_size, image_size)) < threshold / (mask_scale * 2) ** 2).float()
    square_cld = F.max_pool2d(cld, mask_scale + 1, stride=1, padding=int((mask_scale + 1) // 2))
    square_shd = F.max_pool2d(shd, mask_scale + 1, stride=1, padding=int((mask_scale + 1) // 2))
    square_cld_shd = torch.cat([square_cld, square_shd], dim=0)
    return square_cld_shd

def erode_dilate_cld_shd(cld_shd, mask_dilation_kernel=7, blur_kernel_size=3, blur_sigma=1):
    """
    Apply erosion and dilation to cloud and shadow masks.

    Args:
        cld_shd (torch.tensor): A torch tensor of shape (2, h, w) containing cloud and shadow masks.
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

    processed_cld = process_mask(cld_shd[0])
    processed_shd = process_mask(cld_shd[1])

    return torch.cat([processed_cld, processed_shd], dim=0)


def random_opacity():
    """Generate a random opacity value."""
    return torch.rand(1).item() * 0.5


class CRDataset(Dataset):
    """
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Retrieves the target and input images, masks, and timestamps for a given index.

    Example:
        data = {
            1: {"ROI": "roi1234", "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value1"},
            2: {"ROI": "roi5678", "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value2"},
            3: {"ROI": "roi9101", "s2_toa": [(time, fpath), (time, fpath)], "other_key": "value3"}
        }
        dataset = CRDataset('dataset.json', ['roi1', 'roi2'], 3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    """

    def __init__(self, metadata, selected_rois, num_frames, clds_shds, mode="stp"):
        
        if mode == "stp":
            self.metadata = {ID: info for ID, info in metadata.items() if info["ROI"][0] in selected_rois}
            self.num_frames = num_frames
            self.clds_shds = clds_shds
            self.mode = mode
            
        elif mode == "seq2point":
            self.metadata = metadata
            self.num_frames = num_frames
            self.mode = mode
            

    def __len__(self):
        return len(self.metadata.keys())

    def __getitem__(self, idx):
        if self.mode == "stp":
            sample = self.metadata[idx]
            latlong = sample["ROI"][1]
            sensors = sorted([sensor for sensor in sample.keys() if sensor != "ROI"])
        elif self.mode == "seq2point":
            sample = self.metadata[str(idx)]
            latlong = sample["roi"][1]
            sensors = sorted([sensor for sensor in sample.keys() if sensor != "roi"])
            
        if "s2_toa" not in sample:
            raise ValueError("The sample does not contain Sentinel-2 TOA data.")

        # load in images as sets of (timestamp, image) pairs
        inputs = {sensor: [] for sensor in sensors}
        for sensor in sensors:
            sensor_inputs = sample[sensor]
            for sensor_input in sensor_inputs:
                timestamp, fpath = sensor_input
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
                with rs.open(fpath) as src:
                    image = src.read()
                image = torch.from_numpy(image).float()
                try:
                    image = F.center_crop(image, (256, 256))
                except:
                    image = center_crop(image, (256, 256))
                    
                if sensor == "s2_toa":
                    image = np.clip(image, 0, 10000) / 10000
                
                elif sensor == "s1":
                    image[image<-40] = -40
                    image[0] = np.clip(image[0] + 25, 0, 25) / 25
                    image[1] = np.clip(image[1] + 32.5, 0, 32.5) / 32.5
                    image = np.nan_to_num(image, nan=-1)
                    
                inputs[sensor].append([timestamp, image])
                
        # sort images by timestamp
        timestamps = []
        for sensor in sensors:
            inputs[sensor] = sorted(inputs[sensor], key=lambda x: x[0])
            timestamps.extend([timestamp for timestamp, _ in inputs[sensor]])
            
        
        if self.mode == "stp":
        
            # synthetic cloud and shadow masks on s2_toa
            inputs_s2_toa = torch.stack(inputs["s2_toa"])
            inputs_cld_shd = torch.stack([erode_dilate_cld_shd(cld_shd) for cld_shd in inputs["cld_shd"]])
            synthetic_inputs_s2_toa = copy.deepcopy(inputs_s2_toa)
            synthetic_clds_shds = torch.zeros_like(inputs_cld_shd, dtype=torch.float16)
            for i in range(inputs_cld_shd.shape[0]):
                sampled_cld_shd = erode_dilate_cld_shd(sample_cld_shd(self.clds_shds)) * random_opacity()
                squared_cld_shd = square_cld_shd() * random_opacity()
                synthetic_cld_shd = torch.max(sampled_cld_shd, squared_cld_shd)
                synthetic_cld_shd[1] *= (synthetic_cld_shd[0] > 0)  # no shd on cld
                synthetic_clds_shds[i] = synthetic_cld_shd  # Shape: (T, 2, H, W)
            synthetic_inputs_s2_toa += synthetic_clds_shds[:, 0, ...]
            synthetic_inputs_s2_toa -= synthetic_clds_shds[:, 1, ...]
            inputs['s2_toa_synthetic'] = synthetic_inputs_s2_toa

            # format the sample
            # When a day is missing, insert 1s.
            all_timestamps = sorted(set(timestamps))
            start_date = all_timestamps[0]
            end_date = all_timestamps[-1]
            tx = (end_date - start_date).days + 1
            if tx != self.num_frames:
                print(f"Error: {tx} != {self.num_frames}")
                return None
            output_sensors = ["s2_toa_synthetic"] + [sensor for sensor in sensors if sensor != "s2_toa"]
            sample_stp = torch.ones((tx,
                                     sum([inputs[sensor][0][1].shape[0] for sensor in output_sensors]),
                                     inputs[sensor][0][1].shape[-2],
                                     inputs[sensor][0][1].shape[-1]))
            channel_start_index = 0
            for sensor in output_sensors:
                for timestamp, image in inputs[sensor]:
                    day_index = (timestamp - start_date).days
                    sample_stp[day_index, channel_start_index:channel_start_index + image.shape[0], ...] = image
                channel_start_index += image.shape[0]
            # swap axes to (C, T, H, W)
            sample_stp = sample_stp.permute(1, 0, 2, 3)
            inputs_cld_shd = inputs_cld_shd.permute(1, 0, 2, 3)
            return sample_stp, inputs_cld_shd, all_timestamps, latlong
        
        elif self.mode == "seq2point":
            all_timestamps = sorted(set(timestamps))
            start_date = all_timestamps[0]
            time_differences = [round((timestamp - start_date).total_seconds() / (24 * 3600)) for timestamp in all_timestamps]
            for sensor in inputs:
                for i in range(len(inputs[sensor])):
                    inputs[sensor][i][0] = str(inputs[sensor][i][0])
            return inputs, time_differences
        else:
            return inputs, latlong
