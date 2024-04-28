import random
import numpy as np
import torch
import rasterio as rs
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
import glob
from torchvision.transforms import GaussianBlur
import re

class CogDataset_v45(Dataset):
    def __init__(self, num_frames = 10, verbose=False, mode="test", image_size=256):
        self.dataset_path = Path("/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2/spatio_temporal")
        self.num_frames = num_frames
        self.load_spatio_temporal_info()
        self.verbose = verbose
        self.mode = mode

    def __len__(self):
        return 5000
    
    def transforms(self, msi):

        # Horizontal flip
        if torch.rand(1) > 0.5:
            msi = msi.flip(2)

        # Vertical flip
        if torch.rand(1) > 0.5:
            msi = msi.flip(3)

        # Rotation 90*n degrees
        if torch.rand(1) < 0.75:
            n = random.randint(1, 3)
            msi = msi.rot90(n, [2, 3])

        return msi
    
    # def temporal_permutation(self, msi, day_counts):

    #     # check msi shape
    #     C, T, H, W = msi.shape
    #     assert T % 2 == 0
        
    #     # mask = np.random.choice([True, False], size=(1, T, 1, 1), p=[0.5, 0.5])
    #     # y = np.empty_like(msi)
    #     # y[:, ::2, :, :] = msi[:, ::2, :, :] * mask[:, ::2, :, :] + msi[:, 1::2, :, :] * ~mask[:, ::2, :, :]
    #     # y[:, 1::2, :, :] = msi[:, 1::2, :, :] * mask[:, 1::2, :, :] + msi[:, ::2, :, :] * ~mask[:, 1::2, :, :]

    #     mask = np.random.choice([True, False], size=(1, T, 1, 1), p=[0.5, 0.5])
    #     y = np.empty_like(msi)
    #     y[:, ::2, :, :] = msi[:, ::2, :, :] * mask[:, ::2, :, :] + msi[:, 1::2, :, :] * ~mask[:, ::2, :, :]
    #     y[:, 1::2, :, :] = msi[:, 1::2, :, :] * mask[:, 1::2, :, :] + msi[:, ::2, :, :] * ~mask[:, 1::2, :, :]

    #     assert y.shape == msi.shape, y.shape

    #     # Permute day_counts and dates
    #     mask = mask.reshape((T))
    #     z = np.empty_like(day_counts)
    #     z[::2] = day_counts[::2] * mask[::2] + day_counts[1::2] * ~mask[::2]
    #     z[1::2] = day_counts[1::2] * mask[1::2] + day_counts[::2] * ~mask[1::2]

    #     return y, z

    def __getitem__(self, idx):

        # randomly select a row in self.roi_spatio_temporal_info
        row = self.roi_spatio_temporal_info.iloc[random.randint(0, len(self.roi_spatio_temporal_info)-1)]
        roi = row["roi_id"]
        patch_id = row["patch_id"]
        day_counts = row["day_count"]
        dates = row["dates"]
        
        day_random_idx = random.randint(0, len(day_counts)-self.num_frames)        
        FILE_PATH = os.path.join(self.dataset_path, f"{roi}_patch{patch_id}.cog")
        WINDOW = rs.windows.Window(0, day_random_idx * 256, 256, 256 * self.num_frames)

        if self.verbose:
            print(f"""{roi} | patch {patch_id} | day {day_random_idx} | latitude {row["latitude"]:.3f} | longtitude {row["longtitude"]:.3f}""")
            print(f"start date: {day_counts[day_random_idx]} | end date: {day_counts[day_random_idx+self.num_frames]} ")
            print(f"start date: {dates[day_random_idx]} | end date: {dates[day_random_idx+self.num_frames]} ")
        
        with rs.open(FILE_PATH) as src:
            msi = torch.from_numpy(src.read(list(range(1, 16)), window=WINDOW)).float()
            
        # Scale back MSI to raw values. 
        # MSI ranges [0, 10K], SAR ranges [0, 32.5*1K], Cloud[0] ranges [0,100], Cloud[1:5] ranges [0,1]
        msi[:10] *= 1 / 10000
        msi[10:12] *= 1 / 1000
        msi[10] *= 1 / 25
        msi[11] *= 1 / 32.5
        msi[12] *= 1 / 100
        
        # print(msi.shape, self.num_frames)
        assert msi.shape == (15, 256 * self.num_frames, 256)
        msi = msi.reshape(15, self.num_frames, 256, 256)

        meta_info = torch.Tensor([row["latitude"], row["longtitude"], int(dates[day_random_idx].split("_")[1])])
        day_counts = torch.Tensor(day_counts[day_random_idx: day_random_idx+self.num_frames])
        dates = dates[day_random_idx: day_random_idx+self.num_frames]
        
        # if self.mode == "train":
        #     msi = self.transforms(msi)
            # msi, day_counts  = self.temporal_permutation(msi, day_counts)
        
        return msi, meta_info, day_counts, dates

    def load_spatio_temporal_info(self):
        csv_list = glob.glob("/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2/spatio_temporal/roi*.csv")
        self.roi_spatio_temporal_info = []
        for csv_file in csv_list:
            df = pd.read_csv(csv_file)
            if len(self.roi_spatio_temporal_info) == 0:
                df["day_count"] = df['day_count'].apply(lambda x: torch.Tensor([int(num) for num in re.findall(r'\d+', x)]))
                df["dates"] = df['dates'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace("\n", "").split())
                self.roi_spatio_temporal_info = df
            else:
                df["day_count"] = df['day_count'].apply(lambda x: [int(num) for num in re.findall(r'\d+', x)])
                df["dates"] = df['dates'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace("\n", "").split())
                self.roi_spatio_temporal_info = pd.concat([self.roi_spatio_temporal_info, df], ignore_index=True, axis=0)

# batch_size = 4
# dataset = CogDataset_v45(num_frames=10, verbose=False)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
# # dataset.roi_spatio_temporal_info