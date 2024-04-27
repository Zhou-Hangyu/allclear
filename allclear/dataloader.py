from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import rasterio as rs
from torch.utils.data import Dataset, DataLoader


def cloud_mask_threshold(cloud_prob_map, threshold=30):
    """
    Generates a binary cloud mask from a cloud probability map using a specified threshold.

    Args:
        cloud_prob_map (np.ndarray): An array representing the cloud probability map.
        threshold (int): The threshold value to generate the binary mask. Defaults to 30.

    Returns:
        np.ndarray: A binary mask array where pixels with cloud probability above the
                    threshold are marked as True (cloudy), and the rest as False (clear).
    """
    cloud_mask = cloud_prob_map > threshold
    return cloud_mask


class CRDataset(Dataset):
    """
    A PyTorch Dataset class for loading satellite image data for cloud removal tasks.

    Attributes:
        data (pd.DataFrame): A DataFrame containing the test set entries, which includes
                             unique identifiers for target and input images.
        metadata (pd.DataFrame): A DataFrame containing metadata for all the patches,
                                 including file paths and capture dates.
        time_span (int): The number of days before and after the capture date to consider
                         for selecting input images.

    Args:
        dataset_csv (str): File path to the CSV file containing the dataset with target
                           and input image identifiers.
        patch_metadata_csv (str): File path to the CSV file containing metadata for all
                                  available image patches.
        selected_rois (list[str]): A list of selected Region of Interest (ROI) IDs to be
                                   considered in the dataset.
        time_span (int): Time span in days to search for input images around the target
                         image's capture date.
        cloud_percentage_range (tuple[int, int], optional): A range of cloud percentage
                                                            to filter the images. Defaults to None.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Retrieves the target and input images, masks, and timestamps for a given index.

    Example:
        dataset = CRDataset('dataset.csv', 'metadata.csv', ['roi1', 'roi2'], 3)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    """

    def __init__(self, dataset_csv, patch_metadata_csv, selected_rois, time_span, cloud_percentage_range=None):
        self.data = pd.read_csv(dataset_csv)
        self.data = self.data[self.data["ROI ID"].isin(selected_rois)]
        # self.data = self.data[self.data["Target"].split("_")[0][3:].isin(selected_rois)]
        self.metadata = pd.read_csv(patch_metadata_csv)
        self.time_span = time_span

        if cloud_percentage_range:
            # min_cloud, max_cloud = cloud_percentage_range
            # self.metadata = self.metadata[
            #     (self.metadata["Nonzero Cloud Percentage"] >= min_cloud) & (self.metadata["Nonzero Cloud Percentage"] <= max_cloud)
            # ]
            pass

        # self.metadata["capture_date"] = pd.to_datetime(self.metadata["capture_date"])
        # self.metadata.sort_values("capture_date", inplace=True)
        # self.metadata["year"] = self.metadata["capture_date"].dt.year

        # Filter the data to ensure necessary columns are not empty
        # self.metadata = self.metadata.dropna(subset=["ROI File Path", "Nonzero Cloud Percentage" if cloud_percentage_range else None])

        # # Pre-compute least cloudy image per year for each ROI
        # self.least_cloudy_per_year = {}
        # grouped = self.metadata.groupby(["ROI ID", "year"])
        # for name, group in grouped:
        #     least_cloudy_idx = group["Nonzero Cloud Percentage"].idxmin()
        #     self.least_cloudy_per_year[name] = self.metadata.loc[least_cloudy_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the target and inputs for the given index from the dataset CSV.
        entry = self.data.iloc[idx]
        target_id = entry["Target"]
        input_ids = entry[[col for col in self.data.columns if "Input" in col]].dropna().tolist()

        # Find the corresponding rows in the metadata for the target and inputs.
        target_metadata = self.metadata[self.metadata["uid"] == target_id].iloc[0]
        input_metadatas = [self.metadata[self.metadata["uid"] == input_id].iloc[0] for input_id in input_ids]

        # Load the target image.
        with rs.open(target_metadata["ROI File Path"].replace("s2", "s2_toa")) as src:
            target_image = src.read(window=rs.windows.Window(*eval(target_metadata["Offset"]), 256, 256))
            target_image = torch.from_numpy(target_image).float()
            target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min())

        # Initialize lists for input images, cloud masks, shadow masks, and timestamps.
        input_images = []
        cloud_masks = []
        shadow_masks = []
        timestamps = []

        # Load the input images and their corresponding masks.
        for input_metadata in input_metadatas:
            # Load input image.
            with rs.open(input_metadata["ROI File Path"].replace("s2", "s2_toa")) as src:
                image = src.read(window=rs.windows.Window(*eval(input_metadata["Offset"]), 256, 256))
                image = torch.from_numpy(image).float()
                image = (image - image.min()) / (image.max() - image.min())
                input_images.append(image)

            # Load cloud and shadow masks.
            # patch_info = {"Shadow Cloud File Path": input_metadata["Shadow Cloud File Path"], "Offset": input_metadata["Offset"]}
            # shadow_cloud_info = s2_patch_shadow_cloud_percentage(patch_info)

            with rs.open(input_metadata["Shadow Cloud File Path"]) as src:
                cloud_mask = src.read(2, window=rs.windows.Window(*eval(input_metadata["Offset"]), 256, 256))
                shadow_mask = src.read(4, window=rs.windows.Window(*eval(input_metadata["Offset"]), 256, 256))

            # cloud_prob = np.where(np.isnan(cloud_prob), 100, cloud_prob)
            # shadow_mask = np.where(np.isnan(shadow_mask), 1, shadow_mask)
            # cloud_mask = cloud_mask_threshold(cloud_prob, 30)

            cloud_masks.append(torch.from_numpy(cloud_mask).unsqueeze(0).float())
            shadow_masks.append(torch.from_numpy(shadow_mask).unsqueeze(0).float())

            # Collect timestamps.
            capture_date = datetime.strptime(input_metadata["capture_date"], "%Y-%m-%d %H:%M:%S")
            timestamps.append(capture_date.timestamp())

        # Stack the lists of images and masks into tensors.
        input_images = torch.stack(input_images)
        cloud_masks = torch.stack(cloud_masks)
        shadow_masks = torch.stack(shadow_masks)
        timestamps = torch.tensor(timestamps)

        return {
            "input_images": input_images,  # Shape: (T, C, H, W)
            "target": target_image,  # Shape: (C, H, W)
            "timestamps": timestamps,  # Shape: (T)
            "cloud_masks": cloud_masks,  # Shape: (T, 1, H, W)
            "shadow_masks": shadow_masks,  # Shape: (T, 1, H, W)
        }


class CogDataset_v41(Dataset):
    def __init__(self, num_s2_frames=10):
        self.dataset_path = Path("/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed/spatio_temporal")
        self.num_s2_frames = num_s2_frames
        self.load_spatio_temporal_info()
        self.mode = "MSI"

    def __len__(self):
        return 2048

    def __getitem__(self, idx):
        # randomly select a row in self.roi_spatio_temporal_info
        row = self.roi_spatio_temporal_info.iloc[random.randint(0, len(self.roi_spatio_temporal_info) - 1)]
        roi = row["roi"]
        patch_id = row["patch_id"]
        day_counts = row["day_count"]

        day_random_idx = random.randint(0, len(day_counts) - self.num_s2_frames)
        FILE_PATH = os.path.join(self.dataset_path, f"{roi}_patch{patch_id}.cog")
        WINDOW = rs.windows.Window(0, day_random_idx * 256, 256, 256 * self.num_s2_frames)

        with rs.open(FILE_PATH) as src:
            msi = torch.from_numpy(src.read(list(range(1, 11)), window=WINDOW))
        assert msi.shape == (10, 256 * self.num_s2_frames, 256)
        msi = msi.reshape(10, self.num_s2_frames, 256, 256)
        return msi

    def load_spatio_temporal_info(self):
        csv_list = glob.glob("/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed/spatio_temporal/roi*.csv")
        self.roi_spatio_temporal_info = []
        for csv_file in csv_list:
            df = pd.read_csv(csv_file)
            if len(self.roi_spatio_temporal_info) == 0:
                df["day_count"] = df["day_count"].apply(lambda x: [int(num) for num in re.findall(r"\d+", x)])
                self.roi_spatio_temporal_info = df
            else:
                df["day_count"] = df["day_count"].apply(lambda x: [int(num) for num in re.findall(r"\d+", x)])
                self.roi_spatio_temporal_info = pd.concat([self.roi_spatio_temporal_info, df], ignore_index=True, axis=0)


# Example usage
if __name__ == "__main__":
    patch_metadata_csv = "/share/hariharan/cloud_removal/metadata/roi40-45_s2_patches.csv"
    selected_rois = ["roi40", "roi42"]
    time_span = 3
    cloud_percentage_range = (20, 30)

    dataset = CRDataset(patch_metadata_csv, selected_rois, time_span, cloud_percentage_range)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    for data in dataloader:
        print(data["images"].shape, data["timestamps"].shape)
