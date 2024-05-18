import rasterio as rs
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import torch


def plot_lulc_metrics(metrics_data, dpi=200, save_dir=None, model_config=None):
    """
    Plot LULC metrics for each class and metric type using dynamic world v1 colors.

    Args:
    metrics_data (dict): A dictionary with class indices as keys and dictionaries of metrics as values.
    
    Example:
    metrics_data = {
        0: {'MAE': 0.1, 'RMSE': 0.2, 'PSNR': 30.0, 'SAM': 0.1, 'SSIM': 0.9},
        1: {'MAE': 0.2, 'RMSE': 0.3, 'PSNR': 29.0, 'SAM': 0.2, 'SSIM': 0.8},
        2: {'MAE': 0.3, 'RMSE': 0.4, 'PSNR': 28.0, 'SAM': 0.3, 'SSIM': 0.7},
        3: {'MAE': 0.4, 'RMSE': 0.5, 'PSNR': 27.0, 'SAM': 0.4, 'SSIM': 0.6},
        4: {'MAE': 0.5, 'RMSE': 0.6, 'PSNR': 26.0, 'SAM': 0.5, 'SSIM': 0.5},
        5: {'MAE': 0.6, 'RMSE': 0.7, 'PSNR': 25.0, 'SAM': 0.6, 'SSIM': 0.4},
        6: {'MAE': 0.7, 'RMSE': 0.8, 'PSNR': 24.0, 'SAM': 0.7, 'SSIM': 0.3},
        7: {'MAE': 0.8, 'RMSE': 0.9, 'PSNR': 23.0, 'SAM': 0.8, 'SSIM': 0.2},
        8: {'MAE': 0.9, 'RMSE': 1.0, 'PSNR': 22.0, 'SAM': 0.9, 'SSIM': 0.1}
    }
    plot_lulc_metrics(metrics_data)
    """
    dw_colors = ['#000000',  # -1 as black (unused here unless you have class -1)
                 '#419bdf',  # 0
                 '#397d49',  # 1
                 '#88b053',  # 2
                 '#7a87c6',  # 3
                 '#e49635',  # 4
                 '#dfc35a',  # 5
                 '#c4281b',  # 6
                 '#a59b8f',  # 7
                 '#b39fe1']  # 8

    # Prepare the figure and subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5), sharex=True, dpi=dpi)

    # Ensure the order of metrics is consistent across subplots
    metric_order = ['MAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']

    # Collect data for each metric to plot
    class_indices = sorted(metrics_data.keys())
    for ax, metric in zip(axs, metric_order):
        values = [metrics_data[cls].get(metric, float('nan')) for cls in class_indices]
        ax.bar(class_indices, values,
               color=[dw_colors[cls + 1] for cls in class_indices])  # cls + 1 to match the colors
        ax.set_title(metric)
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_xticks(class_indices)
        ax.grid(True)

    plt.tight_layout()
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{model_config}_lulc_metrics.png'), bbox_inches='tight')
    else:
        plt.show()



def cloud_mask_threshold(cloud_prob_map, threshold=30):
    """Create a binary cloud mask based on the cloud probability map and a threshold."""
    cloud_mask = cloud_prob_map > threshold
    return cloud_mask


def load_image_center_crop(image, center_crop=False, size=(256, 256)):
    """Load an image and optionally apply a center crop. Image shape: [C, H, W]."""
    if isinstance(image, str):
        with rs.open(image) as src:
            data = src.read()
    else:
        data = image
    if center_crop:
        width, height = data.shape[1], data.shape[2]
        center_x, center_y = width // 2, height // 2
        offset_x = center_x - size[0] // 2
        offset_y = center_y - size[1] // 2
        data = data[:, offset_y:offset_y + size[1], offset_x:offset_x + size[0]]
    return data


def visualize_one_image(
    msi=None,
    msi_channels=(3, 2, 1),
    sar=None,
    metadata=None,
    cloud=None,
    cloud_channel=None,
    cloud_color="red",
    shadow=None,
    shadow_channel=None,
    shadow_color="blue",
    lulc=None,
    lulc_channel=0,
    lulc_color=None,
    default_opacity=0.5,
    save_dir=None,
    dpi=100,
    center_crop=True,
    center_crop_shape=(256, 256),
    with_grid=False,
):
    """
    Visualize a satellite image (MSI or SAR) with optional overlays and a grid.
    """

    def normalize(array):
        '''
        normalize: normalize a numpy array so all value are between 0 and 1
        '''
        array_min, array_max = np.nanmin(array), np.nanmax(array)
        return (array - array_min) / (array_max - array_min)

    plt.figure(figsize=(12, 12), dpi=dpi)

    data = None
    if msi is not None:
        msi_data = load_image_center_crop(msi, center_crop, center_crop_shape)
        for channel in msi_channels:
            msi_data[channel, ...] = normalize(msi_data[channel, ...])
        data = msi_data
        plt.imshow(msi_data[msi_channels, ...].transpose(1, 2, 0), interpolation="nearest", vmin=0, vmax=1)
    elif sar is not None:
        sar_data = load_image_center_crop(sar, center_crop, center_crop_shape)
        sar_rgb = np.zeros((3, *sar_data.shape[1:]))
        sar_rgb[0, ...] = normalize(sar_data[0, ...])  # VV
        sar_rgb[1, ...] = normalize(sar_data[1, ...])  # VH
        sar_rgb[2, ...] = normalize(sar_data[1, ...] - sar_data[0, ...])  # VH - VV
        data = sar_rgb
        plt.imshow(sar_rgb.transpose(1, 2, 0), interpolation="nearest", vmin=0, vmax=1)

    if metadata:
        plt.title(f'ROI (Lat, Long): {metadata["roi"]} ({metadata["latitude"]}, {metadata["longitude"]}), Date: {metadata["date_y_m"]}, Satellite: {metadata["satellite"]}')
    ax = plt.gca()
    ax.grid(which="major", visible=False)
    if with_grid:
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, data.shape[2], 256), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[1], 256), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    plt.xlabel("West-East")
    plt.ylabel("North-South")

    overlays = []

    # Load cloud data if specified
    if cloud is not None:
        if isinstance(cloud, str):
            with rs.open(cloud) as src:
                cloud_data = src.read()[cloud_channel, ...]
        else:
            cloud_data = cloud
        overlays.append((cloud_data, cloud_color))

    # Load shadow data if specified
    if shadow and shadow_channel is not None:
        with rs.open(shadow) as src:
            shadow_data = src.read()
        overlays.append((shadow_data[shadow_channel, ...], shadow_color))

    # Load LULC data if specified
    if lulc is not None:
        with rs.open(lulc) as src:
            lulc_data = src.read()
        lulc_cmap = (
            ListedColormap(lulc_color)
            if lulc_color
            else ListedColormap(["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635", "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"])
        )
        # lulc_cmap.set_bad(color='black')  # Set the color for NaN values
        overlays.append((lulc_data[lulc_channel, ...], lulc_cmap))

    # Apply overlays
    for overlay_data, color in overlays:
        if isinstance(color, str):
            overlay_image = np.zeros((*overlay_data.shape, 4))  # Prepare an RGBA image
            color_rgba = matplotlib.colors.to_rgba(color)
            overlay_image[..., :3] = color_rgba[:3]  # RGB
            overlay_image[..., 3] = overlay_data * default_opacity  # Alpha
            plt.imshow(overlay_image)
        elif isinstance(color, ListedColormap):
            plt.imshow(overlay_data, cmap=color, alpha=default_opacity, vmin=0, vmax=8, interpolation="nearest")
            legend_labels = ["water", "trees", "grass", "flooded vegetation", "crops", "shrub and scrub", "built", "bare", "snow and ice"]
            legend_colors = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635", "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]
            legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(legend_colors, legend_labels)]
            plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    if save_dir:
        save_path = os.path.join(save_dir, f'roi{metadata["ROI ID"]}_{metadata["Date"]}_{metadata["Satellite"]}_visualization.png')
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


# def visualize(
#     roi_id,
#     date,
#     satellite="s2",
#     patch_id=None,
#     root="/share/hariharan/cloud_removal/MultiSensor/dataset",
#     save_dir="/share/hariharan/cloud_removal/results/dataset/msi_cloud_shadow",
#     save=False,
# ):
#     """date: y_m_d"""
#     y, m, d = date.split("_")
#     roi = f"roi{roi_id}"
#     msi_fname = f"{roi}_{satellite}_{date}_median.cog"
#     msi_fpath = os.path.join(root, roi, f"{y}_{m}", satellite, msi_fname)
#     shadow_cloud_fpath = msi_fpath.replace(satellite, "shadow")
#     with rs.open(msi_fpath) as src:
#         msi = src.read()
#     with rs.open(shadow_cloud_fpath) as src:
#         cloud_prob = src.read(1)
#         cloud_mask_30 = cloud_mask_threshold(cloud_prob, 30)
#         shadow_mask = src.read(3)
#         shadow_mask = np.where(np.isnan(shadow_mask), 1, shadow_mask)
#     metadata = {"ROI ID": roi_id, "Date": date, "Satellite": satellite}
#     if save:
#         visualize_with_grid(msi, metadata, overlays=[cloud_mask_30, shadow_mask], overlay_colors=["red", "blue"], save_dir=save_dir)
#     else:
#         visualize_with_grid(msi, metadata, overlays=[cloud_mask_30, shadow_mask], overlay_colors=["red", "blue"])
