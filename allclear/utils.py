import rasterio as rs
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def cloud_mask_threshold(cloud_prob_map, threshold=30):
    """Create a binary cloud mask based on the cloud probability map and a threshold."""
    cloud_mask = cloud_prob_map > threshold
    return cloud_mask


def visualize_with_grid(
    msi=None,
    msi_metadata=None,
    msi_channels=(3, 2, 1),
    sar=None,
    sar_metadata=None,
    sar_channels=(0,),
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
):
    """
    Visualize a multi-channel MSI with optional overlays and a grid.
    """
    if msi is not None:
        # Load MSI data
        with rs.open(msi) as src:
            msi_data = src.read()
    elif sar is not None:
        # Load SAR data
        with rs.open(sar) as src:
            sar_data = src.read()
        msi_data = np.zeros((3, *sar_data.shape[1:]))
        msi_data[0, ...] = sar_data[sar_channels[0], ...]
        msi_data[1, ...] = sar_data[sar_channels[0], ...]
        msi_data[2, ...] = sar_data[sar_channels[0], ...]

    # Initialize overlay list
    overlays = []

    # Load cloud data if specified
    if cloud and cloud_channel is not None:
        with rs.open(cloud) as src:
            cloud_data = src.read()
        overlays.append((cloud_data[cloud_channel, ...], cloud_color))

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

    # Setup figure
    plt.figure(figsize=(12, 12), dpi=200)

    if msi is not None:
        # Process and display MSI image
        p2, p98 = np.percentile(msi_data[msi_channels, ...], [2, 98])
        msi_normalized = np.clip((msi_data[msi_channels, ...] - p2) / (p98 - p2), 0, 1)
        plt.imshow(msi_normalized.transpose((1, 2, 0)), interpolation="nearest")
    elif sar is not None:
        # Process and display SAR image
        p2, p98 = np.percentile(sar_data[sar_channels, ...], [2, 98])
        sar_normalized = np.clip((sar_data[sar_channels, ...] - p2) / (p98 - p2), 0, 1)
        plt.imshow(sar_normalized.transpose((1, 2, 0)), interpolation="nearest", cmap="gray")

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

    # Setup grid and metadata if metadata is available
    if msi_metadata:
        if isinstance(msi_metadata, dict):
            plt.title(
                f'ROI (Lat, Long): {msi_metadata["ROI_ID"]} ({msi_metadata["x"]}, {msi_metadata["y"]}), Date: {msi_metadata["Date"]}, Satellite: {msi_metadata["Satellite"]}'
            )
        else:
            metadata = np.load(msi_metadata, allow_pickle=True).item()
            plt.title(f'ROI: {metadata["ROI ID"]}, Date: {metadata["Date"]}, Satellite: {metadata["Satellite"]}')
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, msi_data.shape[2], 256), minor=True)
    ax.set_yticks(np.arange(-0.5, msi_data.shape[1], 256), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.grid(which="major", visible=False)
    plt.xlabel("Longtitude (X)")
    plt.ylabel("Latitude (Y)")

    # Save the figure if a directory is provided
    if save_dir:
        save_path = os.path.join(save_dir, f'roi{metadata["ROI ID"]}_{metadata["Date"]}_{metadata["Satellite"]}_visualization.png')
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


def visualize(
    roi_id,
    date,
    satellite="s2",
    patch_id=None,
    root="/share/hariharan/cloud_removal/MultiSensor/dataset",
    save_dir="/share/hariharan/cloud_removal/results/dataset/msi_cloud_shadow",
    save=False,
):
    """date: y_m_d"""
    y, m, d = date.split("_")
    roi = f"roi{roi_id}"
    msi_fname = f"{roi}_{satellite}_{date}_median.cog"
    msi_fpath = os.path.join(root, roi, f"{y}_{m}", satellite, msi_fname)
    shadow_cloud_fpath = msi_fpath.replace(satellite, "shadow")
    with rs.open(msi_fpath) as src:
        msi = src.read()
    with rs.open(shadow_cloud_fpath) as src:
        cloud_prob = src.read(1)
        cloud_mask_30 = cloud_mask_threshold(cloud_prob, 30)
        shadow_mask = src.read(3)
        shadow_mask = np.where(np.isnan(shadow_mask), 1, shadow_mask)
    metadata = {"ROI ID": roi_id, "Date": date, "Satellite": satellite}
    if save:
        visualize_with_grid(msi, metadata, overlays=[cloud_mask_30, shadow_mask], overlay_colors=["red", "blue"], save_dir=save_dir)
    else:
        visualize_with_grid(msi, metadata, overlays=[cloud_mask_30, shadow_mask], overlay_colors=["red", "blue"])
