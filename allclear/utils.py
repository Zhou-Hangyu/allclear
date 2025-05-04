import rasterio as rs
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

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
    vmax = [0.1, 0.1, 40, 15, 1]

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
        ax.set_ylim(0, vmax[metric_order.index(metric)])

    plt.tight_layout()
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{model_config}_lulc_metrics.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{model_config}_lulc_metrics.png'), bbox_inches='tight')
    else:
        plt.show()



def cloud_mask_threshold(cloud_prob_map, threshold=30):
    """Create a binary cloud mask based on the cloud probability map and a threshold."""
    cloud_mask = cloud_prob_map > threshold
    return cloud_mask


def load_image_center_crop(image, channels=None, center_crop=False, size=(256, 256)):
    """Load an image and optionally apply a center crop. Image shape: [C, H, W]."""
    if isinstance(image, str):
        with rs.open(image) as src:
            if channels is not None:
                data = src.read(channels)
            else:
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

def normalize(array, clip=True, min_value=0, max_value=None, min_percentile=1, max_percentile=99, percentile_norm=False):
    '''
    normalize: normalize a numpy array so all value are between 0 and 1
    '''
    if clip:
        array = np.clip(array, min_value, max_value)
    if percentile_norm:
        array_min, array_max = np.nanpercentile(array, (min_percentile, max_percentile))
    else:
        array_min, array_max = min_value, max_value
    try:
        normalized_array = (array - array_min) / (array_max - array_min)
    except Exception as e:
        normalized_array = array
    return np.clip(normalized_array, 0, 1)

def visualize_one_image(
    msi=None,
    msi_channels=(3, 2, 1),
    sar=None,
    metadata=None,
    cld_shdw=None,
    cloud_channel=2,
    shadow_channel=5,
    cloud_color="red",
    shadow_color="blue",
    lulc=None,
    lulc_channel=0,
    lulc_color=None,
    default_opacity=0.5,
    save_dir=None,
    dpi=100,
    center_crop=True,
    center_crop_shape=(256, 256),
):
    """
    Visualize a satellite image (multi-spectral or SAR) with optional overlays and center crop.

    Parameters:
    - msi: np.ndarray or str (default=None)
        Multi-Spectral Image data or file path.
    - msi_channels: tuple (default=(3, 2, 1))
        Channels to use for RGB visualization from the multi-spectral image.
    - sar: np.ndarray or str (default=None)
        Synthetic Aperture Radar image data or file path.
    - metadata: dict (default=None)
        Dictionary containing metadata for the image (e.g., ROI, latitude, longitude, date, satellite).
    - cloud: np.ndarray or str (default=None)
        Cloud mask data or file path.
    - cloud_channel: int (default=None)
        Specific channel to use from the cloud mask data.
    - cloud_color: str (default="red")
        Color to visualize the cloud mask.
    - shadow: np.ndarray or str (default=None)
        Shadow mask data or file path.
    - shadow_channel: int (default=None)
        Specific channel to use from the shadow mask data.
    - shadow_color: str (default="blue")
        Color to visualize the shadow mask.
    - lulc: np.ndarray or str (default=None)
        Land Use/Land Cover classification data or file path.
    - lulc_channel: int (default=0)
        Specific channel to use from the LULC data.
    - lulc_color: list or ListedColormap (default=None)
        Colors or colormap for visualizing LULC classes.
    - default_opacity: float (default=0.5)
        Default opacity for overlay masks.
    - save_dir: str (default=None)
        Directory to save the output image.
    - dpi: int (default=100)
        Dots per inch (DPI) for the output image.
    - center_crop: bool (default=True)
        Whether to apply center cropping to the image.
    - center_crop_shape: tuple (default=(256, 256))
        Shape of the center crop (height, width).

    Returns:
    None
        The function displays the image with optional overlays and saves it if a directory is specified.

    Description:
    This function visualizes a satellite image (either multi-spectral or SAR) with optional overlays for clouds, shadows,
    and land use/land cover (LULC) classifications. It allows for normalization of image data and can display metadata
    such as region of interest (ROI), latitude, longitude, date, and satellite information. The function also supports
    center cropping to focus on a specific part of the image and saves the visualization if a save directory is provided.
    """

    plt.figure(figsize=(12, 12), dpi=dpi)

    if msi is not None:
        msi_data = load_image_center_crop(msi, center_crop=center_crop, size=center_crop_shape)
        msi_data = normalize(msi_data, max_value=3000, min_percentile=1, max_percentile=99)
        plt.imshow(msi_data[msi_channels, ...].transpose(1, 2, 0), interpolation="nearest", vmin=0, vmax=1)
    elif sar is not None:
        sar_data = load_image_center_crop(sar, center_crop=center_crop, size=center_crop_shape)
        sar_rgb = np.zeros((3, *sar_data.shape[1:]))
        sar_rgb[0, ...] = normalize(sar_data[0, ...], clip=False)  # VV
        sar_rgb[1, ...] = normalize(sar_data[1, ...], clip=False)  # VH
        sar_rgb[2, ...] = normalize(sar_data[1, ...] - sar_data[0, ...], clip=False)  # VH - VV
        plt.imshow(sar_rgb.transpose(1, 2, 0), interpolation="nearest", vmin=0, vmax=1)

    if metadata:
        plt.title(f'ROI (Lat, Long): {metadata["roi"]} ({metadata["latitude"]}, {metadata["longitude"]}), Date: {metadata["date_y_m"]}, Satellite: {metadata["satellite"]}')
    ax = plt.gca()
    ax.grid(which="major", visible=False)
    if not center_crop:
        center_x = data.shape[1] // 2
        center_y = data.shape[2] // 2
        rect = plt.Rectangle((center_x - center_crop_shape[0] // 2, center_y - center_crop_shape[1] // 2),
                             center_crop_shape[0], center_crop_shape[1],
                             linewidth=3, edgecolor='w', facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
    plt.xlabel("West-East")
    plt.ylabel("North-South")

    overlays = []

    # Load cloud data if specified
    if cld_shdw is not None:
        if isinstance(cld_shdw, str):
            cld_shdw_data = load_image_center_crop(cld_shdw, channels=[cloud_channel, shadow_channel], center_crop=center_crop, size=center_crop_shape)
        else:
            cld_shdw_data = cld_shdw
        overlays.append((cld_shdw_data, [cloud_color, shadow_color]))

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
        elif isinstance(color, list):
            for cls, cls_color in enumerate(color):
                overlay_image = np.zeros((*overlay_data[cls].shape, 4))
                color_rgba = matplotlib.colors.to_rgba(cls_color)
                overlay_image[..., :3] = color_rgba[:3]
                overlay_image[..., 3] = overlay_data[cls] * default_opacity
                plt.imshow(overlay_image)

    if save_dir:
        save_path = os.path.join(save_dir, f'roi{metadata["ROI ID"]}_{metadata["Date"]}_{metadata["Satellite"]}_visualization.png')
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


def preprocess(img, sensor, min_value, max_value):
    """
    Reshape img from (C, H, W) to (H, W, C) and scale the image, then return as numpy array.
    """
    img = img.permute(1, 2, 0).numpy()
    if sensor == "s2_toa":
        img = img[:, :, [3, 2, 1]]
        img = normalize(img, min_value=min_value, max_value=max_value, percentile_norm=False)
    elif sensor == "s1":
        sar = np.zeros((img.shape[0], img.shape[1], 3))
        sar[...,0] = normalize(img[...,0], clip=False, min_percentile=0, max_percentile=100, percentile_norm=True)  # VV
        sar[...,1] = normalize(img[...,1], clip=False, min_percentile=0, max_percentile=100, percentile_norm=True)   # VH
        sar[...,2] = normalize(img[...,1] - img[...,0], clip=False, min_percentile=0, max_percentile=100, percentile_norm=True)  # VV - VH
        img = sar
    elif sensor == "loss_mask":
        img = img
    return img



def visualize_batch(data, min_value, max_value, show_fig=False, save_fig=True, args=None):
    """
    Format:
    columns: timestamps
    rows: loss mask, targets, outputs, inputs (sensor1, sensor2, ...)

    sensors: list of sensors in the order of the channels (e.g., ["s2_toa", "s1", "landsat8", "landsat9"])
    Required data shape:
    - inputs: (B, T, C, H, W)
    - outputs: (B, T, C, H, W)
    - targets: (B, T, C, H, W)
    - loss_masks: (B, T, 1, H, W)
    - timestamps: (B, T)
    - geolocations: (B, 2)
    - roi_ids: (B)
    """
    sensors = data["sensors"]
    inputs = data["inputs"].cpu()
    outputs = data["outputs"].cpu()
    outputs_real = data["outputs_real"].cpu()
    targets = data["targets"].cpu()
    loss_masks = data["loss_masks"].cpu()
    timestamps = data["timestamps"].cpu()
    geolocations = data["geolocations"].cpu()
    roi_ids = data["rois"]

    channels = {
        "s2_toa": list(range(13)),
        "s1": list(range(2)),
        "landsat8": list(range(11)),
        "landsat9": list(range(11)),
    }

    for bid in range(inputs.size(0)):

        nrows = 4 + len(sensors)
        ncols = timestamps.shape[1]
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        fig.suptitle(f"ROI: {roi_ids[bid]}  Geolocation: ({geolocations[bid, 0].item():.3f}, {geolocations[bid, 1].item():.3f})", size=14)
        for fid, timestamp in enumerate(timestamps[bid]):
            axs[0, fid].set_title(f"{datetime.fromtimestamp(timestamps[bid, fid].item()).strftime('%Y-%m-%d')}", size=14)
            loss_mask = preprocess(loss_masks[bid, fid, 0].unsqueeze(0), "loss_mask", min_value, max_value)
            axs[0, fid].imshow(loss_mask, cmap="gray")
            if fid == 0: axs[0, fid].set_ylabel(f"Loss Masks", size=14)
            target = preprocess(targets[bid, fid, channels[sensors[0]]], sensors[0], min_value, max_value)
            axs[1, fid].imshow(target)
            if fid == 0: axs[1, fid].set_ylabel(f"Targets ({sensors[0]})", size=14)
            output_real = preprocess(outputs_real[bid, fid, channels[sensors[0]]], sensors[0], min_value, max_value)
            axs[2, fid].imshow(output_real)
            if fid == 0: axs[2, fid].set_ylabel(f"OutputsR({sensors[0]})", size=14)
            output = preprocess(outputs[bid, fid, channels[sensors[0]]], sensors[0], min_value, max_value)
            axs[3, fid].imshow(output)
            if fid == 0: axs[3, fid].set_ylabel(f"Outputs ({sensors[0]})", size=14)
            start_channel = 0
            for sid, sensor in enumerate(sensors):
                sensor_channels = [channel + start_channel for channel in channels[sensor]]
                input = preprocess(inputs[bid, fid, sensor_channels], sensor, min_value, max_value)
                axs[sid + 4, fid].imshow(input)
                if fid == 0: axs[sid + 4, fid].set_ylabel(f"Inputs ({sensor})", size=14)
                start_channel += len(channels[sensor])

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        if show_fig:
            plt.show()
        if save_fig:
            os.makedirs(os.path.join(args.output_dir, args.runname, "vis"), exist_ok=True)
            plt.savefig(
                os.path.join(args.output_dir, args.runname, "vis", f"EP{str(args.epoch)}_S{str(args.global_step)}_B{str(bid)}_Vmin{str(min_value)}_Vmax{str(max_value)}.png"))
        plt.close()

def benchmark_visualization(inputs, args, metrics=None):
    """ Visualize the input, target, and output images for benchmarking.
    Args:
        inputs: Dictionary containing the following keys:
            - input_images: Input images (B, T, C, H, W)
            - target: Target images (B, C, H, W)
            - output: Output images (B, C, H, W)

    Returns:
    """

    inputx = inputs["input_images"].cpu()
    target = inputs["target"].cpu()
    output = inputs["output"].cpu()

    input_vis_bands = [3,2,1]
    target_vis_bands = [3,2,1]

    if args.model_name == "uncrtaints" and args.exp_name.lower() == "diagonal_1":
        input_vis_bands = [5,4,3]
        target_vis_bands = [3,2,1]
    else:
        raise ValueError("Invalid dataset type or model name")
    
    for batch_id in range(inputx.size(0)):

        data_id = args.eval_iter*args.batch_size + batch_id
    
        for value_multiplier in [2, 5]:
            
            fig, axes = plt.subplots(1,5, figsize=(10,3), dpi=200)

            try:
                if metrics is not None:
                    fig.suptitle(f"""ROI: {inputs["roi"][batch_id]} |  Geolocation: ({inputs["latlong"][0][batch_id].item():.3f}, {inputs["latlong"][1][batch_id].item():.3f}) | PSNR: {metrics.psnrs[batch_id].item():.2f} | SSIM: {metrics.ssims[batch_id].item():.2f} | SAM: {metrics.sams[batch_id].item():.2f} | MAE: {metrics.maes[batch_id].item():.2f}""",
                        size=12, y=.99)
                else:                  
                    fig.suptitle(f"""ROI: {inputs["roi"][batch_id]} |  Geolocation: ({inputs["latlong"][0][batch_id].item():.3f}, {inputs["latlong"][1][batch_id].item():.3f})""", 
                        size=12, y=.99)
            except:
                if metrics is not None:
                    fig.suptitle(f"""Dataset: AllClear | Experiment: {args.exp_name} \n Data id: {data_id} | PSNR: {metrics.psnrs[batch_id].item():.2f} | SSIM: {metrics.ssims[batch_id].item():.2f} | SAM: {metrics.sams[batch_id].item():.2f} | MAE: {metrics.maes[batch_id].item():.2f}""", size=12, y=.99)
                else:
                    fig.suptitle(f"""Dataset: AllClear | Experiment: {args.exp_name} \n Data id: {data_id}""", size=12, y=.99)
                pass 
        
            for frame_id in range(0,3):
                
                ax = axes[frame_id]
                x = inputx[batch_id][frame_id][input_vis_bands]
                x = np.transpose(x, (1,2,0))
                x = x * value_multiplier
                x = np.clip(x, 0, 1)
                try:
                    ax.set_title("Input \n" + datetime.fromtimestamp(inputs["timestamps"][batch_id, frame_id].item()).strftime('%Y-%m-%d'))
                except:
                    ax.set_title(f"Input t={frame_id+1}")
                ax.imshow(x)
            
            ax = axes[3]
            x = output[batch_id][0][target_vis_bands]
            x = np.transpose(x, (1,2,0))
            x = x * value_multiplier
            x = np.clip(x, 0, 1)
            ax.imshow(x)
            ax.set_title("Prediction")
            
            ax = axes[4]
            x = target[batch_id][0][target_vis_bands]
            x = np.transpose(x, (1,2,0))
            x = x * value_multiplier
            x = np.clip(x, 0, 1)
            ax.imshow(x)
            try:
                ax.set_title("Target \n" + datetime.fromtimestamp(inputs["target_timestamps"][frame_id].item()).strftime('%Y-%m-%d'))
            except:
                ax.set_title("Target")
        
            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])
        
            fig.tight_layout()
            # plt.pause(0.1)
    
            if args.model_name.lower() == "uncrtaints":
                fpath = f"""/share/hariharan/cloud_removal/results/visualization-unc-AllClear/"""
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                plt.savefig(f"""/share/hariharan/cloud_removal/results/visualization-unc-AllClear/vm{value_multiplier}__{data_id}__{args.model_name}_[{args.exp_name}].png""")
            elif args.model_name.lower() == "pmaa":
                if not os.path.exists(f"/share/hariharan/cloud_removal/results/visualization-pmaa"):
                    os.makedirs(f"/share/hariharan/cloud_removal/results/visualization-pmaa")
                plt.savefig(f"""/share/hariharan/cloud_removal/results/visualization-pmaa/vm{value_multiplier}__{data_id}__{args.model_name}_[{args.exp_name}].png""")
            elif args.model_name.lower() == "dae":
                if not os.path.exists(f"/share/hariharan/cloud_removal/results/visualization-dae"):
                    os.makedirs(f"/share/hariharan/cloud_removal/results/visualization-dae")
                plt.savefig(f"""/share/hariharan/cloud_removal/results/visualization-dae/vm{value_multiplier}__{data_id}__{args.model_name}_[{args.exp_name}].png""")
            # plt.pause(0.1)
            plt.close()