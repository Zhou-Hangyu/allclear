import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import sys, os, json
import pandas as pd

if "ck696" in os.getcwd():
    sys.path.append("/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model")
    sys.path.append("/share/hariharan/ck696/allclear/baselines")
    sys.path.append("/share/hariharan/ck696/allclear")
else:
    sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/model/")
    sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/")

from allclear.utils import plot_lulc_metrics

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # avoid running out of shared memory handles (https://github.com/pytorch/pytorch/issues/11201)

from allclear import CRDataset
from allclear import UnCRtainTS, LeastCloudy, Mosaicing, DAE, CTGAN, UTILISE, PMAA, DiffCR

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Metrics:
    """
    Required shapes:
    - outputs: (B, T, C, H, W)
    - targets: (B, 1, C, H, W)
    - masks: (B, 1, 1, H, W)
    """
    def __init__(self, outputs, targets, masks, lulc_maps=None, device=torch.device("cuda"), batch_size=32):
        self.device = torch.device(device)
        self.outputs = outputs.reshape(-1, *outputs.shape[2:]) # (B, T, C, H, W) -> (B*T, C, H, W)
        self.targets = targets.reshape(-1, *targets.shape[2:])
        self.masks = masks.reshape(-1, *masks.shape[2:]).repeat(1, targets.shape[2], 1, 1) # broadcast masks to all time steps
        if lulc_maps is not None:
            self.lulc_maps = lulc_maps.reshape(-1, *lulc_maps.shape[2:])
            self.masked_lulc_maps = (self.lulc_maps + 1) * self.masks - 1 # lulc maps are 0-indexed
        self.batch_size = batch_size

        # print(f"Evaluating Masked Metrics using {self.device}...")
        self.maes, self.rmses, self.psnrs, self.sams, self.ssims = self.batch_process(self.masks)

    def evaluate_aggregate(self):
        return {
            "MAE": self.maes.nanmean().item(),
            "RMSE": self.rmses.nanmean().item(),
            "PSNR": self.psnrs.nanmean().item(),
            "SAM": self.sams.nanmean().item(),
            "SSIM": self.ssims.nanmean().item(),
        }

    def evaluate_strat_lulc(self, mode="map"):
        """mode can be 'map'. 'label' mode is deprecated."""
        lulc_metrics = {}
        num_classes = 9
        for c in range(num_classes):
            if mode == "map":
                mask = self.masked_lulc_maps == c
                metrics = self.batch_process(mask)
                lulc_metrics[c] = {
                    "MAE": metrics[0].nanmean().item(),
                    "RMSE": metrics[1].nanmean().item(),
                    "PSNR": metrics[2].nanmean().item(),
                    "SAM": metrics[3].nanmean().item(),
                    "SSIM": metrics[4].nanmean().item(),
                }
            else:
                raise ValueError(f"Invalid mode: {mode}")
        return lulc_metrics


    def batch_process(self, masks):
        num_batches = (self.outputs.shape[0] + self.batch_size - 1) // self.batch_size
        maes, rmses, psnrs, sams, ssims = [], [], [], [], []


        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.outputs.shape[0])
            batch_outputs = self.outputs[start:end].to(self.device)
            batch_targets = self.targets[start:end].to(self.device)
            batch_masks = masks[start:end].to(self.device)

            assert batch_outputs.shape == batch_targets.shape, f"Output Shape: {batch_outputs.shape}, Target Shape: {batch_targets.shape}"
            if batch_outputs.shape != batch_masks.shape:
                print(f"The shape of the output ({batch_outputs.shape}) and mask ({batch_masks.shape}) is different")
                print("Select the first channel of the mask as the mask for the output")
                assert batch_outputs.size(0) == batch_masks.size(0)
                assert batch_outputs.size(2) == batch_masks.size(2)
                assert batch_outputs.size(3) == batch_masks.size(3)
                batch_masks = batch_masks[:,0:1]

            maes.append(self.mae(batch_outputs, batch_targets, batch_masks))
            rmses.append(self.rmse(batch_outputs, batch_targets, batch_masks))
            psnrs.append(self.psnr(batch_outputs, batch_targets, batch_masks))
            sams.append(self.sam(batch_outputs, batch_targets, batch_masks))
            ssims.append(self.ssim(batch_outputs, batch_targets, batch_masks))

        maes = torch.cat(maes, dim=0)
        rmses = torch.cat(rmses, dim=0)
        psnrs = torch.cat(psnrs, dim=0)
        sams = torch.cat(sams, dim=0)
        ssims = torch.cat(ssims, dim=0)

        return maes, rmses, psnrs, sams, ssims

    def mae(self, outputs, targets, masks):
        if masks is None:
            raise ValueError("Mask is required for MAE calculation")

        output_masked = outputs * masks
        target_masked = targets * masks
        mae_masked = F.l1_loss(output_masked, target_masked, reduction='none') * masks
        mae_masked_sum = mae_masked.sum(dim=[-3, -2, -1])
        mask_sum = masks.sum(dim=[-3, -2, -1])
        maes = mae_masked_sum / mask_sum
        return maes

    def rmse(self, outputs, targets, masks):
        if masks is None:
            raise ValueError("Mask is required for MAE calculation")
        output_masked = outputs * masks
        target_masked = targets * masks
        mse_masked = F.mse_loss(output_masked, target_masked, reduction='none') * masks
        mse_masked_sum = mse_masked.sum(dim=[-3, -2, -1])
        mask_sum = masks.sum(dim=[-3, -2, -1])
        mse_masked_mean = mse_masked_sum / mask_sum
        rmses = torch.sqrt(mse_masked_mean)
        return rmses


    def psnr(self, outputs, targets, masks, max_pixel=1.0):
        if masks is None:
            raise ValueError("Mask is required for MAE calculation")
        output_masked = outputs * masks
        target_masked = targets * masks
        mse_masked = F.mse_loss(output_masked, target_masked, reduction='none') * masks
        mse_masked_sum = mse_masked.sum(dim=[-3, -2, -1])
        mask_sum = masks.sum(dim=[-3, -2, -1])
        mse_masked_mean = mse_masked_sum / mask_sum
        psnrs = 20 * torch.log10(max_pixel / torch.sqrt(mse_masked_mean))
        return psnrs

    def sam(self, outputs, targets, masks):
        if masks is None:
            raise ValueError("Mask is required for SAM calculation")
        norm_out = F.normalize(outputs, p=2, dim=2)
        norm_tar = F.normalize(targets, p=2, dim=2)
        dot_product = (norm_out * norm_tar).sum(2).clamp(-1, 1)
        angles = torch.rad2deg(torch.acos(dot_product))
        angles_masked = angles.unsqueeze(2) * masks
        angles_sum = angles_masked.sum(dim=[-3, -2, -1])
        mask_sum = masks.sum(dim=[-3, -2, -1])
        sams = angles_sum / mask_sum
        return sams

    def ssim(self, outputs, targets, masks, window_size=11, k1=0.01, k2=0.03, C1=None, C2=None):
        if masks is None:
            raise ValueError("Mask is required for SSIM calculation")

        # Default C1 and C2 values
        L = 1  # L: dynamic range of the pixel-values (1 for normalized images)
        if C1 is None:
            C1 = (k1 * L) ** 2
        if C2 is None:
            C2 = (k2 * L) ** 2

        def gaussian_window(size, sigma):
            coords = torch.arange(size, dtype=torch.float32, device=self.device)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.view(1, 1, -1) * g.view(1, -1, 1)



        # Create a gaussian kernel, applied to each channel separately
        window = gaussian_window(window_size, 1.5).repeat(outputs.shape[1], 1, 1, 1)

        masks = masks > 0.5

        # Compute SSIM over masked regions
        mu_x = F.conv2d(outputs * masks, window, padding=window_size // 2, groups=outputs.shape[1])
        mu_y = F.conv2d(targets * masks, window, padding=window_size // 2, groups=targets.shape[1])
        sigma_x = F.conv2d(outputs ** 2 * masks, window, padding=window_size // 2,
                           groups=outputs.shape[1])
        sigma_y = F.conv2d(targets ** 2 * masks, window, padding=window_size // 2,
                           groups=targets.shape[1])
        sigma_xy = F.conv2d(outputs * targets * masks, window, padding=window_size // 2,
                            groups=outputs.shape[1])

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x -= mu_x_sq
        sigma_y -= mu_y_sq
        sigma_xy -= mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)

        ssim_map = numerator / denominator
        ssim_map_masked = torch.where(masks, ssim_map, torch.tensor(float('nan'), device=self.device))
        ssims = torch.nanmean(ssim_map_masked, dim=[-3, -2, -1])
        return ssims


class BenchmarkEngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.data_loader = self.setup_data_loader()
        self.model = self.setup_model()

    def setup_model(self):
        if self.args.model_name == "uncrtaints":
            model = UnCRtainTS(self.args)
        elif self.args.model_name == "leastcloudy":
            model = LeastCloudy(self.args)
        elif self.args.model_name == "mosaicing":
            model = Mosaicing(self.args)
        elif self.args.model_name == "dae":
            model = DAE(self.args)
        elif self.args.model_name == "ctgan":
            model = CTGAN(self.args)
        elif self.args.model_name == "utilise":
            model = UTILISE(self.args)
        elif self.args.model_name == "pmaa":
            model = PMAA(self.args)
        elif self.args.model_name == "diffcr":
            model = DiffCR(self.args)
        else:
            raise ValueError(f"Invalid model name: {self.args.model_name}")
        return model

    def setup_data_loader(self):

        if self.args.dataset_type == "AllClear":
            with open(self.args.dataset_fpath, "r") as f:
                dataset = json.load(f)
            with open(self.args.cld_shdw_fpaths, "r") as f:
                cld_shdw_fpaths = json.load(f)
            print(f"Selected ROIs: {self.args.selected_rois}")
            selected_rois = self.args.selected_rois if (self.args.selected_rois is not None) and ("all" not in self.args.selected_rois)  else "all"
            
            if self.args.unique_roi == 1:
                # selected_rois = "unique"
                print(f"Number of total ROIs: {len(dataset)}")
                unique_dataset = {}
                unique_indices = []
                count = 0
                prev_id = "0"
                for ID, info in dataset.items():
                    roi_id = info["roi"][0]
                    if prev_id == roi_id or roi_id in unique_indices: 
                        unique_indices.append(roi_id)
                    else:
                        unique_dataset[str(count)] = dataset[ID]
                        count += 1
                        prev_id = roi_id
                dataset = unique_dataset
                # dataset = {str(i): self.dataset[ID] for i, ID in enumerate(dataset.keys())} # reindex the dataset
                print(f"Number of unique ROIs: {len(dataset)}")
            
            dataset = CRDataset(
                dataset=dataset,
                selected_rois=selected_rois,
                main_sensor=self.args.main_sensor,
                aux_sensors=self.args.aux_sensors,
                aux_data=self.args.aux_data,
                tx=self.args.tx,
                target_mode=self.args.target_mode,
                cld_shdw_fpaths=cld_shdw_fpaths,
                do_preprocess=self.args.do_preprocess,  # NOTE: Set this to False for all baselines
            )
            return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        
        elif self.args.dataset_type == "SEN12MS-CR-TS":
            if "ck696" in os.getcwd():
                sys.path.append("/share/hariharan/ck696/allclear/baselines/UnCRtainTS/data")
            else:
                sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/data/")
            from SEN12MSCRTS import SEN12MSCRTS
            dt_test = SEN12MSCRTS(split='test', region='all', 
                                  sample_type="cloudy_cloudfree", 
                                    n_input_samples=self.args.tx, 
                                    rescale_method=self.args.sen12mscrts_rescale_method,
                                    )
            
            return DataLoader(dt_test, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        
        elif self.args.dataset_type == "CTGAN":

            if "ck696" in os.getcwd():
                sys.path.append("/share/hariharan/ck696/allclear/baselines/PMAA")
            else:
                sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/PMAA")

            from dataset_new import Sen2_MTC
            
            class CTGAN_OPT:
                def __init__(self):
                    self.root = "/share/hariharan/ck696/allclear/baselines/PMAA/data"
                    self.test_mode = "test"
                
                def __iter__(self):
                    return iter(self.__dict__.items())
                
                def __getitem__(self, key):
                    return self.__dict__[key]

            opt = CTGAN_OPT()
            test_data = Sen2_MTC(opt, mode="test")
            test_loader = DataLoader(test_data, 
                                     batch_size=self.args.batch_size,
                                     shuffle=False, 
                                     num_workers=self.args.num_workers,
                                     drop_last=False)

            return test_loader
        
    def convert_data_format_from_sen12mscrts(self, batch):
        # print(batch.keys())

        s1 = torch.stack(batch["input"]["S1"], dim=1)
        s2 = torch.stack(batch["input"]["S2"], dim=1)

        input_images = torch.cat([s1, s2], dim=2)
        input_cld_shdw = torch.stack(batch["input"]["masks"], dim=1).unsqueeze(1).repeat(1,2,1,1,1)

        s1 = torch.stack(batch["target"]["S1"], dim=1)
        s2 = torch.stack(batch["target"]["S2"], dim=1)

        target = torch.cat([s1, s2], dim=2)
        target_cld_shdw = torch.stack(batch["target"]["masks"], dim=1).unsqueeze(1).repeat(1,2,1,1,1)

        dates = torch.stack((torch.stack(batch['input']['S1 TD']),
                             torch.stack(batch['input']['S2 TD']))).float().mean(dim=0)
        
        if self.args.sen12mscrts_reset_dates == "zeros":
            dates = torch.zeros_like(dates)
        elif self.args.sen12mscrts_reset_dates == "min":
            dates = dates - dates.min()
        elif self.args.sen12mscrts_reset_dates == "none":
            dates = dates

        inputs = {
            "input_images": input_images.permute(0,2,1,3,4),
            "input_cld_shdw": input_cld_shdw,
            "target": target.permute(0,2,1,3,4),
            "target_cld_shdw": target_cld_shdw,
            "dw": None,
            "target_dw": None,
            "time_differences": dates
        }

        return inputs
    
    def convert_data_format_from_ctgan(self, batch):

        bands = [3,2,1,7]
        inputs = {}
        real_A, real_B, image_names = batch

        # real_A = [IMG1, IMG2, IMg3] 
        # IMG: bs x c x h x w
        input_images = torch.stack(real_A, dim=2) * 0.5 + 0.5
        # Input_images: bs x c x t x h x w
        target = real_B.unsqueeze(2) * 0.5 + 0.5

        bs, nc, ct, nw, nh = input_images.shape
        input_images_placeholder = torch.zeros((bs, 15, 3, 256, 256))
        target_placeholder = torch.zeros((bs, 15, 1, 256, 256))
        input_images_placeholder[:,bands,...] = input_images
        target_placeholder[:,bands,...] = target
        input_cld_shdw = torch.zeros((bs, 2, ct, nw, nh))
        target_cld_shdw = torch.zeros((bs, 2, 1, nw, nh))

        inputs = {
            "input_images": input_images_placeholder,
            "input_cld_shdw": input_cld_shdw,
            "target": target_placeholder,
            "target_cld_shdw": target_cld_shdw,
            "dw": None,
            "target_dw": None,
            "time_differences": None,
        }

        return inputs



    def run(self):
        print("Running Benchmark...")
        outputs_all = []
        targets_all = []
        target_non_cld_shdw_masks_all = []
        target_lulc_maps = []

        per_batch_metrics = {
            "MAE": [],
            "RMSE": [],
            "PSNR": [],
            "SAM": [],
            "SSIM": []
        }

        lulc_metrics_all = {i: {"MAE": [], "RMSE": [], "PSNR": [], "SAM": [], "SSIM": []} for i in range(9)}


        for data_id, data in tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc="Evaluating Batches"):

            if data_id == 100:
                break

            if self.args.dataset_type == "SEN12MS-CR-TS":
                data = self.convert_data_format_from_sen12mscrts(data)

            elif self.args.dataset_type == "CTGAN":
                data = self.convert_data_format_from_ctgan(data)

            with torch.no_grad():
                data = self.model.preprocess(data)
                targets_all.append(data["target"].cpu())
                target_cld_shdw_mask = (data["target_cld_shdw"][:,0,...] + data["target_cld_shdw"][:,1,...]) > 0
                target_non_cld_shdw_mask = torch.logical_not(target_cld_shdw_mask).cpu()
                target_non_cld_shdw_masks_all.append(target_non_cld_shdw_mask)
                if self.args.dataset_type == "AllClear": target_lulc_maps.append(data["target_dw"].cpu())
                outputs = self.model.forward(data)
                outputs_all.append(outputs["output"].cpu())

            if self.args.save_plots:
                for i in range(self.args.batch_size):
                    continue

            # Compute metrics for the current batch
            batch_outputs = torch.cat(outputs_all, dim=0)
            batch_targets = torch.cat(targets_all, dim=0)
            batch_masks = torch.cat(target_non_cld_shdw_masks_all, dim=0).unsqueeze(1)
            if self.args.dataset_type == "AllClear": 
                batch_lulc_maps = torch.cat(target_lulc_maps, dim=0)
                metrics = Metrics(batch_outputs, batch_targets, batch_masks, device=self.device, lulc_maps=batch_lulc_maps)
            else:
                metrics = Metrics(batch_outputs, batch_targets, batch_masks, device=self.device)
            batch_results = metrics.evaluate_aggregate()

            # Store per-batch metrics
            for key in per_batch_metrics:
                per_batch_metrics[key].append(batch_results[key])
            
            # Store stratified metrics for each LULC class
            if self.args.dataset_type == "AllClear": 
                strat_lulc_metrics = metrics.evaluate_strat_lulc(mode="map")
                for lulc_class, class_metrics in strat_lulc_metrics.items():
                    for metric, value in class_metrics.items():
                        lulc_metrics_all[lulc_class][metric].append(value)

            # Clear lists for the next batch
            outputs_all.clear()
            targets_all.clear()
            target_non_cld_shdw_masks_all.clear()
            target_lulc_maps.clear()

        # Compute final aggregate metrics
        final_results = {key: torch.tensor(values).nanmean().item() for key, values in per_batch_metrics.items()}
        print(final_results)

        print(f"experiment_output_path: {self.args.experiment_output_path}")

        # Compute final stratified metrics for each LULC class
        os.makedirs(f"{self.args.experiment_output_path}", exist_ok=True)
        os.makedirs(f"{self.args.experiment_output_path}/{self.args.dataset_type}", exist_ok=True)

        if self.args.dataset_type == "AllClear": 
            final_lulc_metrics = {
                lulc_class: {metric: torch.tensor(values).nanmean().item() for metric, values in class_metrics.items()}
                for lulc_class, class_metrics in lulc_metrics_all.items()
            }
            print(final_lulc_metrics)
            plot_lulc_metrics(final_lulc_metrics, save_dir=self.args.experiment_output_path, model_config=f"{self.args.model_name}_map")
        
            final_lulc_metrics = pd.DataFrame(final_lulc_metrics)
            final_lulc_metrics.to_csv(f"{self.args.experiment_output_path}/{self.args.model_name}_lulc_metrics.csv", index=True)

        final_results = pd.DataFrame(final_results, index=[0])
        final_results.to_csv(f"{self.args.experiment_output_path}/{self.args.dataset_type}/{self.args.model_name}_results.csv", index=False)

        self.cleanup()

        if self.args.dataset_type == "AllClear": 
            return final_results, final_lulc_metrics
        else:
            return final_results

    def cleanup(self):
        pass

    def save_results(self, output, target, timestamp, res_dir):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Engine for Models and Datasets")
    parser.add_argument("--baseline-base-path", type=str, help="Path to the baseline codebase")
    parser.add_argument("--dataset-fpath", type=str, default="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_test_4k_v1.json" ,required=True, help="Path to dataset metadata file")
    parser.add_argument("--model-name", type=str, required=True, help="Model to use for benchmarking")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for data loading")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--data-split", type=str, default="train", help="Data split to use [train, val, test]")
    parser.add_argument("--device", type=str, required=True, help="Device to run the model on")
    parser.add_argument("--main-sensor", type=str, default="s2_toa", help="Main sensor for the dataset")
    parser.add_argument("--aux-sensors", type=str, nargs="+", help="Auxiliary sensors for the dataset")
    parser.add_argument("--aux-data", type=str, nargs="+",default=["cld_shdw", "dw"], help="Auxiliary data for the dataset")
    parser.add_argument("--target-mode", type=str, default="s2p", choices=["s2p", "s2s"], help="Target mode for the dataset")
    parser.add_argument("--cld-shdw-fpaths", type=str, default="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json", help="Path to cloud shadow masks")
    parser.add_argument("--do-preprocess", action="store_true", help="Preprocess the data before running the model")
    parser.add_argument("--dataset-type", type=str, default="AllClear", 
                        choices=["AllClear", "CTGAN", "SEN12MS-CR", "SEN12MS-CR-TS"], help="Type of dataset")
    # parser.add_argument("--dataset-type", type=str, default="SEN12MS-CR", choices=["SEN12MS-CR", "SEN12MS-CR-TS"], help="Type of dataset")
    parser.add_argument("--input-t", type=int, default=3, help="Number of input time points (for time-series datasets)")
    parser.add_argument("--selected-rois", type=str, nargs="+", help="Selected ROIs for benchmarking")
    parser.add_argument("--tx", type=int, default=3, help="Number of images in a sample for the dataset")
    parser.add_argument("--experiment-output-path", type=str, default="/share/hariharan/cloud_removal/results/baselines", help="Path to save the experiment results")
    parser.add_argument("--save-plots", action="store_true", help="Save plots for the experiment")
    parser.add_argument("--eval-bands", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], help="Evaluation bands for the dataset")
    parser.add_argument("--unique-roi", type=int, default=0, help="0 uses all metadata, 1 uses only unique ROI")
    parser.add_argument("--draw-vis", type=int, default=0, help="0 dont draw, 1 draw results")
    
    uc_args = parser.add_argument_group("UnCRtainTS Arguments")
    uc_args.add_argument("--uc-exp-name", type=str, default="noSAR_1", help="Experiment name for UnCRtainTS")
    uc_args.add_argument("--uc-root1", type=str, default="/share/hariharan/cloud_removal/SEN12MSCRTS", help="Root 1 for UnCRtainTS")
    uc_args.add_argument("--uc-root2", type=str, default="/share/hariharan/cloud_removal/SEN12MSCRTS", help="Root 2 for UnCRtainTS")
    uc_args.add_argument("--uc-root3", type=str, default="/share/hariharan/cloud_removal/SEN12MSCR", help="Root 3 for UnCRtainTS")
    uc_args.add_argument("--uc-weight-folder", type=str, default="/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/results", help="Folder containing weights for UnCRtainTS")
    uc_args.add_argument("--uc-s1", type=int, default=0, help="0: No Sentinel-1, 1: Include Sentinel-1")

    dae_args = parser.add_argument_group("DAE Arguments")
    dae_args.add_argument("--dae-image-size", type=int, default=256, help="Image size for DAE")
    dae_args.add_argument("--dae-in-channel", type=int, default=12, help="Input channels for DAE")
    dae_args.add_argument("--dae-out-channel", type=int, default=13, help="Output channels for DAE")
    dae_args.add_argument("--dae-LPB", type=int, default=1, help="Layer per block")
    dae_args.add_argument("--dae-max-d0", type=int, default=128, help="The maximum dimension")
    dae_args.add_argument("--dae-max-dim", type=int, default=512, help="Max dimension for DAE")
    dae_args.add_argument("--dae-model-blocks", type=str, default="CCCCAA", help="Model blocks for DAE")
    dae_args.add_argument("--dae-norm-num-groups", type=int, default=4, help="Number of groups for normalization in DAE")
    dae_args.add_argument("--dae-checkpoint", type=str, default="/share/hariharan/cloud_removal/allclear/experimental_scripts/results/ours/dae/'model_test-run-3-loss2*10_0_9800.pt'", help="Checkpoint for DAE")

    dae_args = parser.add_argument_group("PMAA Arguments")
    dae_args.add_argument("--pmaa-checkpoint", type=str, default='/share/hariharan/ck696/allclear/baselines/PMAA/pretrained/pmaa_new.pth', help="Specified PMAA trained on Sen12_MTC_new or Sen12_MTC_old")

    dae_args = parser.add_argument_group("DiffCR Arguments")
    dae_args.add_argument("--diff-checkpoint", type=str, default="/share/hariharan/ck696/allclear/baselines/DiffCR/pretrained/diffcr_new.pth", help="Specified PMAA trained on Sen12_MTC_new or Sen12_MTC_old")

    ctgan_args = parser.add_argument_group("CTGAN Arguments")
    ctgan_args.add_argument("--ctgan-gen-checkpoint", type=str, default="/share/hariharan/ck696/allclear/baselines/CTGAN/Pretrain/G_epoch97_PSNR21.259-002.pth", help="Generator checkpoint for CTGAN")

    util_args = parser.add_argument_group("UTILISE Arguments")
    util_args.add_argument("--utilise-config", type=str, default="/share/hariharan/cloud_removal/allclear/baselines/U-TILISE/configs/demo_sen12mscrts.yaml", help="Config file for UTILISE")
    util_args.add_argument("--utilise-checkpoint", type=str, default="/share/hariharan/cloud_removal/allclear/baselines/U-TILISE/checkpoints/utilise_sen12mscrts_wo_s1.pth", help="Checkpoint for UTILISE")
    
    sen12mscrts_args = parser.add_argument_group("SEN12MSCRTS Arguments")
    sen12mscrts_args.add_argument("--sen12mscrts-rescale-method", type=str, default="default", choices=["default", "resnet", "allclear"], help="Rescale method for SEN12MSCRTS")
    sen12mscrts_args.add_argument("--sen12mscrts-reset-dates", type=str, default="none", choices=["none", "zeros", "min"], help="Reset dates for SEN12MSCRTS")

    ctgan_args = parser.add_argument_group("CTGAN Arguments")
    # sen12mscrts_args.add_argument("--sen12mscrts-rescale-method", type=str, default="default", choices=["default", "resnet", "allclear"], help="Rescale method for SEN12MSCRTS")
    # sen12mscrts_args.add_argument("--sen12mscrts-reset-dates", type=str, default="none", choices=["none", "zeros", "min"], help="Reset dates for SEN12MSCRTS")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    benchmark_args = parse_arguments()
    if benchmark_args.model_name == "uncrtaints":
        from baselines.UnCRtainTS.model.parse_args import create_parser
        uc_args = create_parser(mode="test").parse_args([
            '--experiment_name', benchmark_args.uc_exp_name,
            '--root1', benchmark_args.uc_root1,
            '--root2', benchmark_args.uc_root2,
            '--root3', benchmark_args.uc_root3,
            '--weight_folder', benchmark_args.uc_weight_folder])
        args = argparse.Namespace(**{**vars(uc_args), **vars(benchmark_args)})
    elif benchmark_args.model_name in ["leastcloudy", "mosaicing", "dae", "ctgan", "utilise", "pmaa", "diffcr"]:
        args = benchmark_args
    else:
        raise ValueError(f"Invalid model name: {benchmark_args.model_name}")
    print("Loading Benchmark Engine...")
    print(f"Unique ROI: {args.unique_roi}")
    engine = BenchmarkEngine(args)
    engine.run()
