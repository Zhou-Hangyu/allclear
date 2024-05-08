import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import sys, os
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
from allclear import UnCRtainTS, LeastCloudy, Mosaicing, Simple3DUnet, CTGAN, UTILISE, PMAA
from baselines.UnCRtainTS.model.parse_args import create_parser

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Metrics:
    def __init__(self, outputs, targets, masks, lulc=None, lulc_maps=None, device=torch.device("cuda"), batch_size=32):
        self.device = torch.device(device)
        self.outputs = outputs.view(-1, *outputs.shape[2:]) # (B, T, C, H, W) -> (B*T, C, H, W)
        self.targets = targets.view(-1, *targets.shape[2:])
        self.masks = masks.view(-1, *masks.shape[2:]).repeat(1, targets.shape[2], 1, 1) # broadcast masks to all time steps
        self.lulc = lulc if lulc is not None else None
        if lulc_maps is not None:
            self.lulc_maps = lulc_maps.view(-1, *lulc_maps.shape[2:])
            self.masked_lulc_maps = (self.lulc_maps + 1) * self.masks - 1 # lulc maps are 0-indexed
        self.batch_size = batch_size

        print(f"Evaluating Masked Metrics using {self.device}...")
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
        """mode can be 'map' or 'label'"""
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
            elif mode == "label":
                lulc_c = self.lulc == c
                lulc_metrics[c] = {
                    "MAE": self.maes[lulc_c].nanmean().item(),
                    "RMSE": self.rmses[lulc_c].nanmean().item(),
                    "PSNR": self.psnrs[lulc_c].nanmean().item(),
                    "SAM": self.sams[lulc_c].nanmean().item(),
                    "SSIM": self.ssims[lulc_c].nanmean().item(),
                }
            else:
                raise ValueError(f"Invalid mode: {mode}")
        return lulc_metrics


    def batch_process(self, masks):
        num_batches = (self.outputs.shape[0] + self.batch_size - 1) // self.batch_size
        maes, rmses, psnrs, sams, ssims = [], [], [], [], []

        # print("mask shape": masks.shape)
        # print(f"Mask shape: {masks.shape}")

        for i in tqdm(range(num_batches), desc="Processing batches"):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.outputs.shape[0])
            batch_outputs = self.outputs[start:end].to(self.device)
            batch_targets = self.targets[start:end].to(self.device)
            batch_masks = masks[start:end].to(self.device)

            assert batch_outputs.shape == batch_targets.shape, f"Output Shape: {batch_outputs.shape}, Target Shape: {batch_targets.shape}"
            if batch_outputs.shape != batch_masks.shape:
                print("The shape of the output and mask is different")
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
        elif self.args.model_name == "simpleunet":
            model = Simple3DUnet(self.args)
        elif self.args.model_name == "ctgan":
            model = CTGAN(self.args)
        elif self.args.model_name == "utilise":
            model = UTILISE(self.args)
        elif self.args.model_name == "pmaa":
            model = PMAA(self.args)
        else:
            raise ValueError(f"Invalid model name: {self.args.model_name}")
        return model

    def setup_data_loader(self):
        dataset = CRDataset(
            self.args.data_path, self.args.metadata_path, self.args.selected_rois, self.args.time_span, self.args.eval_mode, self.args.cloud_percentage_range
        )
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def run(self):
        outputs_all = []
        targets_all = []
        target_masks_all = []
        target_lulc_labels = []
        target_lulc_maps = []
        for data in tqdm(self.data_loader, desc="Running Benchmark"):
            with torch.no_grad():
                data = self.model.preprocess(data)
                targets_all.append(data["target"].cpu())
                target_mask = 1 - data["target_cloud_mask"].cpu() # negate to get non-cloud mask
                target_masks_all.append(target_mask)
                target_lulc_labels.append(data["target_lulc_label"].cpu())
                target_lulc_maps.append(data["target_lulc_map"].cpu())
                outputs = self.model.forward(data)
                outputs_all.append(outputs["output"].cpu())
                # save results
                # for i in range(self.args.batch_size):
                #     # save results in tif format
                #     self.save_results(outputs[i], targets[i], data["timestamps"][i], self.args.res_dir)
                # print(outputs[0].shape)
            if self.args.save_plots:
                for i in range(self.args.batch_size):
                    # save_rgb_side_by_side(outputs[0][i].squeeze(0), targets[i], data["timestamps"][i], self.args.experiment_output_path)
                    # print(outputs[0].squeeze(0).shape)
                    # save_batch_visualization(data, outputs[0].squeeze(1).detach().cpu(), self.args.experiment_output_path, data["timestamps"][i], i)
                    continue

        outputs = torch.cat(outputs_all, dim=0)
        targets = torch.cat(targets_all, dim=0)
        masks = torch.cat(target_masks_all, dim=0)
        lulc_labels = torch.cat(target_lulc_labels, dim=0)
        lulc_maps = torch.cat(target_lulc_maps, dim=0)

        metrics = Metrics(outputs, targets, masks, lulc=lulc_labels, lulc_maps=lulc_maps)
        results = metrics.evaluate_aggregate()
        print(results)

        strat_lulc_label = metrics.evaluate_strat_lulc(mode="label")
        strat_lulc = metrics.evaluate_strat_lulc(mode="map")
        plot_lulc_metrics(strat_lulc_label, save_dir=self.args.experiment_output_path, model_config=f"{self.args.model_name}_label")
        plot_lulc_metrics(strat_lulc, save_dir=self.args.experiment_output_path, model_config=f"{self.args.model_name}_map")

        # Remove B10 from the outputs and targets for evaluation
        # targets = targets[:, :, self.args.eval_bands, :, :]
        self.cleanup()
        return outputs, targets, masks, lulc_labels, lulc_maps

    def cleanup(self):
        pass

    def save_results(self, output, target, timestamp, res_dir):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Engine for Models and Datasets")
    parser.add_argument("--baseline-base-path", type=str, required=True, help="Path to the baseline codebase")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to metadata file")
    parser.add_argument("--model-name", type=str, required=True, help="Model to use for benchmarking")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for data loading")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--data-split", type=str, default="train", help="Data split to use [train, val, test]")
    parser.add_argument("--device", type=str, required=True, help="Device to run the model on")
    parser.add_argument("--dataset-type", type=str, default="SEN12MS-CR", choices=["SEN12MS-CR", "SEN12MS-CR-TS"], help="Type of dataset")
    parser.add_argument("--input-t", type=int, default=3, help="Number of input time points (for time-series datasets)")
    parser.add_argument("--selected-rois", type=int, nargs="+", required=True, help="Selected ROIs for benchmarking")
    parser.add_argument("--time-span", type=int, default=3, help="Time span for the dataset")
    parser.add_argument("--cloud-percentage-range", type=int, nargs=2, default=[20, 30], help="Cloud percentage range for the dataset")
    parser.add_argument("--experiment-output-path", type=str, default="/share/hariharan/cloud_removal/results/baselines", help="Path to save the experiment results")
    parser.add_argument("--save-plots", action="store_true", help="Save plots for the experiment")
    parser.add_argument("--eval-mode", type=str, default="toa", choices=["toa", "sr"], help="Evaluation mode for the dataset")
    parser.add_argument("--eval-bands", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], help="Evaluation bands for the dataset")

    uc_args = parser.add_argument_group("UnCRtainTS Arguments")
    uc_args.add_argument("--uc-exp-name", type=str, default="noSAR_1", help="Experiment name for UnCRtainTS")
    uc_args.add_argument("--uc-root1", type=str, default="/share/hariharan/cloud_removal/SEN12MSCRTS", help="Root 1 for UnCRtainTS")
    uc_args.add_argument("--uc-root2", type=str, default="/share/hariharan/cloud_removal/SEN12MSCRTS", help="Root 2 for UnCRtainTS")
    uc_args.add_argument("--uc-root3", type=str, default="/share/hariharan/cloud_removal/SEN12MSCR", help="Root 3 for UnCRtainTS")
    uc_args.add_argument("--uc-weight-folder", type=str, default="/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/results", help="Folder containing weights for UnCRtainTS")

    su_args = parser.add_argument_group("Simple3DUnet Arguments")
    su_args.add_argument("--su-image-size", type=int, default=256, help="Image size for Simple3DUnet")
    su_args.add_argument("--su-in-channel", type=int, default=12, help="Input channels for Simple3DUnet")
    su_args.add_argument("--su-out-channel", type=int, default=3, help="Output channels for Simple3DUnet")
    su_args.add_argument("--su-max-dim", type=int, default=512, help="Max dimension for Simple3DUnet")
    su_args.add_argument("--su-model-blocks", type=str, default="CRRAAA", help="Model blocks for Simple3DUnet")
    su_args.add_argument("--su-num-groups", type=int, default=4, help="Number of groups for normalization in Simple3DUnet")
    su_args.add_argument("--su-checkpoint", type=str, default="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_12.pt", help="Checkpoint for Simple3DUnet")

    su_args = parser.add_argument_group("PMAA Arguments")
    su_args.add_argument("--pmaa-model", type=str, default="new", help="Specified PMAA trained on Sen12_MTC_new or Sen12_MTC_old")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    benchmark_args = parse_arguments()
    if benchmark_args.model_name == "uncrtaints":
        uc_args = create_parser(mode="test").parse_args([
            '--experiment_name', benchmark_args.uc_exp_name,
            '--root1', benchmark_args.uc_root1,
            '--root2', benchmark_args.uc_root2,
            '--root3', benchmark_args.uc_root3,
            '--weight_folder', benchmark_args.uc_weight_folder])
        args = argparse.Namespace(**{**vars(uc_args), **vars(benchmark_args)})
    elif benchmark_args.model_name in ["leastcloudy", "mosaicing", "simpleunet", "ctgan", "utilise", "pmaa"]:
        args = benchmark_args
    else:
        raise ValueError(f"Invalid model name: {benchmark_args.model_name}")
    engine = BenchmarkEngine(args)
    engine.run()