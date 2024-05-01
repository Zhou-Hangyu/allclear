import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import sys


sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/model")
sys.path.append("/share/hariharan/cloud_removal/allclear")

from allclear import CRDataset
from allclear import UnCRtainTS, LeastCloudy, Mosaicing, Simple3DUnet
from baselines.UnCRtainTS.model.parse_args import create_parser

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")







class Metrics:
    def __init__(self, outputs, targets, masks, device=torch.device("cpu")):
        self.device = torch.device(device)
        self.outputs = outputs.to(self.device).view(-1, *outputs.shape[2:]) # (B, T, C, H, W) -> (B*T, C, H, W)
        self.targets = targets.to(self.device).view(-1, *targets.shape[2:])
        self.masks = masks.to(self.device).view(-1, *masks.shape[2:]).repeat(1, targets.shape[2], 1, 1) # broadcast masks to all time steps

        print(f"Evaluating Metrics using {self.device}...")
        self.maes = self.mae()
        self.rmses = self.rmse()
        self.psnrs = self.psnr()
        self.sams = self.sam()
        self.ssims = self.ssim()

    def mae(self):
        if self.masks is None:
            raise ValueError("Mask is required for MAE calculation")
        output_masked = self.outputs * self.masks
        target_masked = self.targets * self.masks
        mae_masked = F.l1_loss(output_masked, target_masked, reduction='none') * self.masks
        mae_masked_sum = mae_masked.sum(dim=[-3, -2, -1])
        mask_sum = self.masks.sum(dim=[-3, -2, -1])
        maes = mae_masked_sum / mask_sum
        return maes

    def rmse(self):
        if self.masks is None:
            raise ValueError("Mask is required for RMSE calculation")
        output_masked = self.outputs * self.masks
        target_masked = self.targets * self.masks
        mse_masked = F.mse_loss(output_masked, target_masked, reduction='none') * self.masks
        mse_masked_sum = mse_masked.sum(dim=[-3, -2, -1])
        mask_sum = self.masks.sum(dim=[-3, -2, -1])
        mse_masked_mean = mse_masked_sum / mask_sum
        rmses = torch.sqrt(mse_masked_mean)
        return rmses


    def psnr(self, max_pixel=1.0):
        if self.masks is None:
            raise ValueError("Mask is required for RMSE calculation")
        output_masked = self.outputs * self.masks
        target_masked = self.targets * self.masks
        mse_masked = F.mse_loss(output_masked, target_masked, reduction='none') * self.masks
        mse_masked_sum = mse_masked.sum(dim=[-3, -2, -1])
        mask_sum = self.masks.sum(dim=[-3, -2, -1])
        mse_masked_mean = mse_masked_sum / mask_sum
        psnrs = 20 * torch.log10(max_pixel / torch.sqrt(mse_masked_mean))
        return psnrs

    def sam(self):
        if self.masks is None:
            raise ValueError("Mask is required for SAM calculation")
        norm_out = F.normalize(self.outputs, p=2, dim=2)
        norm_tar = F.normalize(self.targets, p=2, dim=2)
        dot_product = (norm_out * norm_tar).sum(2).clamp(-1, 1)
        angles = torch.rad2deg(torch.acos(dot_product))
        angles_masked = angles.unsqueeze(2) * self.masks
        angles_sum = angles_masked.sum(dim=[-3, -2, -1])
        mask_sum = self.masks.sum(dim=[-3, -2, -1])
        sams = angles_sum / mask_sum
        return sams

    def ssim(self, window_size=11, k1=0.01, k2=0.03, C1=None, C2=None):
        if self.masks is None:
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
        window = gaussian_window(window_size, 1.5).repeat(self.outputs.shape[1], 1, 1, 1)

        masks = self.masks > 0.5

        # Compute SSIM over masked regions
        mu_x = F.conv2d(self.outputs * masks, window, padding=window_size // 2, groups=self.outputs.shape[1])
        mu_y = F.conv2d(self.targets * masks, window, padding=window_size // 2, groups=self.targets.shape[1])
        sigma_x = F.conv2d(self.outputs ** 2 * masks, window, padding=window_size // 2,
                           groups=self.outputs.shape[1])
        sigma_y = F.conv2d(self.targets ** 2 * masks, window, padding=window_size // 2,
                           groups=self.targets.shape[1])
        sigma_xy = F.conv2d(self.outputs * self.targets * masks, window, padding=window_size // 2,
                            groups=self.outputs.shape[1])

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


    def evaluate_aggregate(self):
        return {
            "MAE": self.maes.mean().item(),
            "RMSE": self.rmses.mean().item(),
            "PSNR": self.psnrs.mean().item(),
            "SAM": self.sams.mean().item(),
            "SSIM": self.ssims.mean().item(),
        }

    def evaluate_stratified(self):
        pass

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
        for data in tqdm(self.data_loader, desc="Running Benchmark"):
            with torch.no_grad():
                data = self.model.preprocess(data)
                targets_all.append(data["target"].cpu())
                target_mask = 1 - data["target_cloud_mask"].cpu() # negate to get non-cloud mask
                target_masks_all.append(target_mask)
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

        metrics = Metrics(outputs, targets, masks)
        results = metrics.evaluate_aggregate()
        print(results)

        # Remove B10 from the outputs and targets for evaluation
        # targets = targets[:, :, self.args.eval_bands, :, :]
        self.cleanup()
        return outputs, targets, masks

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
    parser.add_argument("--experiment-output-path", type=str, default="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init", help="Path to save the experiment results")
    parser.add_argument("--save-plots", action="store_true", help="Save plots for the experiment")
    parser.add_argument("--eval-mode", type=str, default="toa", choices=["toa", "sr"], help="Evaluation mode for the dataset")
    parser.add_argument("--eval-bands", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], help="Evaluation bands for the dataset")

    uc_args = parser.add_argument_group("UnCRtainTS Arguments")
    uc_args.add_argument("--uc-exp-name", type=str, default="noSAR_1", help="Experiment name for UnCRtainTS")
    uc_args.add_argument("--uc-root1", type=str, default="/share/hariharan/cloud_removal/SEN12MSCRTS", help="Root 1 for UnCRtainTS")
    uc_args.add_argument("--uc-root2", type=str, default="/share/hariharan/cloud_removal/SEN12MSCRTS", help="Root 2 for UnCRtainTS")
    uc_args.add_argument("--uc-root3", type=str, default="/share/hariharan/cloud_removal/SEN12MSCR", help="Root 3 for UnCRtainTS")
    uc_args.add_argument("--uc-weight-folder", type=str, default="/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/results", help="Folder containing weights for UnCRtainTS")

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
    elif benchmark_args.model_name in ["leastcloudy", "mosaicing", "simpleunet"]:
        args = benchmark_args
    else:
        raise ValueError(f"Invalid model name: {benchmark_args.model_name}")
    engine = BenchmarkEngine(args)
    engine.run()