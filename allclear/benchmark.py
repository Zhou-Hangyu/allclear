import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from tqdm import tqdm
import sys

# import lpips
# from pytorch_msssim import ssim, ms_ssim


sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/model")
sys.path.append("/share/hariharan/cloud_removal/allclear")

from allclear import CRDataset
from allclear import UnCRtainTS, LeastCloudy, Mosaicing, Simple3DUnet
from baselines.UnCRtainTS.model.parse_args import create_parser

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")







class Metrics:
    def __init__(self, args, outputs, targets, masks):
        self.device = torch.device(args.device)
        self.outputs = outputs.to(self.device)
        self.targets = targets.to(self.device)
        self.masks = masks.to(self.device)

    def mae(self):
        if self.masks is None:
            raise ValueError("Mask is required for MAE calculation")
        output_masked = self.outputs * self.masks
        target_masked = self.targets * self.masks
        mae_masked = F.l1_loss(output_masked, target_masked, reduction='none') * self.masks
        mae_masked_sum = mae_masked.sum(dim=[3, 4])
        mask_sum = self.masks.sum(dim=[3, 4])
        mae_masked_mean = mae_masked_sum / mask_sum
        mean_mae = mae_masked_mean.mean()
        return mean_mae

    def rmse(self):
        if self.masks is None:
            raise ValueError("Mask is required for RMSE calculation")
        output_masked = self.outputs * self.masks
        target_masked = self.targets * self.masks
        mse_masked = F.mse_loss(output_masked, target_masked, reduction='none') * self.masks
        mse_masked_sum = mse_masked.sum(dim=[3, 4])
        mask_sum = self.masks.sum(dim=[3, 4])
        mse_masked_mean = mse_masked_sum / mask_sum
        rmse_masked = torch.sqrt(mse_masked_mean)
        mean_rmse = rmse_masked.mean()
        return mean_rmse


    def psnr(self, max_pixel=1.0):
        if self.masks is None:
            raise ValueError("Mask is required for RMSE calculation")
        output_masked = self.outputs * self.masks
        target_masked = self.targets * self.masks
        mse_masked = F.mse_loss(output_masked, target_masked, reduction='none') * self.masks
        mse_masked_sum = mse_masked.sum(dim=[3, 4])
        mask_sum = self.masks.sum(dim=[3, 4])
        mse_masked_mean = mse_masked_sum / mask_sum
        psnr_masked = 20 * torch.log10(max_pixel / torch.sqrt(mse_masked_mean))
        mean_psnr = psnr_masked.mean()
        return mean_psnr

    def sam(self):
        if self.masks is None:
            raise ValueError("Mask is required for SAM calculation")
        norm_out = F.normalize(self.outputs, p=2, dim=2)
        norm_tar = F.normalize(self.targets, p=2, dim=2)
        dot_product = (norm_out * norm_tar).sum(2).clamp(-1, 1)
        angles = torch.acos(dot_product)
        angles_masked = angles.unsqueeze(2) * self.masks
        angles_sum = angles_masked.sum(dim=[-2, -1])
        mask_sum = self.masks.sum(dim=[-2, -1])
        angles_mean = angles_sum / mask_sum
        mean_sam = angles_mean.mean()
        return mean_sam

    # @staticmethod
    # def ssim(output, target, mask=None, data_range=1.0):
    #     if mask is None:
    #         raise ValueError("Mask is required for SSIM calculation")
    #     masked_output = output * mask
    #     masked_target = target * mask
    #     return ssim(masked_output, masked_target, data_range=data_range, size_average=True)

    def gaussian(self, window_size, sigma):
        """Create a 1D Gaussian window."""
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """Create a 2D Gaussian window."""
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def ssim(self, img1, img2, mask, size_average=True):
        """Compute the SSIM index between two images.
        Reference: https://github.com/Po-Hsun-Su/pytorch-ssim
        """
        if img1.size() != img2.size():
            raise ValueError("Input images must have the same dimensions.")

        (_, _, channel, _, _) = img1.size()
        if channel != self.channel:
            self.channel = channel
            self.window = self.create_window(self.window_size, self.channel)
            self.window = self.window.type_as(img1).to(img1.device)

        # Apply mask to images
        img1 = img1 * mask
        img2 = img2 * mask

        # Preparing the SSIM window
        window = self.window.expand(img1.size(0), channel, self.window_size, self.window_size).contiguous()

        # SSIM calculation
        mu1 = F.conv2d(img1.view(-1, *img1.shape[-3:]), window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2.view(-1, *img2.shape[-3:]), window, padding=self.window_size // 2, groups=channel)
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2.pow(2)
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1 * mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.view(img1.size(0), img1.size(1), -1).mean(2)



    def lpips(self, output, target, mask=None):
        if mask is None:
            raise ValueError("Mask is required for LPIPS calculation")
        valid = mask > 0.5
        valid = valid.expand_as(output)

        if valid.any():
            valid_output = output * valid
            valid_target = target * valid
            return self.lpips_loss_fn(valid_output[0,0], valid_target[0,0]).mean()
        else:
            return torch.tensor(float('nan'))

    def evaluate(self, psnr_max=1.0):
        return {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "PSNR": self.psnr(max_pixel=psnr_max),
            "SAM": self.sam(),
            # "SSIM": self.ssim(output, target, mask),
            # "LPIPS": self.lpips(output, target, mask)
        }

#
#
# from torch.utils.data.dataloader import default_collate
# def custom_collate_fn(batch):
#     # Filter out all None values
#     filtered_batch = [b for b in batch if b is not None]
#     if len(filtered_batch) == 0:
#         return None  # or handle this scenario appropriately
#     return default_collate(filtered_batch)


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
        # return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=custom_collate_fn, drop_last=True)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def run(self):
        metrics = Metrics(self.args)
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

        # Remove B10 from the outputs and targets for evaluation
        # toa_no_b10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
        # outputs = outputs[:, :, toa_no_b10, :, :]
        # targets = targets[:, :, toa_no_b10, :, :]
        targets = targets[:, :, self.args.eval_bands, :, :]

        # evaluation_results = metrics.evaluate(outputs, targets, masks)
        # print(evaluation_results)
        # def eval_step(engine, batch):
        #     return batch
        #
        # default_evaluator = Engine(eval_step)
        # metric = SSIM(data_range=1.0)
        # metric.attach(default_evaluator, 'ssim')
        # print(outputs.flatten(0,1).shape)
        # state = default_evaluator.run([[outputs.flatten(0,1), targets.flatten(0,1)]])
        # print(state.metrics['ssim'])
        return outputs, targets, masks

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