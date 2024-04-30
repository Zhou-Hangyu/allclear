import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
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
    def __init__(self, args):
        self.device = torch.device(args.device)
        # self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
    @staticmethod
    def mae(output, target, mask=None):
        if mask is None:
            raise ValueError("Mask is required for MAE calculation")
        mae_mask = torch.mean(F.l1_loss(output, target, reduction='none') * mask)
        return mae_mask

    @staticmethod
    def rmse(output, target, mask=None):
        if mask is None:
            raise ValueError("Mask is required for RMSE calculation")
        mse_mask = torch.mean(F.mse_loss(output, target, reduction='none') * mask)
        return torch.sqrt(mse_mask)


    @staticmethod
    def psnr(output, target, mask=None, max_pixel=1.0):
        if mask is None:
            raise ValueError("Mask is required for PSNR calculation")
        mse_mask = torch.mean(F.mse_loss(output, target, reduction='none') * mask)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse_mask))

    # @staticmethod
    # def ssim(output, target, mask=None, data_range=1.0):
    #     if mask is None:
    #         raise ValueError("Mask is required for SSIM calculation")
    #     masked_output = output * mask
    #     masked_target = target * mask
    #     return ssim(masked_output, masked_target, data_range=data_range, size_average=True)

    @staticmethod
    def gaussian_window(size, sigma, device):
        """
        Create a Gaussian window for SSIM computation.
        """
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        grid = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        grid /= grid.sum()
        return grid.view(1, 1, 1, -1) * grid.view(1, 1, 1, -1).transpose(-1, -2)

    def ssim(self, output, target, mask=None, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=1.0):
        if mask is None:
            raise ValueError("Mask is required for SSIM calculation")

        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        window = self.gaussian_window(window_size, sigma, self.device).to(output.device)

        mu_x = F.conv2d(output, window, padding=window_size // 2, groups=1)
        mu_y = F.conv2d(target, window, padding=window_size // 2, groups=1)

        sigma_x = F.conv2d(output * output, window, padding=window_size // 2, groups=1)
        sigma_y = F.conv2d(target * target, window, padding=window_size // 2, groups=1)
        sigma_xy = F.conv2d(output * target, window, padding=window_size // 2, groups=1)

        mu_x_mu_y = mu_x * mu_y
        sigma_x += -mu_x * mu_x
        sigma_y += -mu_y * mu_y
        sigma_xy += -mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        if mask is not None:
            ssim_map = ssim_map * mask

        return ssim_map[mask.bool()].mean()

    def sam(self, output, target, mask=None):
        if mask is None:
            raise ValueError("Mask is required for SAM calculation")

        flat_output = output.flatten(start_dim=2)  # Flatten B, T, H, W into one dimension
        flat_target = target.flatten(start_dim=2)
        norm_out = F.normalize(flat_output, p=2, dim=2)
        norm_tar = F.normalize(flat_target, p=2, dim=2)
        dot_product = (norm_out * norm_tar).sum(2)

        flat_mask = mask.flatten(start_dim=2)
        if flat_mask.any():
            masked_dot_product = dot_product[flat_mask.bool()].clamp(-1, 1)
            angle = torch.acos(masked_dot_product)
            return angle.mean()
        else:
            return torch.tensor(float('nan'))

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

    def evaluate(self, output, target, mask):
        return {
            "MAE": self.mae(output, target, mask),
            "RMSE": self.rmse(output, target, mask),
            "PSNR": self.psnr(output, target, mask),
            # "SSIM": self.ssim(output, target, mask),
            # "SAM": self.sam(output, target, mask),
            # "LPIPS": self.lpips(output, target, mask)
        }
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from ignite.utils import *
# class Metrics:
#     pass


from torch.utils.data.dataloader import default_collate
def custom_collate_fn(batch):
    # Filter out all None values
    filtered_batch = [b for b in batch if b is not None]
    if len(filtered_batch) == 0:
        return None  # or handle this scenario appropriately
    return default_collate(filtered_batch)


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
        def eval_step(engine, batch):
            return batch

        default_evaluator = Engine(eval_step)
        metric = SSIM(data_range=1.0)
        metric.attach(default_evaluator, 'ssim')
        print(outputs.flatten(0,1).shape)
        state = default_evaluator.run([[outputs.flatten(0,1), targets.flatten(0,1)]])
        print(state.metrics['ssim'])
        return outputs, targets

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