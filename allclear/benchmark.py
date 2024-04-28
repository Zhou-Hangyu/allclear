import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import sys


sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/model")
sys.path.append("/share/hariharan/cloud_removal/allclear")

from allclear import CRDataset
from allclear import UnCRtainTS, LeastCloudy, Mosaicing, Simple3DUnet
from baselines.UnCRtainTS.model.parse_args import create_parser

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Metrics:
    @staticmethod
    def psnr(output, target, max_pixel=1.0):
        # mse = F.mse_loss(output, target, reduction='none')
        mse = F.mse_loss(output, target)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))

    @staticmethod
    def mae(output, target):
        return F.l1_loss(output, target)

    @staticmethod
    def rmse(output, target):
        return torch.sqrt(F.mse_loss(output, target))

    def ssim(self):
        pass

    def evaluate(self, output, target):
        return {"PSNR": self.psnr(output, target), "MAE": self.mae(output, target), "RMSE": self.rmse(output, target)}


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
            self.args.data_path, self.args.metadata_path, self.args.selected_rois, self.args.time_span, self.args.cloud_percentage_range
        )
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def run(self):
        metrics = Metrics()
        outputs_all = []
        targets_all = []
        for data in tqdm(self.data_loader, desc="Running Benchmark"):
            with torch.no_grad():
                data = self.model.preprocess(data)
                targets = data["target"].to(self.device)
                targets_all.append(targets.cpu())
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
        targets = torch.cat(targets_all, dim=0).unsqueeze(1)
        evaluation_results = metrics.evaluate(outputs, targets)
        print(evaluation_results)
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