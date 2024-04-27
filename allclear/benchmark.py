import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys

sys.path.append("/home/hz477/declousion/baselines/UnCRtainTS/model")

# Import model classes
from allclear import CRDataset
from allclear.baselines import UnCRtainTS, LeastCloudy, Mosaicing

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Metrics:
    @staticmethod
    def psnr(output, target, max_pixel=1.0):
        mse = F.mse_loss(output, target)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))

    @staticmethod
    def mae(output, target):
        return F.l1_loss(output, target)

    @staticmethod
    def rmse(output, target):
        return torch.sqrt(F.mse_loss(output, target))

    def evaluate(self, output, target):
        return {"PSNR": self.psnr(output, target), "MAE": self.mae(output, target), "RMSE": self.rmse(output, target)}


class BenchmarkEngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.data_loader = self.setup_data_loader()
        self.model = UnCRtainTS(args)

    def setup_data_loader(self):
        dataset = CRDataset(
            self.args.data_path, self.args.metadata_path, self.args.selected_rois, self.args.time_span, self.args.cloud_percentage_range
        )
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def run(self):
        metrics = Metrics()
        outputs_all = []
        targets_all = []
        for data in self.data_loader:
            # print(data["timestamps"])
            # images, batch_positions = data["input_images"].to(self.device), data["timestamps"].float().mean(dim=0)[None].to(self.device)
            targets = data["target"].to(self.device)
            targets_all.append(targets.cpu())
            with torch.no_grad():
                # outputs = self.model.forward(images, batch_positions=batch_positions)
                outputs = self.model.forward(data)
                outputs_all.append(outputs[0].cpu())
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
    parser.add_argument("--model-name", type=str, required=True, help="Model to use for benchmarking")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for data loading")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--data-split", type=str, default="train", help="Data split to use [train, val, test]")
    parser.add_argument("--device", type=str, required=True, help="Device to run the model on")
    parser.add_argument("--dataset-type", type=str, default="SEN12MS-CR", choices=["SEN12MS-CR", "SEN12MS-CR-TS"], help="Type of dataset")
    parser.add_argument("--input-t", type=int, default=3, help="Number of input time points (for time-series datasets)")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--input-dim", type=int, required=True, help="Input dimension for the model")
    parser.add_argument("--output-dim", type=int, required=True, help="Output dimension for the model")
    parser.add_argument("--selected-rois", type=str, nargs="+", required=True, help="Selected ROIs for benchmarking")
    parser.add_argument("--time-span", type=int, default=3, help="Time span for the dataset")
    parser.add_argument("--cloud-percentage-range", type=int, nargs=2, default=[20, 30], help="Cloud percentage range for the dataset")

    uc_args = parser.add_argument_group("UnCRtainTS Arguments")
    uc_args.add_argument()

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    engine = BenchmarkEngine(args)
    engine.run()
