import argparse
from abc import ABC, abstractmethod
import os, json, datetime, sys
from datetime import datetime
import torch

sys.path.append("/home/hz477/declousion/baselines/UnCRtainTS/model")


# Import model classes
from baselines.UnCRtainTS.model.src.model_utils import get_model, load_checkpoint
from baselines.UnCRtainTS.model.src.utils import str2list
from baselines.UnCRtainTS.model.parse_args import create_parser


class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass


class UnCRtainTS(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # to_date = lambda string: datetime.strptime(string, "%Y-%m-%d")
        to_date = lambda string: datetime.strptime(string, "%Y-%m-%d").timestamp()
        self.S1_LAUNCH = to_date("2014-04-03")
        self.S2_BANDS = 13

        self.config = self.get_config()  # bug
        self.model = get_model(self.config).to(self.device)

        ckpt_n = f"_epoch_{self.config.resume_at}" if self.config.resume_at > 0 else ""
        load_checkpoint(self.config, self.config.weight_folder, self.model, f"model{ckpt_n}")

        self.model.eval()

    def get_model_config(self):
        pass

    def get_config(self):
        parser = create_parser(mode="test")
        conf_path = os.path.join(self.args.baseline_base_path, self.args.weight_folder, self.args.experiment_name, "conf.json")
        with open(conf_path, "r") as f:
            model_config = json.load(f)
            t_args = argparse.Namespace()
            # do not overwrite the following flags by their respective values in the config file
            no_overwrite = [
                "pid",
                "device",
                "resume_at",
                "trained_checkp",
                "res_dir",
                "weight_folder",
                "root1",
                "root2",
                "root3",
                "max_samples_count",
                "batch_size",
                "display_step",
                "plot_every",
                "export_every",
                "input_t",
                "region",
                "min_cov",
                "max_cov",
                "f",
            ]
            conf_dict = {key: val for key, val in model_config.items() if key not in no_overwrite}
            for key, val in vars(self.args).items():
                if key in no_overwrite:
                    conf_dict[key] = val
            t_args.__dict__.update(conf_dict)
            config, unknown = parser.parse_known_args(namespace=t_args)  # avoid error for unknown arguments (e.g., -f from jupyter notebook)
        config = str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
        return config

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """
        # print(inputs)
        input_imgs = inputs["input_images"].to(self.device)
        target_imgs = inputs["target"].to(self.device)
        masks = inputs["cloud_masks"].to(self.device)

        capture_dates = inputs["timestamps"]
        # Dates handling (see `dataLoader.py` and `train_reconstruct.py`)
        # s2_td = [(d - self.S1_LAUNCH).days for d in capture_dates]
        # dates = torch.tensor(s2_td, dtype=torch.float32).to(self.device)
        dates = capture_dates - self.S1_LAUNCH

        model_inputs = {"A": input_imgs, "B": target_imgs, "dates": dates, "masks": masks}
        with torch.no_grad():
            self.model.set_input(model_inputs)
            self.model.forward()
            self.model.get_loss_G()
            self.model.rescale()
            out = self.model.fake_B
            if hasattr(self.model.netG, "variance") and self.model.netG.variance is not None:
                var = self.model.netG.variance
                self.model.netG.variance = None
            else:
                var = out[:, :, self.S2_BANDS :, ...]
            out = out[:, :, : self.S2_BANDS, ...]
            # TODO: add uncertainty calculation and results saving.
        return out, var


class LeastCloudy(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def get_model_config(self):
        # No specific model configuration required for least cloudy as it's not a learning-based method
        return None

    def forward(self, inputs):
        input_imgs = inputs["images"]
        masks = inputs["masks"]

        # Determine the least cloudy image by summing up the cloud masks for each time point
        cloudiness = masks.sum(dim=(2, 3))  # Sum over height and width
        least_cloudy_index = cloudiness.argmin(dim=1)

        # Select the least cloudy image for each example in the batch
        batch_indices = torch.arange(input_imgs.shape[0])
        least_cloudy_img = input_imgs[batch_indices, least_cloudy_index]

        return least_cloudy_img


class Mosaicing(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def get_model_config(self):
        # No specific model configuration required for mosaicing as it's not a learning-based method
        return None

    def forward(self, inputs):
        input_imgs = inputs["images"]
        masks = inputs["masks"]

        # Vectorized operation
        # Assuming masks use 1 for cloudy and 0 for clear
        # Invert masks: 0 for cloudy, 1 for clear
        clear_masks = 1 - masks
        # Use clear_masks to zero out cloudy pixels
        clear_pixels = input_imgs * clear_masks
        # Sum the pixel values along the time dimension
        sum_clear_pixels = clear_pixels.sum(dim=1)
        # Sum the clear views along the time dimension
        sum_clear_views = clear_masks.sum(dim=1)
        # Avoid division by zero by replacing 0 with 1 for clear view counts
        sum_clear_views[sum_clear_views == 0] = 1
        # Compute the average by dividing sum of pixel values by number of clear views
        mosaiced_img = sum_clear_pixels / sum_clear_views

        # For pixels with no clear views at all, set to 0.5
        no_clear_views = sum_clear_views == 0
        mosaiced_img[no_clear_views] = 0.5

        return mosaiced_img
