import argparse
from abc import ABC, abstractmethod
import os, json, datetime, sys
from datetime import datetime
import torch

sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/model/")
sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/")

class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def preprocess(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        pass

    @abstractmethod
    def forward(self, inputs):
        pass


class UnCRtainTS(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        from baselines.UnCRtainTS.model.src.model_utils import get_model, load_checkpoint
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
        from baselines.UnCRtainTS.model.src.utils import str2list
        from baselines.UnCRtainTS.model.parse_args import create_parser
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

    def preprocess(self, inputs):
        inputs["input_images"] = torch.clip(inputs["input_images"]/10000, 0, 1).to(self.device)
        inputs["target"] = torch.clip(inputs["target"]/10000, 0, 1).to(self.device)
        inputs["input_cloud_masks"] = inputs["input_cloud_masks"].to(self.device)
        inputs["input_shadow_masks"] = inputs["input_shadow_masks"].to(self.device)
        return inputs

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """
        input_imgs = inputs["input_images"]
        target_imgs = inputs["target"]
        masks = inputs["input_cloud_masks"]
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
        # return out, var
        return {"output": out, "variance": var}


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
    

class Simple3DUnet(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # to_date = lambda string: datetime.strptime(string, "%Y-%m-%d")
        to_date = lambda string: datetime.strptime(string, "%Y-%m-%d").timestamp()

        self.config = self.get_config()  # bug
        self.model = self.get_model().to(self.device)
        self.model.eval()

    def get_model_config(self):
        pass

    def get_model(self):

        from SimpleUnet.diffusers_src import UNet3DConditionModel

        down_block_dict = {"C": "DownBlock3D", "A": "CrossAttnDownBlock3D", "J": "DownBlockJust2D", "R": "CrossAttnDownBlock2D1D"}
        up_block_dict = {"C": "UpBlock3D", "A": "CrossAttnUpBlock3D", "J": "UpBlockJust2D", "R": "CrossAttnUpBlock2D1D"}
        down_block_list = [down_block_dict[b] for b in self.config.model_blocks]
        up_block_list = [up_block_dict[b] for b in self.config.model_blocks][::-1]

        model = UNet3DConditionModel(
            sample_size=self.config.image_size,  # the target image resolution
            in_channels=self.config.in_channel,  # the number of input channels, 3 for RGB images
            out_channels=self.config.out_channel,  # the number of output channels
            layers_per_block=self.config.LPB,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, self.config.max_dim, self.config.max_dim, self.config.max_dim, self.config.max_dim)[:len(self.config.model_blocks)],  # the number of output channels for each UNet block
            down_block_types=down_block_list,  # the down block sequence
            up_block_types=up_block_list,  # the up block sequence
            norm_num_groups=self.config.norm_num_groups,  # the number of groups for normalization
        )

        PATH = "/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v45_0426_I12O3T12_BlcCRRAAA_LR2e_05_LPB1_GNorm4_MaxDim512_NoTimePerm/model_6.pt"
        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        model.eval()

        return model

    def get_config(self):

        class Args:
            image_size = 256
            in_channel = 12
            out_channel = 3
            LPB = 1
            max_dim = 512
            model_blocks = 'CRRAAA'
            cross_attention_dim = 32
            norm_num_groups = 4

        config = Args()
        return config

    def preprocess(self, inputs):
        inputs["input_images"] = torch.clip(inputs["input_images"]/10000, 0, 1).to(self.device)
<<<<<<< HEAD
        inputs["target"] = torch.clip(inputs["target"]/10000, 0, 1).to(self.device)[:, 1:4]
        inputs["cloud_masks"] = inputs["cloud_masks"].to(self.device)
=======
        inputs["target"] = torch.clip(inputs["target"]/10000, 0, 1).to(self.device)
        inputs["input_cloud_masks"] = inputs["input_cloud_masks"].to(self.device)
        inputs["input_shadow_masks"] = inputs["input_shadow_masks"].to(self.device)
>>>>>>> e696019ed3c3691acfa033c368b78a9948cf7b95
        return inputs
    
    def compute_day_differences(self, timestamps):
        timestamp_diffs = timestamps - timestamps[:, 0:1]
        day_diffs = (timestamp_diffs / 86400).round().to(self.device)
        return day_diffs
    
    def update_model_position_token(self, model, day_diffs):
        for p1, p2 in model.named_parameters():
            if 'position' in p1:
                p2.data = day_diffs

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """

        """
        Transformation
            - input_imgs: (B, T, C, H, W) -> (B, C, T, H, W)
            - input_imgs: [0,1] -> [-1,1]
            - input_imgs channels: (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12) -> (B4, B3, B2, B5, B6, B7, B8, B8A, B11, B12) + (dummy, dummy), dummy = 0
        """

        BS, T, _, H, W = inputs["input_images"].shape
        
        # Input transformation
        input_imgs = inputs["input_images"] * 2 - 1
        input_imgs = input_imgs.permute(0, 2, 1, 3, 4)
        input_imgs = input_imgs[:, [3, 2, 1, 4, 5, 6, 7, 8, 11, 12, 12, 12]]
        # input_imgs[:, -2:] = -1
        input_imgs = input_imgs.to(self.device)

        input_buffer = torch.ones((BS, 12, 12, H, W)).to(self.device) * -1
        input_buffer[:, :, :3] = input_imgs
        input_buffer[:, :, 3:6] = input_imgs
        input_buffer[:, :, 6:9] = input_imgs
        input_buffer[:, :, 9:12] = input_imgs

        # Update day counts and day token
        # day_counts = self.compute_day_differences(inputs["timestamps"])
        day_counts = torch.arange(6).to(self.device).unsqueeze(0).repeat(BS, 1).float() * 3

        self.update_model_position_token(self.model, day_counts)
        emb = torch.zeros((self.args.batch_size, 2, 1024)).to(self.args.device)
        buffer = torch.zeros_like(inputs["target"])

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = self.model(input_buffer, 1, encoder_hidden_states=emb, return_dict=False)[0] * 0.5 + 0.5
        # buffer = inputs["input_images"][:, T//2]
        prediction = torch.flip(prediction[:, :, 3], dims=[1])

        #save prediction and input_imgs, and targets results in numpy format
        # self.args.res_dir = "/share/hariharan/ck696/Decloud/UNet/results/test_buffer"
        # import numpy as np
        # np.save(f"{self.args.res_dir}/prediction.npy", prediction.cpu().numpy())
        # np.save(f"{self.args.res_dir}/input_imgs.npy", input_imgs.cpu().numpy())
        # np.save(f"{self.args.res_dir}/input_buffer.npy", input_buffer.cpu().numpy())
        # np.save(f"{self.args.res_dir}/targets.npy", inputs["target"].cpu().numpy())
        # # print min max of prediction and input_buffer
        # print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")
        # print(f"Input buffer min: {input_buffer.min()}, max: {input_buffer.max()}")
        # assert 0 == 1

        return  {"output": prediction}
