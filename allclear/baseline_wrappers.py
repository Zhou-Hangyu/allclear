import argparse
from abc import ABC, abstractmethod
import os, json, datetime, sys
from datetime import datetime
import torch
import torch.nn.functional as F
# from allclear import benchmark_visualization

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

if "ck696" in os.getcwd():
    sys.path.append("/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model")
    sys.path.append("/share/hariharan/ck696/allclear/baselines")
    sys.path.append("/share/hariharan/ck696/allclear")
else:
    sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS/model/")
    sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/")

def s2_boa2toa(s2_boa):
    """Cast Sentinel-2 Bottom of Atmosphere (BOA) data to the shape of Top of Atmosphere (TOA) data.
    Specifically: BxTx12xHxW -> BxTx13xHxW
    - Use zeros for band10.
    - Remove bands not present in TOA data."""
    B, T, _, H, W = s2_boa.shape
    before_band10 = s2_boa[:, :, :10, :, :]
    zeros = torch.zeros((B, T, 1, H, W), device=s2_boa.device)
    after_band10 = s2_boa[:, :, 10:12, :, :]
    s2_toa = torch.cat((before_band10, zeros, after_band10), dim=2)
    return s2_toa

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
        
        self.args = args
        self.config = self.get_config()  # bug
        self.model = get_model(self.config).to(self.device)

        ckpt_n = f"_epoch_{self.config.resume_at}" if self.config.resume_at > 0 else ""
        load_checkpoint(self.config, self.config.weight_folder, self.model, f"model{ckpt_n}")
        self.model.eval()

        if "noSAR_1" in self.args.experiment_name:
            self.num_input_dims = 13
        else:
            self.num_input_dims = 15
        self.S2_BANDS = 13

    def get_model_config(self):
        pass

    def get_config(self):
        from baselines.UnCRtainTS.model.src.utils import str2list
        from baselines.UnCRtainTS.model.parse_args import create_parser
        parser = create_parser(mode="test")
        logger.info(f"Using UnCRtainTS config: {self.args.baseline_base_path}")
        logger.info(f"Using UnCRtainTS weight_folder: {self.args.weight_folder}")
        logger.info(f"Using UnCRtainTS experiment_name: {self.args.experiment_name}")

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
        inputs["input_images"] = inputs["input_images"].to(self.device)
        inputs["target"] = inputs["target"].to(self.device)
        inputs["input_cld_shdw"] = inputs["input_cld_shdw"].to(self.device)
        inputs["input_images"] = inputs["input_images"].permute(0, 2, 1, 3, 4)[:,:,:self.num_input_dims]
        inputs["target"] = inputs["target"].permute(0, 2, 1, 3, 4)[:,:,:self.S2_BANDS]
        inputs["input_cld_shdw"] = torch.clip(inputs["input_cld_shdw"].sum(dim=1),0,1)

        if self.args.dataset_type == "AllClear" and "diagonal_1" in self.args.exp_name:
            inputs["input_images"] = torch.cat([inputs["input_images"][:,:,2:], inputs["input_images"][:,:,:2]], dim=2)

        return inputs

    def forward(self, inputs, batch_id=None):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """
        input_imgs = inputs["input_images"]
        target_imgs = inputs["target"]
        masks = inputs["input_cld_shdw"]
        dates = inputs["time_differences"]
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

        # if self.args.draw_vis:
        #     inputs["output"] = out.cpu()
        #     benchmark_visualization(inputs, self.args)
            
        return {"output": out, "variance": var}


class LeastCloudy(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def get_model_config(self):
        # No specific model configuration required for least cloudy as it's not a learning-based method
        return None

    def preprocess(self, inputs):
        inputs["input_images"] = inputs["input_images"].permute(0,2,1,3,4).to(self.device)
        inputs["target"] = inputs["target"].permute(0,2,1,3,4).to(self.device)
        inputs["masks"] = inputs["input_cld_shdw"].permute(0,2,1,3,4).to(self.device)
        return inputs

    def forward(self, inputs):
        input_imgs = inputs["input_images"]
        cloudiness = inputs["masks"].sum(dim=(2, 3, 4))  # Sum over height and width
        least_cloudy_index = cloudiness.argmin(dim=1)
        batch_indices = torch.arange(input_imgs.shape[0])
        least_cloudy_img = input_imgs[batch_indices, least_cloudy_index, :13].unsqueeze(1)
        return {"output": least_cloudy_img}


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

class DAE(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.eval_mode = "s2p"  # TODO: implement s2s too.
        self.model = self.get_model().to(self.device)
        self.model.eval()
        self.channels = {
            "s2_toa": list(range(1, 14)),
            "s1": [1, 2],
            "landsat8": list(range(1, 12)),
            "landsat9": list(range(1, 12)),
            "cld_shdw": [2, 5],
            "dw": [1],
        }
    def get_model_config(self):
        pass

    def get_model(self):
        from baselines.DAE.diffusers_src import UNet3DConditionModel
        down_block_dict = {"C": "DownBlock3D", "A": "CrossAttnDownBlock3D", "J": "DownBlockJust2D", "R": "CrossAttnDownBlock2D1D"}
        up_block_dict = {"C": "UpBlock3D", "A": "CrossAttnUpBlock3D", "J": "UpBlockJust2D", "R": "CrossAttnUpBlock2D1D"}
        down_block_list = [down_block_dict[b] for b in self.args.dae_model_blocks]
        up_block_list = [up_block_dict[b] for b in self.args.dae_model_blocks][::-1]
        model = UNet3DConditionModel(
            sample_size=self.args.dae_image_size,  # the target image resolution
            in_channels=self.args.dae_in_channel,  # the number of input channels, 3 for RGB images
            out_channels=self.args.dae_out_channel,  # the number of output channels
            layers_per_block=self.args.dae_LPB,  # how many ResNet layers to use per UNet block
            block_out_channels=(self.args.dae_max_d0, 128, 256, self.args.dae_max_dim, self.args.dae_max_dim, self.args.dae_max_dim, self.args.dae_max_dim)[:len(self.args.dae_model_blocks)],  # the number of output channels for each UNet block
            down_block_types=down_block_list,  # the down block sequence
            up_block_types=up_block_list,  # the up block sequence
            norm_num_groups=self.args.dae_norm_num_groups,  # the number of groups for normalization
        )
        params = torch.load(self.args.dae_checkpoint, map_location=torch.device('cpu'))
        filtered_params = {k: v for k, v in params.items() if "custom_pos_embed.position" not in k}
        model.load_state_dict(filtered_params, strict=False)
        print(f"Total Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        # for key in params.keys():
        #     if "custom_pos_embed.position" in key: break
        # self.update_model_number_position_token(model) # TODO: figure out what these are
        return model

    def preprocess(self, inputs):
        if self.eval_mode == "s2p":
            # Insert a dummy image (all cloud or no info) at the correct temporal position for each sensor of the target image to mimic s2p.
            s2_toa_placeholder = torch.ones(len(self.channels['s2_toa']), 1, *inputs["target"].shape[2:])
            s1_placeholder = torch.ones(len(self.channels['s1']), 1, *inputs["target"].shape[2:]) - 1
            target_placeholder = torch.cat([s2_toa_placeholder, s1_placeholder], dim=0).squeeze(1)
            s2p_input_images = []
            s2p_targets = []
            s2p_timestamps = []
            s2p_target_indices = []
            for i in range(inputs["target"].shape[0]):
                target_index = (inputs['timestamps'][i] <= inputs['target_timestamps'][i]).sum()
                s2p_input_image = torch.cat([inputs["input_images"][i, :, :target_index],
                                             target_placeholder,
                                             inputs["input_images"][i, :, target_index:]], dim=1)
                # s2p_target = torch.cat([inputs["input_images"][i, :len(self.channels['s2_toa']), :target_index],
                #                         inputs["target"][i],
                #                         inputs["input_images"][i, :len(self.channels['s2_toa']), target_index:]], dim=1)
                s2p_timestamp = torch.cat([inputs['timestamps'][i, :target_index], inputs['target_timestamps'][i].unsqueeze(0), inputs['timestamps'][i, target_index:]], dim=0)
                s2p_input_images.append(s2p_input_image)
                # s2p_targets.append(s2p_target)
                s2p_timestamps.append(s2p_timestamp)
                s2p_target_indices.append(target_index)
            inputs["input_images"] = torch.stack(s2p_input_images)
            # inputs["target"] = torch.stack(s2p_targets)
            inputs["timestamps"] = torch.stack(s2p_timestamps)
            inputs["time_differences"] = self.compute_day_differences(inputs["timestamps"])
            inputs["target_indices"] = torch.tensor(s2p_target_indices)
            # print(inputs['timestamps'].shape, inputs['target_timestamps'].shape)
            # target_index = (inputs['timestamps'] <= inputs['target_timestamps']).sum(dim=1)
            # s2p_input_images = torch.cat([inputs["input_images"][:,:,:target_index],
            #                                     target_placeholder,
            #                                     inputs["input_images"][:,:,target_index:]], dim=2)
            # inputs["target"] = torch.cat([inputs["input_images"][:,:,:target_index],
            #                                     inputs["target"],
            #                                     inputs["input_images"][:,:,target_index:]], dim=2)
            # inputs["input_images"] = s2p_input_images
        inputs["input_images"] = inputs["input_images"].to(self.device)  # (B, C, T, H, W)
        inputs["target"] = inputs["target"].permute(0, 2, 1, 3, 4)
        return inputs  # Other preprocessing steps handled by the dataloader

    def compute_day_differences(self, timestamps):
        timestamp_diffs = timestamps - timestamps[:, 0:1]
        day_diffs = (timestamp_diffs / 86400).round().to(self.device)
        return day_diffs
    
    def update_model_position_token(self, model, day_diffs):
        for p1, p2 in model.named_parameters():
            if 'position' in p1:
                p2.data = day_diffs

    # def update_model_number_position_token(self, model):
    #
    #     for name, p2 in model.named_buffers():
    #         # print(name)
    #         if "custom_pos_embed.pe" in name:
    #             _, old_max_seq_len, old_embed_dim = p2.shape
    #             new_buffer = torch.zeros((_, self.num_pos_tokens, old_embed_dim))
    #
    #             # Access the attribute dynamically to replace the original buffer
    #             parts = name.split('.')
    #             obj = model
    #             for part in parts[:-1]:
    #                 obj = getattr(obj, part)
    #             setattr(obj, parts[-1], new_buffer)

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, 1, C, H, W)
            - masks: (B, T, 1, H, W)
            - dates: (B, T)
        """

        """
        Transformation
            - input_imgs: (B, T, C, H, W) -> (B, C, T, H, W)
            - input_imgs: [0,1] -> [-1,1]
            - input_imgs channels: (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12) -> (B4, B3, B2, B5, B6, B7, B8, B8A, B11, B12) + (dummy, dummy), dummy = 0
        """
        self.update_model_position_token(self.model, inputs["time_differences"])
        emb = torch.zeros((inputs["input_images"].shape[0], 2, 1024)).to(self.device) # TODO: figure out what these are
        with torch.no_grad():
            # pred = self.model(inputs["input_images"][:,:,:4,...], 1, encoder_hidden_states=emb, return_dict=False)[0]
            pred = self.model(inputs["input_images"], 1, encoder_hidden_states=emb, return_dict=False)[0]
            if self.eval_mode == "s2p":
                preds = []
                for i in range(pred.shape[0]):
                    preds.append(pred[i, :,inputs["target_indices"][i], ...].unsqueeze(1))
                pred = torch.stack(preds, dim=0)
            pred = pred.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)

        # # save prediction and input_imgs, and targets results in numpy format
        # self.args.res_dir = "/share/hariharan/ck696/Decloud/UNet/results/test_buffer"
        # import numpy as np
        # np.save(f"{self.args.res_dir}/prediction.npy", prediction.cpu().numpy())
        # np.save(f"{self.args.res_dir}/input_imgs.npy", input_imgs.cpu().numpy())
        # np.save(f"{self.args.res_dir}/input_buffer.npy", input_buffer.cpu().numpy())
        # np.save(f"{self.args.res_dir}/targets.npy", inputs["target"].cpu().numpy())
        # # print min max of prediction and input_buffer
        # print(f"Input_imgs min: {inputs["input_images"].min()}, max: {inputs["input_images"].max()}")
        # print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")

        # print(f"Input buffer min: {input_buffer.min()}, max: {input_buffer.max()}")
        # assert 0 == 1

        return {"output": pred}
    

class CTGAN(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.image_size = 256
        self.load_gen = args.ctgan_gen_checkpoint

        from baselines.CTGAN.model.CTGAN import CTGAN_Generator

        model_GEN = CTGAN_Generator(self.image_size)
        model_GEN.load_state_dict(torch.load(self.load_gen))

        self.model = model_GEN
        self.model = self.model.to(self.device)
        self.model.eval()

        # Bands: R, G, B, NIR
        if "Pretrain" in args.ctgan_gen_checkpoint:
            self.bands = (3,2,1,7)
        elif "checkpoints" in args.ctgan_gen_checkpoint:
            self.bands = (3,2,1,7)
        else:
            raise ValueError("Invalid CTGAN model checkpoint")
        
        print("bands", self.bands)

    def get_model_config(self):
        pass

    def preprocess(self, inputs):
        inputs["input_images"] = inputs["input_images"][:, self.bands].to(self.device)
        inputs["target"] = inputs["target"][:, self.bands].permute(0,2,1,3,4).to(self.device)
        return inputs

    def preprocess_input(self, batch):
        real_As = batch["input_images"] * 2 - 1
        return [real_As[:,:,0].cuda(), real_As[:,:,1].cuda(), real_As[:,:,2].cuda()]
    
    def forward(self, inputs):
        real_A  = self.preprocess_input(inputs)
        fake_B, cloud_mask, aux_pred = self.model(real_A)
        output = fake_B.unsqueeze(1) * 0.5 + 0.5
        return {"output": output}


class UTILISE(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        if "ck696" in os.getcwd():
            sys.path.append("/share/hariharan/ck696/allclear/baselines/U-TILISE")
        else:
            sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/U-TILISE")
        

        print(f"====== UTILISE ======")
        print(f"checkpoint: {args.utilise_checkpoint}")
        print(f"export_path: {args.experiment_output_path}")
        from lib.eval_tools_benchmarking import Imputation

        if "band4" in args.experiment_output_path:
            self.num_channels = 4
        elif "w_s1" in args.experiment_output_path:
            self.num_channels = 15
        elif "wo_s1" in args.experiment_output_path:
            self.num_channels = 13
        else:
            raise ValueError("Invalid experiment_output_path")

        print(f"Using {self.num_channels} bands for UTILISE")
        utilise = Imputation(args.utilise_config, 
                             method='utilise', 
                             checkpoint=args.utilise_checkpoint,
                             num_channels=self.num_channels,
                             )
        self.model = utilise.model.to(self.device)
        self.model.eval()
        print("Note!!! Using UTILISE is a seq-to-seq model. Using the middle frame fro prediction may not be accurate.")


    def get_model_config(self):
        pass

    def preprocess(self, inputs):
        inputs["input_images"] = inputs["input_images"].permute(0, 2, 1, 3, 4)[:,:,:self.num_channels].contiguous().to(self.device)
        inputs["target"] = inputs["target"][:,:self.num_channels].to(self.device).permute(0,2,1,3,4)
        return inputs

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, 1, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """
        # Model I/O (bs, t, c, h, w)
        output = self.model(inputs["input_images"])[:,2:3]
        return {"output": output}
    


class PMAA(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        if "ck696" in os.getcwd():
            sys.path.append("/share/hariharan/ck696/allclear/baselines/PMAA/model")
        else:
            sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/PMAA/model")
        
        checkpoint = args.pmaa_checkpoint
            
        print(f"====== PMAA ======")
        print(f"Using PMAA checkpoint: {checkpoint}")

        from pmaa import PMAA
        self.model = PMAA(32, 4) if "pmaa_old" not in checkpoint else PMAA(32, 3)
        def replace_batchnorm(model):
            for name, child in model.named_children():
                if isinstance(child, torch.nn.BatchNorm2d):
                    child: torch.nn.BatchNorm2d = child
                    setattr(model, name, torch.nn.InstanceNorm2d(child.num_features))
                else:
                    replace_batchnorm(child)
        replace_batchnorm(self.model)
        # load
        self.model.load_state_dict(torch.load(checkpoint))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_bands = (3,2,1,7)
        self.output_bands = (3,2,1,7) if "pmaa_old" not in checkpoint else (3,2,1)

        print(f"Using PMAA input bands: {self.input_bands}")
        print(f"Using PMAA output bands: {self.output_bands}")

    def get_model_config(self):
        pass

    def preprocess(self, inputs):

        inputs["input_images"] = inputs["input_images"][:,self.input_bands].permute(0,2,1,3,4).to(self.device)
        inputs["target"] = inputs["target"][:,self.output_bands].permute(0,2,1,3,4).to(self.device)

        return inputs

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, 1, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """
        x = inputs["input_images"] * 2 - 1
        output, _, _ = self.model(x)
        output = output.unsqueeze(1) * 0.5 + 0.5

        inputs["output"] = output

        if self.args.draw_vis:
            benchmark_visualization(inputs, self.args)

            # inputs["output"] = out.cpu()
            # benchmark_visualization(inputs, self.args)
            # # save the output and input images in numpy format
            # input_dir = "/share/hariharan/ck696/allclear/experiments/input.npy"
            # output_dir = "/share/hariharan/ck696/allclear/experiments/output.npy"
            # target_dir = "/share/hariharan/ck696/allclear/experiments/target.npy"
            # mask_dir = "/share/hariharan/ck696/allclear/experiments/mask.npy"
            # import numpy as np
            # np.save(input_dir, input_imgs.cpu().numpy())
            # np.save(output_dir, out.cpu().numpy())
            # np.save(target_dir, target_imgs.cpu().numpy())
            # np.save(mask_dir, masks.cpu().numpy())

            # print(input_imgs.max(), input_imgs.min())
            # print(out.max(), out.min())
            # print(target_imgs.max(), target_imgs.min())
            # print(inputs.keys())
            torch.save(inputs, "/share/hariharan/ck696/allclear/experiments/pmaa_inputs.pth")
            assert 0 == 1

        return {"output": output}

class DiffCR(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        if "ck696" in os.getcwd():
            sys.path.append("/share/hariharan/ck696/allclear/baselines/DiffCR")
        else:
            sys.path.append("/share/hariharan/cloud_removal/allclear/baselines/DiffCR")

        from core.logger import VisualWriter, InfoLogger
        
        import core.util as Util
        from data import define_dataloader
        from models import create_model, define_network, define_loss, define_metric


        opt = self.get_options()

        phase_logger = InfoLogger(opt)
        # phase_writer = VisualWriter(opt, phase_logger)  
        phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

        print('''set networks and dataset''')
        '''set networks and dataset'''
        # phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
        networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

        print('''set metrics, loss, optimizer and  schedulers''')
        ''' set metrics, loss, optimizer and  schedulers '''
        metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
        losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

        self.model = create_model(
            opt = opt,
            networks = networks,
            phase_loader = None,
            val_loader = None,
            losses = losses,
            metrics = metrics,
            logger = phase_logger,
            writer = None
        )

        # Pretrained model
        # params = torch.load("/share/hariharan/ck696/allclear/baselines/DiffCR/pretrained/diffcr_new.pth")
        # Our model trained on new 2K roi dataset
        print(f"Using DiffCR model checkpoint: {args.diff_checkpoint}")
        params = torch.load(args.diff_checkpoint)

        self.model.netG.load_state_dict(params,strict=False)
        self.model.netG.to(self.device)
        self.model.netG.eval()
        self.bands = (3,2,1)

    def get_options(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '-c', '--config', 
            type=str, 
            # default='config/ours_double_encoder_splitcaCond_splitcaUnet_ALLCLEAR.json', 
            default="/share/hariharan/ck696/allclear/baselines/DiffCR/config/ours_double_encoder_splitcaCond_splitcaUnet_ALLCLEAR.json", 
            help='JSON file for configuration'
        )
        
        parser.add_argument(
            '-p', '--phase', 
            type=str, 
            choices=['train', 'test'], 
            help='Run train or test', 
            default='test'
        )
        
        parser.add_argument(
            '-b', '--batch', 
            type=int, 
            default=2, 
            help='Batch size in every GPU'
        )
        
        parser.add_argument(
            '-gpu', '--gpu_ids', 
            type=str, 
            default="0"
        )
        
        parser.add_argument(
            '-d', '--debug', 
            action='store_true'
        )
        
        parser.add_argument(
            '-P', '--port', 
            default='21012', 
            type=str
        )

        import core.praser as Praser
        args, _ = parser.parse_known_args()
        opt = Praser.parse(args)
        
        return opt

    def get_model_config(self):
        pass

    def preprocess(self, inputs):
        inputs["input_images"] = inputs["input_images"].to(self.device).permute(0,2,1,3,4)[:,:,self.bands]
        inputs["target"] = inputs["target"].to(self.device).permute(0,2,1,3,4)[:,:,self.bands]
        return inputs

    def forward(self, inputs):
        """Refer to `prepare_data_multi()`
        Shapes:
            - input_imgs: (B, T, C, H, W)
            - target_imgs: (B, 1, C, H, W)
            - masks: (B, T, H, W)
            - dates: (B, T)
        """
        bs, c, t, h, w = inputs["input_images"].shape
        # Model I/O (bs, c * t, h, w)
        x = inputs["input_images"] * 2 - 1
        # x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bs, c*t, h, w)
        dummy_y = x[:,:3]
        output, visuals = self.model.netG.restoration(x, y_0=dummy_y, sample_num=self.model.sample_num) # default sample_num=8
        output = output.unsqueeze(1) * 0.5 + 0.5
        return {"output": output}
