import argparse
from datetime import datetime
import os
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import HfFolder, Repository, whoami
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision.transforms import GaussianBlur
from diffusers_src import UNet3DConditionModel

from dataloader_v46 import CogDataset_v46

# Use logging to logging.info out the model summary
import logging
logging.basicConfig(level=logging.INFO)

# version 44
# 1. Use auto scale for visualization
# 2. Clip the range of the prediction to -1 to -0.6 for loss calculation
# version 45
# 1. Use new dataset
# version 46
# 1. Full bands in-out
# version 47
# 1. Add masked as 0 or 1


# Define argument parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument("--runname", type=str, default="Cond3D_v46", help="The model name on the HF Hub")
parser.add_argument("--image_size", type=int, default=256, help="The generated image resolution")
parser.add_argument("--train_bs", type=int, default=1, help="The training batch size")
parser.add_argument("--eval_bs", type=int, default=2, help="The evaluation batch size")
parser.add_argument("--num_epochs", type=int, default=20, help="The number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
parser.add_argument("--model_blocks", type=str, default="CCCCAA", help="The model blocks")
parser.add_argument("--block_type", type=str, default="conv", help="The model blocks", choices=["conv", "resnet"])
parser.add_argument("--LPB", type=int, default=1, help="Layer per block")
parser.add_argument("--postfix", type=str, default="", help="The model name on the HF Hub")

parser.add_argument("--in_channel", type=int, default=15, help="number of input channel")
parser.add_argument("--out_channel", type=int, default=13, help="number of output channel")
parser.add_argument("--max_time_span", type=int, default=12, help="The number of frame")

# cross_attention_dim
parser.add_argument("--norm_num_groups", type=int, default=32, help="The number of group for normalization")
parser.add_argument("--max_d0", type=int, default=128, help="The maximum dimension")
parser.add_argument("--max_dim", type=int, default=512, help="The maximum dimension")
parser.add_argument("--model_type", type=str, default="UNet3D_src", help="The model type", choices=["UNet3D", "UNet3D_src"])
                    
# Experimental Features
parser.add_argument("--num_workers", type=int, default=2, help="The number of workers for data loading")

# Reproducibility
parser.add_argument("--seed", type=int, default=0, help="The random seed")
parser.add_argument("--wandb", type=int, default=0, help="0 for no wandb, 1 for wandb")
parser.add_argument("--push_to_hub", type=int, default=0, help="0 for no push to hub, 1 for push to hub")
parser.add_argument("--mode", type=str, default="test", help="train or test")
parser.add_argument("--vis_freq", type=int, default=50, help="train or test")
args, _ = parser.parse_known_args()
logging.info(args)


args.runname += f"_0429_I{args.in_channel}O{args.out_channel}T{args.max_time_span}-Blc{args.model_blocks}"
args.runname += f"-LR{args.lr}-LPB{args.LPB}"
if args.norm_num_groups != 32:
    args.runname += f"-GNorm{args.norm_num_groups}"
if args.max_dim != 1024:
    args.runname += f"-MaxDim{args.max_dim}"
if args.max_d0 != 128:
    args.runname += f"-MaxD0{args.max_d0}"
if args.postfix != "":
    args.runname += f"-{args.postfix}"
# args.runname += f"-Seed{args.seed}"
args.runname = args.runname.replace(".", "").replace("-", "_")
args.output_dir = "./results/" + args.runname

if args.mode == "test":
    args.push_to_hub = 0
    args.wandb = 0
    args.num_workers = 0
    args.vis_freq = 1
    args.max_time_span = 13
    args.LPB = 1
    args.model_blocks = "CC"
else:
    num_workers = args.num_workers

# Set up the dataset and dataloader
train_dataset = CogDataset_v46(max_num_frames=args.max_time_span, image_size=args.image_size, mode="train")
train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dataset = CogDataset_v46(max_num_frames=args.max_time_span, image_size=args.image_size, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

down_block_dict = {"C": "DownBlock3D", "A": "CrossAttnDownBlock3D", "J": "DownBlockJust2D", "R": "CrossAttnDownBlock2D1D"}
up_block_dict = {"C": "UpBlock3D", "A": "CrossAttnUpBlock3D", "J": "UpBlockJust2D", "R": "CrossAttnUpBlock2D1D"}
down_block_list = [down_block_dict[b] for b in args.model_blocks]
up_block_list = [up_block_dict[b] for b in args.model_blocks][::-1]

    

# Set up the model
model = UNet3DConditionModel(
    sample_size=args.image_size,  # the target image resolution
    in_channels=args.in_channel,  # the number of input channels, 3 for RGB images
    out_channels=args.out_channel,  # the number of output channels
    layers_per_block=args.LPB,  # how many ResNet layers to use per UNet block
    block_out_channels=(args.max_d0, 128, 256, args.max_dim, args.max_dim, args.max_dim, args.max_dim)[:len(args.model_blocks)],  # the number of output channels for each UNet block
    down_block_types=down_block_list,  # the down block sequence
    up_block_types=up_block_list,  # the up block sequence
    norm_num_groups=args.norm_num_groups,  # the number of groups for normalization
)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=512 / args.train_bs * 2,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)

# Set up the accelerator
def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

accelerator = Accelerator(
    mixed_precision="fp16",
    log_with="wandb" if args.wandb else "tensorboard",
    project_dir=os.path.join(args.output_dir, "logs"),
)
accelerator.init_trackers(
    project_name="NoCloud-UNet", 
    config={"model": model.__class__.__name__,},
    init_kwargs={"wandb": {"entity": "cornell-kao", 
                           "name":args.runname, 
                           "config": args,
                           }}
    )
if accelerator.is_main_process:
    if args.push_to_hub:
        repo_name = get_full_repo_name(Path(args.output_dir).name)
        repo = Repository(args.output_dir, clone_from=repo_name)
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.init_trackers("train_example")

# Prepare everything
model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
    model, optimizer, lr_scheduler, train_dataloader
)
args.device = model.device

def update_model_position_token(model, token):
    for p1, p2 in model.named_parameters():
        if 'position' in p1:
            p2.data = token
            # print(p1, "set position")

            
            
            
def pp(img): return img.permute(1,2,0)
def pp2(img, scale=0.3): 
    img = img / scale
    img = torch.clip(img, 0, 1)
    img = img.numpy()
    return img 

def visualization_v44(args, x1, p1, c1, x2, p2, scale=0.3, auto=False):
    
    x1 = x1.cpu()
    p1 = p1.cpu()
    c1 = c1.cpu()
    x2 = x2.cpu()
    p2 = p2.cpu()

    for batch_idx in range(x1.size(0)):
        
        daycount = daycounts[batch_idx].long()
        daycount = daycount - daycount[0]

        fig, axs = plt.subplots(6, args.max_time_span, figsize=(args.max_time_span*2, 12))

        if auto:
            temp = np.percentile(x1[batch_idx,1:4].cpu().numpy(), 98, axis=(0,2,3))
            temp = np.percentile(temp, 20)
            scale = min(temp, scale)
            print(f"Scale: {scale}")
        
        for frame_idx in range(args.max_time_span):
            
            img = pp(x1[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:,:,::-1]
            axs[0, frame_idx].imshow(img)
            if frame_idx == 0: axs[0, frame_idx].set_ylabel("Input \n MSI", size=14)
            
            img = pp(x1[batch_idx, 12:15, frame_idx])
            img = pp2(img, 1)
            img[:,:,0] = img[:,:,1]
            axs[1, frame_idx].imshow(img)
            if frame_idx == 0: axs[1, frame_idx].set_ylabel("Input \n SAR", size=14)

            img = pp(p1[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:,:,::-1]
            axs[2, frame_idx].imshow(img)
            if frame_idx == 0: axs[2, frame_idx].set_ylabel("Output \n RGB", size=14)

            img = pp(c1[batch_idx, 0:1, frame_idx])
            img = 1 - img.cpu().numpy()
            axs[3, frame_idx].imshow(img, vmin=0, vmax=1, cmap="gray")
            if frame_idx == 0: axs[3, frame_idx].set_ylabel("Mask \n Cloud + Shadow", size=14)

            img = pp(x2[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:,:,::-1]
            axs[4, frame_idx].imshow(img)
            if frame_idx == 0: axs[4, frame_idx].set_ylabel("Input \n Noisy MSI", size=14)

            img = pp(p2[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:,:,::-1]
            axs[5, frame_idx].imshow(img)
            if frame_idx == 0: axs[5, frame_idx].set_ylabel("Output \n Prediction", size=14)
            
            axs[0, frame_idx].set_title(f"Day {daycount[frame_idx]}", size=14)
            
        # remove the tickes
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"EP{args.epoch}_S{args.step}_B{batch_idx}_Scale{int(scale*100)}.png"))
        plt.pause(0.2)
        plt.close()

# class Cloud_erosion_and_dilation(nn.Module):
class CloudErosionDilation(nn.Module):

    def __init__(self, args):
        super(CloudErosionDilation, self).__init__()
        self.mask_dilation_kernel = 7
        self.device = args.device
        self.blur_kernel = GaussianBlur(kernel_size=3, sigma=1).to(self.device)

    def forward(self, cloud):
        cloud = cloud.to(self.device)
        cloud = -torch.nn.functional.max_pool2d(cloud*-1., kernel_size=3, stride=1, padding=1)
        cloud =  self.blur_kernel(cloud)
        cloud =  torch.nn.functional.max_pool2d(cloud    , kernel_size=self.mask_dilation_kernel, stride=1, padding=self.mask_dilation_kernel//2)
        cloud =  self.blur_kernel(cloud)
        return cloud.to(self.device)
    
cloud_process = CloudErosionDilation(args).to(args.device)
    
# By random shuffling the clouds across time and batch to sample the clouds
class RandomCloudGenerator(nn.Module):
    def __init__(self, args):
        super(RandomCloudGenerator, self).__init__()
        self.device = args.device

    def forward(self, cloud):

        # Randomly shuffle the clouds across time and batch
        randperm = torch.randperm(cloud.shape[0])
        randperm_time = torch.randperm(cloud.shape[2])
        cloud = cloud[randperm]
        cloud = cloud[:, :, randperm_time]

        # Horizontal flip, vertical flip, and rotation by 90*n degrees
        if torch.rand(1) > 0.5:
            cloud = torch.flip(cloud, [3])
        if torch.rand(1) > 0.5:
            cloud = torch.flip(cloud, [4])
        if torch.rand(1) > 0.5:
            cloud = torch.rot90(cloud, 1, [3, 4])

        return cloud
    
class SquareMaskGenerator(nn.Module):
    def __init__(self, args):
        super(SquareMaskGenerator, self).__init__()
        # self.shape = (args.train_bs, args.time_span, args.image_size, args.image_size)
        self.device = args.device
            
    def forward(self, args):
        
        with torch.no_grad():
            mask_scale = np.random.choice([16, 32, 64])
            threshold  = np.random.uniform(low=0.1, high=0.25)
            mask = (torch.rand((args.train_bs, args.time_span, args.image_size, args.image_size)) < threshold / (mask_scale*2) ** 2).float().to(self.device)
            mask = F.max_pool2d(mask, mask_scale+1, stride=1, padding=int((mask_scale+1)/2))
            mask = mask.unsqueeze(1).expand(-1, 13, -1, -1, -1)
        return mask

class NoisyProcess(nn.Module):
    def __init__(self, args):
        super(NoisyProcess, self).__init__()
        self.args = args
        self.device = args.device
        self.sauare_mask_generator  = SquareMaskGenerator(args).to(self.device)
        self.random_cloud_generator = RandomCloudGenerator(args).to(self.device)

    def forward(self, clean_imgs):
        
        # cloud
        real_cloud = clean_imgs[:, 15:16]
        cloud_mask_synthesis = self.sauare_mask_generator(args).to(self.device) * random_scale(args) * 0.5
        cloud_mask_resampled = self.random_cloud_generator(real_cloud).to(self.device) * random_scale(args) * 0.5
        cloud_mask = torch.max(cloud_mask_synthesis, cloud_mask_resampled)
        
        threshold = 0.3
        clean_imgs_cloud_mask = (torch.clip(raw_data[:,15:16], threshold, 1) - threshold) / (1 - threshold)
        clean_imgs_cloud = clean_imgs[:, :13] * clean_imgs_cloud_mask
        randperm = torch.randperm(clean_imgs_cloud.shape[2])
        perm_clean_imgs_cloud_mask = clean_imgs_cloud_mask[:,:,randperm]
        perm_clean_imgs_cloud = clean_imgs_cloud[:,:,randperm]
        interpolation_mask = perm_clean_imgs_cloud_mask * (1-clean_imgs_cloud_mask)
                
        # shadow
        real_shadow = clean_imgs[:, 15:16]
        shadow_mask_synthesis = self.sauare_mask_generator(args).to(self.device) * random_scale(args) * 0.5
        shadow_mask_resampled = self.random_cloud_generator(real_shadow).to(self.device) * random_scale(args) * 0.5
        shadow_mask = torch.max(shadow_mask_synthesis, shadow_mask_resampled)

        noisy_imgs = copy.deepcopy(clean_imgs)
        noisy_imgs[:, :13] *= (1 - shadow_mask * (random_scale(args) * 0.5 + 0.4))
        noisy_imgs[:, :13] = noisy_imgs[:, :13] * (1-interpolation_mask) + perm_clean_imgs_cloud[:, :13] * interpolation_mask
        noisy_imgs[:, :13] += cloud_mask 

        if args.time_span >= 8:
            batch_temporal_cloud_mask = torch.mean(cloud_mask, dim=[1, 3, 4], keepdim=True) > 0.95
            if batch_temporal_cloud_mask.sum() < args.time_span // 2:
                random_batch_temporal_cloud_mask = (torch.rand([args.train_bs, 1, args.time_span, 1, 1]).to(args.device) > 0.75) * 1.
                if torch.rand(1) > 0.5:
                    noisy_imgs[:,:13] = noisy_imgs[:,:13] * (1 - random_batch_temporal_cloud_mask) + torch.clip(random_batch_temporal_cloud_mask * random_scale(args) * 2, 0, 1)
                else:
                    noisy_imgs[:,:13] = noisy_imgs[:,:13] * (1 - random_batch_temporal_cloud_mask)

        return noisy_imgs

def random_scale(args):
    return torch.rand([args.train_bs, 1, args.time_span, 1, 1]).to(args.device)

noisy_process = NoisyProcess(args)

                    
                    
global_step = 0
next_batch_flag = False
emb = torch.zeros((args.train_bs, 2, 1024)).to(model.device)
for args.epoch in range(args.num_epochs):

    
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, miniters=10)
    progress_bar.set_description(f"Epoch {args.epoch}")

    for args.step, (raw_data, meta_info, daycounts, dates) in enumerate(train_dataloader):
        
        model.train()

        args.time_span = raw_data.size(2)

        # msi_buffer contains 17 bands. 0-9 bands are MSI, 10-12 bands are SAR, 13-14 bands are Cloud and Shadow masks
        with torch.no_grad():
            raw_data = raw_data.to(args.device)
            cloud_mask = cloud_process(raw_data[:, 16:18].sum(dim=1, keepdim=False) >= 1)
            loss_mask = 1 - cloud_mask.unsqueeze(1)
            clean_imgs = raw_data[:, :15]
            noisy_imgs = noisy_process(raw_data)[:, :15]

            clean_imgs_ = clean_imgs * 2 - 1
            noisy_imgs_ = noisy_imgs * 2 - 1

        update_model_position_token(model, daycounts - daycounts.min())

        with accelerator.accumulate(model):

            pred = model(noisy_imgs_, 1, encoder_hidden_states=emb, return_dict=False)[0]

            loss1 = (F.mse_loss(pred, clean_imgs_[:, :args.out_channel], reduction="none") * loss_mask).mean() * 3 # 1, loss2 times scaler

            # normalized to -1, 1
            # focus on 0-3000 range (mis
            pred = torch.clip(pred, -1, -0.6) / 0.4 # return to -1, 1. match the value range
            clean_imgs_ = torch.clip(clean_imgs_, -1, -0.6) / 0.4
            loss2 = (F.mse_loss(pred, clean_imgs_[:, :args.out_channel], reduction="none") * loss_mask).mean()

            loss = loss1 + loss2

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1


        if accelerator.is_main_process and args.step % 100 == 0:
            model.eval()
            for _, (raw_data, meta_info, daycounts, dates) in enumerate(test_dataloader): break
            with torch.no_grad():
                raw_data = raw_data.to(args.device)
                args.time_span = raw_data.size(2)
                cloud_mask = cloud_process(raw_data[:, 16:18].sum(dim=1, keepdim=False) >= 1)
                loss_mask = 1 - cloud_mask.unsqueeze(1)
                clean_imgs = raw_data[:, :15]
                noisy_imgs = noisy_process(raw_data)[:, :15]
                clean_imgs_ = clean_imgs * 2 - 1
                noisy_imgs_ = noisy_imgs * 2 - 1
                update_model_position_token(model, daycounts - daycounts.min())

                clean_pred = model(clean_imgs_, 1, encoder_hidden_states=emb, return_dict=False)[0] * 0.5 + 0.5
                noisy_pred = model(noisy_imgs_, 1, encoder_hidden_states=emb, return_dict=False)[0] * 0.5 + 0.5
                visualization_v44(args, clean_imgs, clean_pred, loss_mask, noisy_imgs, noisy_pred, scale=0.3, auto=False)
                visualization_v44(args, clean_imgs, clean_pred, loss_mask, noisy_imgs, noisy_pred, scale=0.2, auto=False)
                visualization_v44(args, clean_imgs, clean_pred, loss_mask, noisy_imgs, noisy_pred, scale=1.0, auto=True)
        model.train()

    if accelerator.is_main_process:
            
        if args.epoch % 1 == 0:
            # Save the model checkpoint using torch save
            PATH = os.path.join(args.output_dir, f"model_{args.epoch}.pt")
            accelerator.save(model.state_dict(), PATH)
            # PATH = os.path.join(args.output_dir, f"embed_{args.epoch}.pt")
            # accelerator.save(meta_embedding.state_dict(), PATH)