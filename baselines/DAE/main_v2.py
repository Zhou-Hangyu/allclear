import os
import json
from tqdm import tqdm
import wandb
import argparse

from torch.utils.data import DataLoader
# from dataset.dataloader_v1 import CRDataset
from allclear import Metrics, CRDataset

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from experimental_scripts.DAE.diffusers_src import UNet3DConditionModel


def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--runname", type=str, default="Cond3D_v46", help="The model name on the HF Hub")
    parser.add_argument("--image-size", type=int, default=256, help="The generated image resolution")
    parser.add_argument("--train-bs", type=int, default=1, help="The training batch size")
    parser.add_argument("--eval-bs", type=int, default=2, help="The evaluation batch size")
    parser.add_argument("--num-epochs", type=int, default=20, help="The number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
    parser.add_argument("--model-blocks", type=str, default="CCCCAA", help="The model blocks")
    parser.add_argument("--block-type", type=str, default="conv", help="The model blocks", choices=["conv", "resnet"])
    parser.add_argument("--LPB", type=int, default=1, help="Layer per block")
    parser.add_argument("--postfix", type=str, default="", help="The model name on the HF Hub")

    parser.add_argument("--in-channel", type=int, default=15, help="number of input channel")
    parser.add_argument("--out-channel", type=int, default=13, help="number of output channel")
    parser.add_argument("--max-time-span", type=int, default=12, help="The number of frame")

    # cross_attention_dim
    parser.add_argument("--norm-num-groups", type=int, default=32, help="The number of group for normalization")
    parser.add_argument("--max-d0", type=int, default=128, help="The maximum dimension")
    parser.add_argument("--max-dim", type=int, default=512, help="The maximum dimension")
    parser.add_argument("--model-type", type=str, default="UNet3D_src", help="The model type",
                        choices=["UNet3D", "UNet3D_src"])

    # Experimental Features
    parser.add_argument("--num-workers", type=int, default=2, help="The number of workers for data loading")
    parser.add_argument("--dataset", type=str, default="/share/hariharan/allclear/metadata/v3/s2s_tx3_v1.json.json",
                        help="The file path of the dataset")
    parser.add_argument("--output-dir", type=str,
                        default="/share/hariharan/cloud_removal/allclear/experimental_scripts/results/ours/dae",
                        help="The output directory")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--wandb", type=int, default=0, help="0 for no wandb, 1 for wandb")
    parser.add_argument("--mode", type=str, default="test", help="train or test")
    parser.add_argument("--vis-freq", type=int, default=50, help="train or test")
    parser.add_argument("--do-preprocess", action="store_true", help="Preprocess the data before running the model")

    # jupyter notebook artifact
    parser.add_argument("-f", type=str, default="", help="jupyter notebook artifact")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    with open(args.dataset) as f:
        dataset = json.load(f)
    selected_rois = "all"
    main_sensor = "s2_toa"
    aux_sensors = ["s1"]
    aux_data = ["cld_shdw"]
    tx = 3
    # target_mode = "s2p"
    target_mode = "s2s"
    with open("/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json") as f:
        cld_shdw_fpaths = json.load(f)

    runname = f"{args.runname}_{args.model_type}_{args.model_blocks}_{args.lr}_{args.norm_num_groups}_{args.max_d0}_{args.max_dim}_{args.postfix}"

    # Set up the data
    train_dataset = CRDataset(dataset=dataset,
                              selected_rois=selected_rois,
                              main_sensor=main_sensor,
                              aux_sensors=aux_sensors,
                              aux_data=aux_data,
                              tx=tx,
                              target_mode=target_mode,
                              cld_shdw_fpaths=cld_shdw_fpaths,
                              do_preprocess=args.do_preprocess, )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    # TODO: add validataion set
    # val_dataset = CRDataset(dataset=dataset,
    #                         selected_rois=selected_rois,
    #                         main_sensor=main_sensor,
    #                         aux_sensors=aux_sensors,
    #                         aux_data=aux_data,
    #                         tx=tx,
    #                         target_mode=target_mode,
    #                         cld_shdw_fpaths=cld_shdw_fpaths,
    #                         do_preprocess=args.do_preprocess,)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=args.num_workers,
    #                             pin_memory=True)

    # Set up the model
    down_block_dict = {"C": "DownBlock3D", "A": "CrossAttnDownBlock3D", "J": "DownBlockJust2D",
                       "R": "CrossAttnDownBlock2D1D"}
    up_block_dict = {"C": "UpBlock3D", "A": "CrossAttnUpBlock3D", "J": "UpBlockJust2D", "R": "CrossAttnUpBlock2D1D"}
    down_block_list = [down_block_dict[b] for b in args.model_blocks]
    up_block_list = [up_block_dict[b] for b in args.model_blocks][::-1]
    model = UNet3DConditionModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=args.in_channel,  # the number of input channels, 3 for RGB images
        out_channels=args.out_channel,  # the number of output channels
        layers_per_block=args.LPB,  # how many ResNet layers to use per UNet block
        block_out_channels=(args.max_d0, 128, 256, args.max_dim, args.max_dim, args.max_dim, args.max_dim)[
                           :len(args.model_blocks)],  # the number of output channels for each UNet block
        down_block_types=down_block_list,  # the down block sequence
        up_block_types=up_block_list,  # the up block sequence
        norm_num_groups=args.norm_num_groups,  # the number of groups for normalization
    )

    base_bs = 1
    base_lr = 0.00001
    lr = base_lr * args.train_bs / base_bs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    num_training_steps = len(train_dataloader) * args.num_epochs
    warmup_percentage = 0.05
    num_warmup_steps = int(num_training_steps * warmup_percentage)  # 512 / args.train_bs * 2

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="wandb" if args.wandb else "tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
        device_placement=True,
    )
    accelerator.init_trackers(
        project_name="allclear-dae-v1",
        config={"model": model.__class__.__name__,
                "lr": args.lr,
                "architecture": f"{args.model_type}-{args.model_blocks}",
                "dataset": args.dataset.split("/")[-1],
                "main_sensor": main_sensor,
                "aux_sensors": aux_sensors,
                "aux_data": aux_data,
                "tx": tx,
                "target_mode": target_mode,
                "epochs": args.num_epochs, },
        init_kwargs={"wandb": {"entity": "cornell-kao", "name": runname, }}
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        # accelerator.init_trackers("train_example")

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )
    args.device = model.device

    def update_model_position_token(model, token):
        for p1, p2 in model.named_parameters():
            if 'position' in p1:
                p2.data = token

    global_step = 0
    next_batch_flag = False
    emb = torch.zeros((args.train_bs, 2, 1024)).to(model.device)
    for args.epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {args.epoch}")

        for bid, data in enumerate(train_dataloader):
            model.train()
            update_model_position_token(model, data["time_differences"])
            loss_mask = torch.logical_not(
                (data['target_cld_shdw'][:, 0, ...] + data['target_cld_shdw'][:, 1, ...]) > 0).unsqueeze(1)

            with accelerator.accumulate(model):
                pred = model(data['input_images'], 1, encoder_hidden_states=emb, return_dict=False)[0]
                loss1 = (F.mse_loss(pred, data['target'][:, :args.out_channel],
                                    reduction="none") * loss_mask).mean() * 3
                pred = torch.clip(pred, -1, -0.6) / 0.4
                clean_imgs_ = torch.clip(data['target'], -1, -0.6) / 0.4
                loss2 = (F.mse_loss(pred, clean_imgs_[:, :args.out_channel], reduction="none") * loss_mask).mean()
                loss = loss1 + loss2
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "loss1": loss1.detach().item(),
                    "loss2": loss2.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            if accelerator.is_main_process and bid % 1000 == 0 and bid > 0:
                # save checkpoint
                PATH = os.path.join(args.output_dir, f"model_{args.runname}_{args.epoch}_{bid}.pt")
                accelerator.save(model.state_dict(), PATH)
                model.eval()
            model.train()

        progress_bar.close()

        if accelerator.is_main_process and args.epoch % 1 == 0:
            # Save the model checkpoint using torch save
            PATH = os.path.join(args.output_dir, f"model_{args.runname}_{args.epoch}.pt")
            accelerator.save(model.state_dict(), PATH)
    accelerator.end_training()


if __name__ == "__main__":
    main()
