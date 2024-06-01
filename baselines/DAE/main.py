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
from baselines.DAE.diffusers_src import UNet3DConditionModel

from allclear import visualize_batch


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
    parser.add_argument("--main-sensor", type=str, default="s2_toa", help="Main sensor for the dataset")
    parser.add_argument("--aux-sensors", type=str, nargs="+", help="Auxiliary sensors for the dataset")
    parser.add_argument("--aux-data", type=str, nargs="+",default=["cld_shdw", "dw"], help="Auxiliary data for the dataset")
    parser.add_argument("--target-mode", type=str, default="s2p", choices=["s2p", "s2s"], help="Target mode for the dataset")
    parser.add_argument("--cld-shdw-fpaths", type=str, default="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json", help="Path to cloud shadow masks")
    parser.add_argument("--tx", type=int, default=3, help="Number of images in a sample for the dataset")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--wandb", type=int, default=0, help="0 for no wandb, 1 for wandb")
    parser.add_argument("--mode", type=str, default="test", help="train or test")
    parser.add_argument("--vis-freq", type=int, default=50, help="train or test")
    parser.add_argument("--do-preprocess", action="store_true", help="Preprocess the data before running the model")
    parser.add_argument("--checkpoint", type=str, default=None, help="The checkpoint to load")

    # jupyter notebook artifact
    parser.add_argument("-f", type=str, default="", help="jupyter notebook artifact")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    with open(args.dataset) as f:
        dataset = json.load(f)
    main_sensor = args.main_sensor
    aux_sensors = args.aux_sensors
    aux_data = args.aux_data
    tx = args.tx
    target_mode = args.target_mode

    with open("/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json") as f:
        cld_shdw_fpaths = json.load(f)

    # load in train and val rois
    with open("/share/hariharan/cloud_removal/metadata/v3/train_rois_20k.txt") as f:
        train_rois = f.read().splitlines()
    with open("/share/hariharan/cloud_removal/metadata/v3/val_rois_20k.txt") as f:
        val_rois = f.read().splitlines()

    runname = f"{args.runname}"

    # Set up the data
    train_dataset = CRDataset(dataset=dataset,
                              selected_rois=train_rois,
                              main_sensor=main_sensor,
                              aux_sensors=aux_sensors,
                              aux_data=aux_data,
                              tx=tx,
                              target_mode=target_mode,
                              cld_shdw_fpaths=cld_shdw_fpaths,
                              do_preprocess=args.do_preprocess,)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    val_dataset = CRDataset(dataset=dataset,
                            selected_rois=val_rois,
                            main_sensor=main_sensor,
                            aux_sensors=aux_sensors,
                            aux_data=aux_data,
                            tx=tx,
                            target_mode=target_mode,
                            cld_shdw_fpaths=cld_shdw_fpaths,
                            do_preprocess=args.do_preprocess,)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

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
    if args.checkpoint is not None:
        params = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        filtered_params = {k: v for k, v in params.items() if "custom_pos_embed.position" not in k}
        model.load_state_dict(filtered_params, strict=False)

    # base_bs = 1
    # base_lr = 0.00001
    # lr = base_lr * args.train_bs / base_bs
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    num_training_steps = len(train_dataloader) * args.num_epochs
    warmup_percentage = 0.05
    num_warmup_steps = int(num_training_steps * warmup_percentage) # 512 / args.train_bs * 2

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="wandb" if args.wandb else "tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
        device_placement=True
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

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
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
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, miniters=10)
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {args.epoch}")

        for bid, data in enumerate(train_dataloader):
            model.train()
            update_model_position_token(model, data["time_differences"])
            # print(data['target_cld_shdw'].shape, data['input_images'].shape)
            loss_mask = torch.logical_not((data['target_cld_shdw'][:, 0, ...] + data['target_cld_shdw'][:, 1, ...]) > 0).unsqueeze(1)

            with accelerator.accumulate(model):
                pred = model(data['input_images'], 1, encoder_hidden_states=emb, return_dict=False)[0]
                loss1 = (F.mse_loss(pred, data['target'][:, :args.out_channel], reduction="none") * loss_mask).mean() * 3
                pred = torch.clip(pred, 0, 0.5) / 0.5
                clean_imgs_ = torch.clip(data['target'], 0, 0.5) / 0.5
                loss2 = (F.mse_loss(pred, clean_imgs_[:, :args.out_channel], reduction="none") * loss_mask).mean()
                loss = loss1 + loss2
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "loss1": loss1.detach().item(), "loss2": loss2.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            args.global_step = global_step

            if accelerator.is_main_process and bid % 1000 == 0 and bid > 0:
                # save checkpoint
                os.makedirs(os.path.join(args.output_dir, args.runname, "checkpoints"), exist_ok=True)
                PATH = os.path.join(args.output_dir, args.runname, "checkpoints", f"model_{args.runname}_{args.epoch}_{bid}.pt")
                accelerator.save(model.state_dict(), PATH)
                model.eval()
                total_loss1 = 0
                total_loss2 = 0
                total_loss = 0
                num_samples = 0
                max_samples = max(int(10 / (args.tx / 3)), 1)
                inputs = []
                outputs = []
                outputs_real = []
                targets = []
                loss_masks = []
                sensors = [args.main_sensor] + args.aux_sensors
                timestamps = []
                geolocations = []
                roi_ids = []


                # eval on 10 samples from the validation set
                for bid, data in enumerate(val_dataloader):
                    update_model_position_token(model, data["time_differences"])
                    loss_mask = torch.logical_not((data['target_cld_shdw'][:, 0, ...] + data['target_cld_shdw'][:, 1, ...]) > 0).unsqueeze(1)
                    loss_masks.append(loss_mask)
                    geolocations.append(data['latlong'])
                    timestamps.append(data['timestamps'])
                    roi_ids.extend(data['roi'])
                    targets.append(data['target'][:, :args.out_channel])
                    inputs.append(data['input_images'])
                    with torch.no_grad():
                        pred = model(data['input_images'], 1, encoder_hidden_states=emb, return_dict=False)[0]
                        pred_real = model(torch.cat((data['target'], data['input_images'][:,-2:,...]), dim=1), 1, encoder_hidden_states=emb, return_dict=False)[0]
                        outputs.append(pred)
                        outputs_real.append(pred_real)
                        loss1 = (F.mse_loss(pred, data['target'][:, :args.out_channel],
                                            reduction="none") * loss_mask).mean() * 3
                        pred = torch.clip(pred, 0, 0.5) / 0.5
                        clean_imgs_ = torch.clip(data['target'], 0, 0.5) / 0.5
                        loss2 = (F.mse_loss(pred, clean_imgs_[:, :args.out_channel],
                                            reduction="none") * loss_mask).mean()
                        loss = loss1 + loss2
                        total_loss1 += loss1.item()
                        total_loss2 += loss2.item()
                        total_loss += loss.item()
                        num_samples += 1
                    if num_samples == max_samples:
                        break

                inputs = torch.cat(inputs, dim=0).permute(0, 2, 1, 3, 4)
                outputs = torch.cat(outputs, dim=0).permute(0, 2, 1, 3, 4)
                outputs_real = torch.cat(outputs_real, dim=0).permute(0, 2, 1, 3, 4)
                targets = torch.cat(targets, dim=0).permute(0, 2, 1, 3, 4)
                loss_masks = torch.cat(loss_masks, dim=0).permute(0, 2, 1, 3, 4)
                timestamps = torch.cat(timestamps, dim=0)
                lats = torch.cat([x[0] for x in geolocations], dim=0)
                lons = torch.cat([x[1] for x in geolocations], dim=0)
                geolocations = torch.stack([lats, lons], dim=1)
                metrics = Metrics(outputs=outputs, targets=targets, masks=loss_masks).evaluate_aggregate()

                logs = {"val_loss": total_loss/max_samples,
                        "val_loss1": total_loss1/max_samples,
                        "val_loss2": total_loss2/max_samples,
                        "val_mae": metrics["MAE"],
                        "val_rmse": metrics["RMSE"],
                        "val_psnr": metrics["PSNR"],
                        "val_sam": metrics["SAM"],
                        "val_ssim": metrics["SSIM"],}
                accelerator.log(logs, step=global_step)

                # vis_num = max(int(5 / (args.tx / 3)), 1)
                # vis_data = {"sensors": sensors, "timestamps": timestamps[:vis_num], "geolocations": geolocations[:vis_num], "rois": roi_ids[:vis_num],
                #             "outputs": outputs[:vis_num], "targets": targets[:vis_num], "loss_masks": loss_masks[:vis_num], "inputs": inputs[:vis_num],
                #             "outputs_real": outputs_real[:vis_num]}
                # visualize_batch(vis_data, min_value=0, max_value=1, args=args)
                # visualize_batch(vis_data, min_value=0, max_value=0.1, args=args)
                # visualize_batch(vis_data, min_value=0, max_value=0.3, args=args)
                # visualize_batch(vis_data, min_value=0.3, max_value=1, args=args)

                model.train()


        if accelerator.is_main_process:

            if args.epoch % 1 == 0:
                os.makedirs(os.path.join(args.output_dir, args.runname, "checkpoints"), exist_ok=True)
                PATH = os.path.join(args.output_dir, args.runname, "checkpoints", f"model_{args.runname}_{args.epoch}.pt")
                accelerator.save(model.state_dict(), PATH)
    accelerator.end_training()


# import json
#     try:
#         dirr = "/scratch/allclear/metadata/v3/UnCRtainTS/"
#         with open(os.path.join(dirr, f"s2p_tx{config.input_t}_train_20k_v1.json")) as file:
#             dataset = json.load(file)
#         with open(os.path.join(dirr, "cld30_shdw30_fpaths_train_20k.json")) as file:
#             cld_shdw_fpaths = json.load(file)
#         with open(os.path.join(dirr, f"train_rois_20k_scaling_pc{int(config.scaling_law*100)}.txt")) as file:
#             train_rois = file.read().splitlines()
#         with open(os.path.join(dirr, "val_rois_20k.txt")) as file:
#             val_rois = file.read().splitlines()
#     except:
#         dirr = "/share/hariharan/cloud_removal/metadata/v3/"
#         with open(os.path.join(dirr, f"s2p_tx{config.input_t}_train_20k_v1.json")) as file:
#             dataset = json.load(file)
#         with open(os.path.join(dirr, "cld30_shdw30_fpaths_train_20k.json")) as file:
#             cld_shdw_fpaths = json.load(file)
#         with open(os.path.join(dirr, f"train_rois_20k_scaling_pc{int(config.scaling_law*100)}.txt")) as file:
#             train_rois = file.read().splitlines()
#         with open(os.path.join(dirr, "val_rois_20k.txt")) as file:
#             val_rois = file.read().splitlines()
#
#
# input_t = tx
# scaling_law = 1, 0.1, 0.01