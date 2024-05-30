import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def pp(img): return img.permute(1,2,0)
def pp2(img, scale=0.3):
    img = img / scale
    img = torch.clip(img, 0, 1)
    img = img.numpy()
    return img

def visualization_v44(args, x1, p1, c1, x2, p2, daycounts, scale=0.3, auto=False):
    x1 = x1.cpu()
    p1 = p1.cpu()
    c1 = c1.cpu()
    x2 = x2.cpu()
    p2 = p2.cpu()

    for batch_idx in range(x1.size(0)):

        daycount = daycounts[batch_idx].long()
        daycount = daycount - daycount[0]

        fig, axs = plt.subplots(6, args.max_time_span, figsize=(args.max_time_span * 2, 12))

        if auto:
            temp = np.percentile(x1[batch_idx, 1:4].cpu().numpy(), 98, axis=(0, 2, 3))
            temp = np.percentile(temp, 20)
            scale = min(temp, scale)
            print(f"Scale: {scale}")

        for frame_idx in range(args.max_time_span):

            img = pp(x1[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:, :, ::-1]
            axs[0, frame_idx].imshow(img)
            if frame_idx == 0: axs[0, frame_idx].set_ylabel("Input \n MSI", size=14)

            img = pp(x1[batch_idx, 12:15, frame_idx])
            img = pp2(img, 1)
            img[:, :, 0] = img[:, :, 1]
            axs[1, frame_idx].imshow(img)
            if frame_idx == 0: axs[1, frame_idx].set_ylabel("Input \n SAR", size=14)

            img = pp(p1[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:, :, ::-1]
            axs[2, frame_idx].imshow(img)
            if frame_idx == 0: axs[2, frame_idx].set_ylabel("Output \n RGB", size=14)

            img = pp(c1[batch_idx, 0:1, frame_idx])
            img = 1 - img.cpu().numpy()
            axs[3, frame_idx].imshow(img, vmin=0, vmax=1, cmap="gray")
            if frame_idx == 0: axs[3, frame_idx].set_ylabel("Mask \n Cloud + Shadow", size=14)

            img = pp(x2[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:, :, ::-1]
            axs[4, frame_idx].imshow(img)
            if frame_idx == 0: axs[4, frame_idx].set_ylabel("Input \n Noisy MSI", size=14)

            img = pp(p2[batch_idx, 1:4, frame_idx])
            img = pp2(img, scale=scale)[:, :, ::-1]
            axs[5, frame_idx].imshow(img)
            if frame_idx == 0: axs[5, frame_idx].set_ylabel("Output \n Prediction", size=14)

            axs[0, frame_idx].set_title(f"Day {daycount[frame_idx]}", size=14)

        # remove the tickes
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"EP{args.epoch}_S{args.step}_B{batch_idx}_Scale{int(scale * 100)}.png"))
        plt.pause(0.2)
        plt.close()