import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ssim import SSIM


@torch.inference_mode()
def evaluate_RRDB(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    epoch_loss = 0
    ssim_val = 0
    criterion = nn.L1Loss()
    ssim_f = SSIM()
    # iterate over the validation set
    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                image, mask_true = batch[0], batch[1]
                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                mask_pred = net(image)  # BCHW
                loss = criterion(mask_pred.squeeze(1), image.squeeze(1))
                epoch_loss += loss.item()
                ssim_val += ssim_f(mask_pred,mask_true)

    net.train()
    return epoch_loss/num_val_batches,ssim_val
