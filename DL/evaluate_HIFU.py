import torch
from torch import nn
from tqdm import tqdm

from ssim import SSIM


@torch.inference_mode()
def evaluate_HIFU(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    epoch_loss = 0
    ssim_val = 0
    criterion = nn.L1Loss()
    ssim_f = SSIM(val_range=2)
    # iterate over the validation set
    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                image, mask_true = batch[0], batch[1]
                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                mask_pred = net(image)  # BCHW
                loss = criterion(mask_pred.squeeze(1), mask_true.squeeze(1))
                epoch_loss += loss.item()
                ssim_val += ssim_f(mask_pred,mask_true)

    net.train()
    return epoch_loss/num_val_batches,ssim_val


def evaluate_HIFU_1D(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    epoch_loss = 0
    ssim_val = 0
    criterion = nn.L1Loss()
    ssim_f = SSIM(val_range=2)
    # iterate over the validation set
    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                image, mask_true = batch[0], batch[1]
                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                for i in range(0, image.shape[2]):
                    images_channel_one = image[:, :, i, :]
                    mask_pred_channel_one = net(images_channel_one)  # BCHW
                    if i == 0:
                        mask_pred = mask_pred_channel_one[:, :, None, :]
                    else:
                        mask_pred = torch.cat((mask_pred, mask_pred_channel_one[:, :, None, :]), dim=2)
                loss = criterion(mask_pred.squeeze(1), mask_true.squeeze(1))
                epoch_loss += loss.item()
                ssim_val += ssim_f(mask_pred,mask_true)

    net.train()
    return epoch_loss/num_val_batches,ssim_val
