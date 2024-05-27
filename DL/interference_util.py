import argparse
import logging
import os
import time

import numpy as np
import torch
import scipy.io as sio
import torchvision

from torchvision import transforms

from HIFU_Diff import Diffuser
from dataload import HF_n_pre, HF_n_PreWithoutLabel
from init_hparams import load_predict_hparams, hparams, init_dataset, init_model
from ssim import SSIM

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image


def init_device():
  if torch.cuda.is_available():
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置device
    torch.cuda.set_device(local_rank)
    # 用nccl后端初始化多进程，一般都用这个
    dist.init_process_group(backend='nccl')
    # 获取device，之后的模型和张量都.to(device)
    device = torch.device("cuda", local_rank)
    logging.info(f'available gpu nums: {torch.cuda.device_count()}')
  else:
    device = 'cpu'

  return device


def load_cond(path):
  RF_cond = sio.loadmat(path)['Data']
  cond = np.array(RF_cond, dtype=np.float32)
  cond = torch.from_numpy(cond)
  cond = cond[None, None, :, :].permute(0, 1, 3, 2)
  return cond

def load_image(path):
  image = Image.open(path)
  transform = torchvision.transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
  ]
  )
  image = transform(image)
  return image

def load_rf(path, isLabel):
  if isLabel:
    RF_cond = sio.loadmat(path)['Label']
  else:
    RF_cond = sio.loadmat(path)['Data']
  cond = np.array(RF_cond, dtype=np.float32)
  cond = torch.from_numpy(cond)
  cond = cond / torch.max(torch.abs(cond))
  cond = cond[None, None, :, :].permute(0, 1, 3, 2)
  # cond = torch.concat([cond[:, :, :, (i * 128):(i * 128 + 128)] for i in range(23)], dim=1)
  return cond


def denoise(DP, diffuser, path_label, path_input, device, n,use_ddim=False,ts_ddim=20,eta_ddim=0.5):
  ssim_f = SSIM()
  label = load_rf(path_label, True)
  input = load_rf(path_input, False)
  input = input.to(device=device, dtype=torch.float32)
  label = label.to(dtype=torch.float32)
  with torch.no_grad():
    if use_ddim:
      noise_pre = diffuser.sample_ddim(DP, input, n, ddim_step=ts_ddim,eta=eta_ddim)
    else:
      noise_pre = diffuser.sample(DP, input, n)
    output = (input + noise_pre).cpu()
    if n != 1:
      label = label.repeat(n, 1, 1, 1)
    idx_ssim = ssim_f(output, label)
    logging.info(f'ssim value = {idx_ssim}')
    # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
    output = output.squeeze(1).numpy()
    output = output.transpose()
    sio.savemat('../store_pre/output1.mat', {'predict': output})


def denoise_MA(DP, diffuser, path_label, path_input, device, n, num_angle,use_ddim=False,ts_ddim=20,eta_ddim=0.5):
  ssim_f = SSIM()
  res = []
  sum_ssim = 0
  for i in range(1, num_angle + 1):
    logging.info(f'angle {i} begin:')
    if i == 1:
      path_input = path_input + '_' + str(i)
      path_label = path_label + '_' + str(i)
    else:
      if i <= 10:
        path_label = path_label[:-1] + str(i)
        path_input = path_input[:-1] + str(i)
      else:
        path_label = path_label[:-2] + str(i)
        path_input = path_input[:-2] + str(i)
    label = load_rf(path_label, True)
    input = load_rf(path_input, False)
    input = input.to(device=device, dtype=torch.float32)
    label = label.to(dtype=torch.float32)
    with torch.no_grad():
      # start = time.time()
      if use_ddim:
        noise_pre = diffuser.sample_ddim(DP, input, n, ddim_step=ts_ddim, eta=eta_ddim)
      else:
        noise_pre = diffuser.sample(DP, input, n)
      # torch.cuda.synchronize()
      # end = time.time()
      # total_time = end - start
      # print('total_time:{:.2f}'.format(total_time))
      output = (input + noise_pre).cpu()
      if n != 1:
        label = label.repeat(n, 1, 1, 1)
      idx_ssim = ssim_f(output, label)
      sum_ssim = sum_ssim + idx_ssim
      logging.info(f'ssim value = {idx_ssim}')
      # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
      output = output.squeeze(1).numpy()
      output = output.transpose()
      if i == 1:
        res = output
      else:
        res = np.concatenate([res, output], axis=2)
  logging.info(f'{num_angle} angle average ssim value = {sum_ssim / num_angle}')
  sio.savemat('../store_pre/output1.mat', {'predict': res})



def denoise_dataset(net, diffuser, save_path, dataloader, epoches, device,use_ddim=False,ts_ddim=20,eta_ddim=0.5):
  num_val_batches = len(dataloader)
  ssim_val = 0
  ssim_f = SSIM()
  with torch.no_grad():
    for epoch in range(epoches):
      logging.info(f'epoch {epoch + 1} begin')
      for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        input, label, name = batch[0], batch[1], batch[2]
        # move inputs and labels to correct device and type
        input = input.to(device=device)
        # label = label.to(device=device)
        if use_ddim:
          noise_pre = diffuser.sample_ddim(net, input, 1, ddim_step=ts_ddim, eta=eta_ddim,disable=True)
        else:
          noise_pre = diffuser.sample(net, input, 1, True)
        output = (input + noise_pre).cpu()
        idx_ssim = ssim_f(output, label)
        logging.info(f'{name[0]} = {idx_ssim}')
        ssim_val += idx_ssim
        ff = name[0]
        output = output.squeeze(1).numpy()
        output = output.transpose()
        sio.savemat(save_path + ff + '.mat', {'predict': output})
    logging.info(f'epoch mean ssim = {ssim_val / num_val_batches}')

def denoise_dataset_mg(net, diffuser, save_path, dataloader, epoches, device, val_sampler,use_ddim=False,ts_ddim=20,eta_ddim=0.5):
  num_val_batches = len(dataloader)
  ssim_val = 0
  ssim_f = SSIM()
  world_size = torch.distributed.get_world_size()
  file_path = save_path + 'output1.txt'
  local_rank = int(os.environ["LOCAL_RANK"])
  net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
  with open(file_path, 'w') as file:
    with torch.no_grad():
      for epoch in range(epoches):
        val_sampler.set_epoch(epoch)
        logging.info(f'epoch {epoch + 1} begin')
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
          input, label, name = batch[0], batch[1], batch[2]
          # move inputs and labels to correct device and type
          input = input.to(device=device)
          # label = label.to(device=device)
          if use_ddim:
            noise_pre = diffuser.sample_ddim(net, input, 1, ddim_step=ts_ddim, eta=eta_ddim,disable=True)
          else:
            noise_pre = diffuser.sample(net, input, 1, True)
          output = (input + noise_pre).cpu()
          idx_ssim = ssim_f(output, label).to(device)
          msg = f'{name[0]} ssim = {idx_ssim}\n'
          file.write(msg)
          logging.info(f'{name[0]} = {idx_ssim}')
          ssim_val += idx_ssim
          ff = name[0]
          output = output.squeeze(1).numpy()
          output = output.transpose()
          if epoches == 1:
            sio.savemat(save_path + ff + '.mat', {'predict': output})
          else:
            path_t = save_path + str(epoch + 1) + '/'
            if not os.path.exists(path_t):
              os.makedirs(path_t)
            sio.savemat(path_t + ff + '.mat', {'predict': output})
        dist.all_reduce(ssim_val, op=dist.ReduceOp.SUM)
      ff = f'epoch {epoch} mean ssim = {ssim_val / num_val_batches / world_size}'
      file.write(ff)
      logging.info(f'epoch mean ssim = {ssim_val / num_val_batches / world_size}')

def denoise_datasetWithoutlabel(DP, diffuser, path_input, path_save, device):
  files = os.listdir(path_input)
  logging.info(f'the length of file is : {len(files)}')
  for file in files:
    path_t = os.path.join(path_input, file)
    path_s = os.path.join(path_save, file)
    input = load_rf(path_t, False)
    input = input.to(device=device, dtype=torch.float32)
    with torch.no_grad():
      noise_pre = diffuser.sample(DP, input, 1)
      output = (input + noise_pre).cpu()
      # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
      output = output.squeeze(1).numpy()
      output = output.transpose()
      sio.savemat(path_s, {'predict': output})

def denoise_datasetWithoutlabel_mg(net, diffuser, save_path, dataloader, epoches, device, val_sampler):
  num_val_batches = len(dataloader)
  local_rank = int(os.environ["LOCAL_RANK"])
  net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
  with torch.no_grad():
    for epoch in range(epoches):
      val_sampler.set_epoch(epoch)
      logging.info(f'epoch {epoch + 1} begin')
      for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        input, name = batch[0], batch[1]
        # move inputs and labels to correct device and type
        input = input.to(device=device)
        # label = label.to(device=device)
        noise_pre = diffuser.sample(net, input, 1, True)
        output = (input + noise_pre).cpu()
        ff = name[0]
        output = output.squeeze(1).numpy()
        output = output.transpose()
        if epoches == 1:
          sio.savemat(save_path + ff + '.mat', {'predict': output})
        else:
          path_t = save_path + str(epoch + 1) + '/'
          if not os.path.exists(path_t):
            os.makedirs(path_t)
          sio.savemat(path_t + ff + '.mat', {'predict': output})

def predict_conv(net, path_label, path_input, device):
  label = load_rf(path_label, True)
  input = load_rf(path_input, False)
  input = input.to(device=device, dtype=torch.float32)
  label = label.to(dtype=torch.float32)
  ssim_f = SSIM()
  with torch.no_grad():
    output = net(input).cpu()
    # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
    idx_ssim = ssim_f(output, label)
    logging.info(f'ssim value = {idx_ssim}')
    output = output.squeeze(1).numpy()
    output = output.transpose()
    sio.savemat('../store_pre/output1.mat', {'predict': output})

def predict_conv_MA(net, path_label, path_input, device, num_angle):
  ssim_f = SSIM()
  res = []
  sum_ssim = 0
  for i in range(1, num_angle + 1):
    logging.info(f'angle {i} begin:')
    if i == 1:
      path_input = path_input + '_' + str(i)
      path_label = path_label + '_' + str(i)
    else:
      if i <= 10:
        path_label = path_label[:-1] + str(i)
        path_input = path_input[:-1] + str(i)
      else:
        path_label = path_label[:-2] + str(i)
        path_input = path_input[:-2] + str(i)
    label = load_rf(path_label, True)
    input = load_rf(path_input, False)
    input = input.to(device=device, dtype=torch.float32)
    label = label.to(dtype=torch.float32)
    with torch.no_grad():
      torch.cuda.synchronize()
      output = net(input).cpu()
      torch.cuda.synchronize()
      # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
      idx_ssim = ssim_f(output, label)
      sum_ssim = sum_ssim + idx_ssim
      logging.info(f'ssim value = {idx_ssim}')
      # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
      output = output.squeeze(1).numpy()
      output = output.transpose()
      if i == 1:
        res = output
      else:
        res = np.concatenate([res, output], axis=2)
  logging.info(f'{num_angle} angle average ssim value = {sum_ssim / num_angle}')
  sio.savemat('../store_pre/output1.mat', {'predict': res})

def predict_dataset_conv(net, path_save, dataloader, epoches, device):
  num_val_batches = len(dataloader)
  ssim_val = 0
  ssim_f = SSIM()
  # iterate over the validation set
  with torch.no_grad():
    for epoch in range(epoches):
      logging.info(f'epoch {epoch + 1} begin')
      for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        input, label, name = batch[0], batch[1], batch[2]
        # move inputs and labels to correct device and type
        input = input.to(device=device)
        # label = label.to(device=device)
        output = net(input).cpu()
        idx_ssim = ssim_f(output, label)
        logging.info(f'{name[0]} = {idx_ssim}')
        ssim_val += idx_ssim
        output = output.squeeze(1).numpy()
        output = output.transpose()
        ff = name[0]
        sio.savemat(path_save + ff + '.mat', {'predict': output})

    logging.info(f'epoch mean ssim = {ssim_val / num_val_batches}')

def predict_conv_1D(net, path_label, path_input, device):
  label = load_rf(path_label, True)
  input = load_rf(path_input, False)
  input = input.to(device=device, dtype=torch.float32)
  label = label.to(dtype=torch.float32)
  ssim_f = SSIM()
  with torch.no_grad():
    input_shape = input.shape
    input = input.view(1, 1, -1)
    predict = net(input)
    predict = predict.view(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    predict = predict.cpu()
    idx_ssim = ssim_f(predict, label)
    logging.info(f'ssim value = {idx_ssim}')
    output = predict.squeeze(1).numpy()
    output = output.transpose()
    sio.savemat('../store_pre/output1.mat', {'predict': output})

def predict_conv_MA_1D(net, path_label, path_input, device, num_angle):
  ssim_f = SSIM()
  res = []
  sum_ssim = 0
  for i in range(1, num_angle + 1):
    logging.info(f'angle {i} begin:')
    if i == 1:
      path_input = path_input + '_' + str(i)
      path_label = path_label + '_' + str(i)
    else:
      if i <= 10:
        path_label = path_label[:-1] + str(i)
        path_input = path_input[:-1] + str(i)
      else:
        path_label = path_label[:-2] + str(i)
        path_input = path_input[:-2] + str(i)
    label = load_rf(path_label, True)
    input = load_rf(path_input, False)
    input = input.to(device=device, dtype=torch.float32)
    label = label.to(dtype=torch.float32)
    with torch.no_grad():
      # start = time.time()
      input_shape = input.shape
      input = input.view(1,1,-1)
      predict = net(input)
      predict = predict.view(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
      # end = time.time()
      # total_time = end - start
      # print('total_time:{:.2f}'.format(total_time))
      output = predict.cpu()
      # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
      idx_ssim = ssim_f(output, label)
      sum_ssim = sum_ssim + idx_ssim
      logging.info(f'ssim value = {idx_ssim}')
      # output = torch.concat([output[:, i, :, :] for i in range(23)], dim=-1)
      output = output.squeeze(1).numpy()
      output = output.transpose()
      if i == 1:
        res = output
      else:
        res = np.concatenate([res, output], axis=2)
  logging.info(f'{num_angle} angle average ssim value = {sum_ssim / num_angle}')
  sio.savemat('../store_pre/output1.mat', {'predict': res})

def predict_dataset_conv_1D(net, path_save, dataloader, epoches, device):
  num_val_batches = len(dataloader)
  ssim_val = 0
  ssim_f = SSIM()
  # iterate over the validation set
  with torch.no_grad():
    for epoch in range(epoches):
      logging.info(f'epoch {epoch + 1} begin')
      for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        input, label, name = batch[0], batch[1], batch[2]
        # move inputs and labels to correct device and type
        input = input.to(device=device)
        # label = label.to(device=device)
        input_shape = input.shape
        input = input.view(1, 1, -1)
        predict = net(input)
        predict = predict.view(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        output = predict.cpu()
        idx_ssim = ssim_f(output, label)
        logging.info(f'{name[0]} = {idx_ssim}')
        ssim_val += idx_ssim
        output = output.squeeze(1).numpy()
        output = output.transpose()
        ff = name[0]
        sio.savemat(path_save + ff + '.mat', {'predict': output})

    logging.info(f'epoch mean ssim = {ssim_val / num_val_batches}')
