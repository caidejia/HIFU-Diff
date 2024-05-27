import argparse
import logging

import torch
import yaml
from torch import nn
from torch.optim import lr_scheduler, AdamW

from FusNet_1D import FusNet_1D
from dataload import HF_n, HF_V2, HF_R, HF_23c
from HIFU_Diff import HFNet, RRDBNet
from Unet import UNet_HIFU_ori
from FusNet import FusNet

hparams = {}


def get_args():
    parser = argparse.ArgumentParser(description='Train the condition ddpm to reduce HIFU noise')
    parser.add_argument('--modify', '-m', action='store_true', default=False, help='set params by argparse')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=400, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=6, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=3e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--timestep', '-ts', default=150, type=int, help='diffusion timestep ')
    return parser.parse_args()


def init_dataset():
    if hparams['dataset']['server']:
        dir_train = hparams['dataset']['dir_train_s']
        dir_val = hparams['dataset']['dir_val_s']
    else:
        dir_train = hparams['dataset']['dir_train_f']
        dir_val = hparams['dataset']['dir_val_f']
    if hparams['dataset']['name'] == 'ori':
        dataset_train = HF_n(dir_train)
        dataset_val = HF_n(dir_val)
        return dataset_train, dataset_val
    elif hparams['dataset']['name'] == 'ori_randomcrop':
        sz = hparams['dataset']['randomcrop']['size']
        sz = tuple([int(x) for x in sz.split('|')])
        dataset_train = HF_R(dir_train, sz)
        dataset_val = HF_R(dir_val, sz)
        return dataset_train, dataset_val
    elif hparams['dataset']['name'] == 'ori_23c':
        dataset_train = HF_23c(dir_train)
        dataset_val = HF_23c(dir_val)
        return dataset_train, dataset_val
    elif hparams['dataset']['name'] == 'fft2Dcond':
        dataset_train = HF_V2(dir_train)
        dataset_val = HF_V2(dir_val)
        return dataset_train, dataset_val


def init_model(device,do_predict=False):
    if hparams['model_name'] == 'ddpm_n':
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        model = HFNet(dim_mults)
        if hparams['load_rrdb'] and not do_predict:
            path = hparams['rrdb_pth']
            state_dict = torch.load(path, map_location=device)
            model.encoder.load_state_dict(state_dict)
            logging.info(f'RRDB pretrained model loaded from {path}')
        elif hparams['load_pth']['flag'] and not do_predict:
            path = hparams['load_pth']['pretrained_pth']
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            logging.info(f'pretrained model loaded from {path}')
        model.to(device=device)
        return model
    elif hparams['model_name'] == 'rrdb':
        model = RRDBNet(1, 1)
        model.to(device=device)
        return model
    elif hparams['model_name'] == 'Unet':
        model = UNet_HIFU_ori(1, 1)
        model.to(device=device)
        return model
    elif hparams['model_name'] == 'FusNet':
        model = FusNet()
        model.to(device=device)
        return model
    elif hparams['model_name'] == '1D_FusNet':
        model = FusNet_1D()
        model.to(device=device)
        return model


def init_optimizer(model):
    if hparams['optimizer'] == 'AdamW':
        lr = hparams['lr']
        return torch.optim.AdamW(lr=lr, params=model.parameters())


def init_scheduler(optimizer):
    if hparams['scheduler']['name'] == 'CosineAnnealingLR':
        if hparams['scheduler']['CosineAnnealingLR']['max_epoch'] == 'epochs':
            T_max = hparams['epochs']
        else:
            T_max = hparams['scheduler']['CosineAnnealingLR']['max_epoch']
        return lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max)
    elif hparams['scheduler']['name'] == 'StepLR':
        step_size = hparams['scheduler']['StepLR']['step_size']
        gamma = hparams['scheduler']['StepLR']['gamma']
        return lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif hparams['scheduler']['name'] == 'ReduceLROnPlateau':
        patience = hparams['scheduler']['ReduceLROnPlateau']['patience']
        factor = hparams['scheduler']['ReduceLROnPlateau']['factor']
        return lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)


def init_loss():
    if hparams['loss'] == 'L2':
        return nn.MSELoss()
    elif hparams['loss'] == 'L1':
        return nn.L1Loss(reduction='mean')



def load_hparams():
    args = get_args()
    with open('config.yaml', 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
    hparams['project_name'] = config['project_name']
    hparams['batch_size'] = config['train']['batch_size']
    hparams['amp'] = config['train']['amp']
    hparams['lr'] = config['train']['lr']
    hparams['epochs'] = config['train']['epochs']
    hparams['checkpoint_bg'] = config['train']['checkpoint_bg']
    hparams['dataset'] = config['dataset']
    hparams['loss'] = config['train']['loss']
    hparams['optimizer'] = config['train']['optimizer']
    hparams['scheduler'] = config['train']['scheduler']
    hparams['model_name'] = config['model']['name']
    hparams['timestep'] = config['model']['timestep']
    hparams['beta_schedule'] = config['model']['beta_schedule']
    hparams['unet_dim_mults'] = config['model']['unet_dim_mults']
    hparams['load_rrdb'] = config['model']['load_rrdb']['flag']
    hparams['rrdb_pth'] = config['model']['load_rrdb']['rrdb_pth']
    hparams['load_pth'] = config['model']['load_pth']
    hparams['predict'] = config['predict']
    hparams['multi_gpu'] = config['train']['multi_gpu']
    if args.modify:
        hparams['batch_size'] = args.batch_size
        hparams['amp'] = args.amp
        hparams['lr'] = args.lr
        hparams['epochs'] = args.epochs
        hparams['timestep'] = args.timestep


def load_predict_hparams():
    with open('config.yaml', 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
    hparams['dataset'] = config['dataset']
    hparams['predict'] = config['predict']
    hparams['model_name'] = config['model']['name']
    hparams['unet_dim_mults'] = config['model']['unet_dim_mults']
    hparams['load_rrdb'] = config['model']['load_rrdb']['flag']
    hparams['load_pth'] = config['model']['load_pth']
    hparams['beta_schedule'] = config['model']['beta_schedule']
    if hparams['dataset']['server']:
        hparams['dataset']['dir_val'] = hparams['dataset']['dir_val_s']
    else:
        hparams['dataset']['dir_val'] = hparams['dataset']['dir_val_f']


def print_params():
    logging.info(f'''basic config:
        project_name:    {hparams['project_name']}
        model:           {hparams['model_name']}
        loss:            {hparams['loss']}
        dataset_deal:    {hparams['dataset']['name']}
        dataset_dir:     {hparams['dataset']['dir_train_s']}
        randomcrop_size: {hparams['dataset']['randomcrop']['size'] if hparams['dataset']['name'] == 'ori_randomcrop' else 'None'}
        optimizer:       {hparams['optimizer']}
        scheduler:       {hparams['scheduler']['name']}
        unet_dim_mults:  {hparams['unet_dim_mults']}
        rrdb_load:       {hparams['load_rrdb']}
        rrdb_load_from:  {hparams['rrdb_pth']}

        Epochs:          {hparams['epochs']}
        timestep:        {hparams['timestep']}
        beta_schedule:   {hparams['beta_schedule']}
        Batch size:      {hparams['batch_size']}
        Learning rate:   {hparams['lr']}
        Checkpoints bg:  {hparams['checkpoint_bg']}
        Mixed Precision: {hparams['amp']}

    ''')

def save_parameters():
    file_path = "parameters.txt"  # 替换为你的文件路径
    file = open(file_path, "w")
    text = f'''basic config:
        project_name:    {hparams['project_name']}
        model:           {hparams['model_name']}
        loss:            {hparams['loss']}
        dataset_deal:    {hparams['dataset']['name']}
        dataset_dir:     {hparams['dataset']['dir_train_s']}
        randomcrop_size: {hparams['dataset']['randomcrop']['size'] if hparams['dataset']['name'] == 'ori_randomcrop' else 'None'}
        optimizer:       {hparams['optimizer']}
        scheduler:       {hparams['scheduler']['name']}
        unet_dim_mults:  {hparams['unet_dim_mults']}
        rrdb_load:       {hparams['load_rrdb']}
        rrdb_load_from:  {hparams['rrdb_pth']}

        Epochs:          {hparams['epochs']}
        timestep:        {hparams['timestep']}
        beta_schedule:   {hparams['beta_schedule']}
        Batch size:      {hparams['batch_size']}
        Learning rate:   {hparams['lr']}
        Checkpoints bg:  {hparams['checkpoint_bg']}
        Mixed Precision: {hparams['amp']}

    '''
    print(text)
    file.write(text)

    # 关闭文件
    file.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    load_hparams()
    save_parameters()
