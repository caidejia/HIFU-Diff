import torch
import glob
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torch
import numpy as np
import scipy.io as scio



class HF_23c(Dataset):
    def __init__(self, path):
        self.base_path = path
        self.file = os.listdir(os.path.join(path, 'input'))

    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, item):
        inp = scio.loadmat(os.path.join(os.path.join(self.base_path, 'input'), self.file[item]))['Data']
        tar = scio.loadmat(os.path.join(os.path.join(self.base_path, 'label'), self.file[item]))['Label']



        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)

        # inp = (inp-inp.min()) / (inp.max()-inp.min())
        # tar = (tar - tar.min()) / (tar.max() - tar.min())

        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)

        # seed = torch.random.seed()
        # torch.random.manual_seed(seed)
        # inp = transforms.RandomCrop((256, 128))(inp)
        # torch.random.manual_seed(seed)
        # tar = transforms.RandomCrop((256, 128))(tar)

        inp = inp / torch.max(torch.abs(inp))
        tar = tar / torch.max(torch.abs(tar))

        inp = inp[None, :, :].permute(0, 2, 1)
        tar = tar[None, :, :].permute(0, 2, 1)

        # inp = transforms.Normalize(0.5001, 0.1676)(inp)
        # tar = transforms.Normalize(0.5001, 0.1676)(tar)

        inp = torch.concat([inp[:, :, (i * 128):(i * 128 + 128)] for i in range(23)], dim=0)
        tar = torch.concat([tar[:, :, (i * 128):(i * 128 + 128)] for i in range(23)], dim=0)
        return inp, tar

class HF_n(Dataset):
    def __init__(self, path):
        self.base_path = path
        self.file = os.listdir(os.path.join(path, 'input'))

    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, item):
        inp = scio.loadmat(os.path.join(os.path.join(self.base_path, 'input'), self.file[item]))['Data']
        tar = scio.loadmat(os.path.join(os.path.join(self.base_path, 'label'), self.file[item]))['Label']


        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)

        # inp = (inp-inp.min()) / (inp.max()-inp.min())
        # tar = (tar - tar.min()) / (tar.max() - tar.min())

        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)

        # seed = torch.random.seed()
        # torch.random.manual_seed(seed)
        # inp = transforms.RandomCrop((256, 128))(inp)
        # torch.random.manual_seed(seed)
        # tar = transforms.RandomCrop((256, 128))(tar)

        inp = inp / torch.max(torch.abs(inp))
        tar = tar / torch.max(torch.abs(tar))

        inp = inp[None, :, :].permute(0, 2, 1).contiguous()
        tar = tar[None, :, :].permute(0, 2, 1).contiguous()

        return inp, tar

class HF_n_pre(Dataset):
    def __init__(self, path):
        self.base_path = path
        self.file = os.listdir(os.path.join(path, 'input'))
        self.label_index = self.get_index()

    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, item):
        inp = scio.loadmat(os.path.join(os.path.join(self.base_path, 'input'), self.file[item]))['Data']
        tar = scio.loadmat(os.path.join(os.path.join(self.base_path, 'label'), self.file[item]))['Label']


        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)

        # inp = (inp-inp.min()) / (inp.max()-inp.min())
        # tar = (tar - tar.min()) / (tar.max() - tar.min())

        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)

        # seed = torch.random.seed()
        # torch.random.manual_seed(seed)
        # inp = transforms.RandomCrop((256, 128))(inp)
        # torch.random.manual_seed(seed)
        # tar = transforms.RandomCrop((256, 128))(tar)

        inp = inp / torch.max(torch.abs(inp))
        tar = tar / torch.max(torch.abs(tar))

        inp = inp[None, :, :].permute(0, 2, 1)
        tar = tar[None, :, :].permute(0, 2, 1)

        return inp, tar,self.label_index[item]

    def get_index(self):
        label_index = []
        for path in self.file:
            index = path.split("/")[-1].split(".mat")[0]
            label_index.append(index)
        return label_index


class HF_n_PreWithoutLabel(Dataset):
    def __init__(self, path):
        self.base_path = path
        self.file = os.listdir(path)
        self.label_index = self.get_index()

    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, item):
        inp = scio.loadmat(os.path.join(self.base_path, self.file[item]))['Data']
        inp = np.array(inp, dtype=np.float32)
        inp = torch.from_numpy(inp)
        inp = inp / torch.max(torch.abs(inp))
        inp = inp[None, :, :].permute(0, 2, 1)

        return inp,self.label_index[item]

    def get_index(self):
        label_index = []
        for path in self.file:
            index = path.split("/")[-1].split(".mat")[0]
            label_index.append(index)
        return label_index


class HF_R(Dataset):
    def __init__(self, path,sz):
        self.base_path = path
        self.file = os.listdir(os.path.join(path, 'input'))
        self.sz=sz

    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, item):
        inp = scio.loadmat(os.path.join(os.path.join(self.base_path, 'input'), self.file[item]))['Data']
        tar = scio.loadmat(os.path.join(os.path.join(self.base_path, 'label'), self.file[item]))['Label']


        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)

        # inp = (inp-inp.min()) / (inp.max()-inp.min())
        # tar = (tar - tar.min()) / (tar.max() - tar.min())

        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        inp = transforms.RandomCrop(self.sz)(inp)
        torch.random.manual_seed(seed)
        tar = transforms.RandomCrop(self.sz)(tar)

        inp = inp / torch.max(torch.abs(inp))
        tar = tar / torch.max(torch.abs(tar))

        inp = inp[None, :, :].permute(0, 2, 1).contiguous()
        tar = tar[None, :, :].permute(0, 2, 1).contiguous()

        return inp, tar


class HF_V2(Dataset):
    def __init__(self, path):
        self.base_path = path
        self.file = os.listdir(os.path.join(path, 'input'))

    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, item):
        inp = scio.loadmat(os.path.join(os.path.join(self.base_path, 'input'), self.file[item]))['Data']
        cond = scio.loadmat(os.path.join(os.path.join(self.base_path, 'input'), self.file[item]))['fftData']
        tar = scio.loadmat(os.path.join(os.path.join(self.base_path, 'label'), self.file[item]))['Label']

        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)
        cond = np.array(cond, dtype=np.float32)
        # inp = (inp-inp.min()) / (inp.max()-inp.min())
        # tar = (tar - tar.min()) / (tar.max() - tar.min())

        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)
        cond = torch.from_numpy(cond)

        # seed = torch.random.seed()
        # torch.random.manual_seed(seed)
        # inp = transforms.RandomCrop((256, 128))(inp)
        # torch.random.manual_seed(seed)
        # tar = transforms.RandomCrop((256, 128))(tar)

        inp = inp / torch.max(torch.abs(inp))
        tar = tar / torch.max(torch.abs(tar))

        inp = inp[None, :, :].permute(0, 2, 1)
        tar = tar[None, :, :].permute(0, 2, 1)
        cond = cond[None, :, :].permute(0, 2, 1)



        return inp, tar,cond





if __name__ == '__main__':
    from ssim import SSIM, MSSSIM
    dataset = HF_23c('data/0530/0530')
    inp, tar = dataset[0]
    msssim = MSSSIM()
    ssim = SSIM()
    import torch.nn as nn
    print((1-msssim(inp, tar)), (1-ssim(inp, tar)))
