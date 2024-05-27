""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class Conv1df(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class SeqDown(nn.Module):  # 前3层使用
    def __init__(self, in_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=4, padding=1, bias=False, stride=1),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=4, padding=3, bias=False, stride=2),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class SeqDown2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=4, padding=1, bias=False, stride=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, in_channels, kernel_size=4, padding=3, bias=False, stride=2),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        return self.seq(x)


class SeqUp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class SeqUp2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            SeqDown(in_channels),
            Conv1df(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Down2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            SeqDown2(in_channels),
            Conv1df(in_channels, out_channels),

        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = SeqUp(in_channels)
        self.conv = Conv1df(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.seq(x1)
        # input is CHW
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class FusNet_1D(nn.Module):
    def __init__(self, bilinear=False):
        super(FusNet_1D, self).__init__()
        self.bilinear = bilinear
        self.inc = Conv1df(1,32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down2(256, 512)
        self.down5 = SeqDown2(512)
        self.up1 = SeqUp2(512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.outc = OutConv(32,1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_b = self.down5(x5)
        x = self.up1(x_b)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    net = FusNet_1D()
    x = torch.randn(10,1,2944)
    y = net(x)
    print(y.size())