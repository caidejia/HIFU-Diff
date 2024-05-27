# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import math

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from fastprogress import progress_bar
from thop import profile
import functools
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from commons import Mish, SinusoidalPosEmb, Residual, Rezero, LinearAttention
from commons import ResnetBlock, Upsample, Downsample, RRDB
from commons import Block as Blocks
from module_util import make_layer

from ssim import SSIM


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32, nb=2, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        self.cond_proj = nn.Conv2d(32, 32, 1, bias=False)

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Blocks(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, cond):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        cond = self.cond_proj(torch.cat(cond[2::3], 1))
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + cond  # b 32 h w
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class HFNet(nn.Module):
    def __init__(self, dim_mults):
        super().__init__()
        self.encoder = RRDBNet(1, 1)
        self.denoise = Unet(32, 1, dim_mults=dim_mults)
        self.ssim_loss = SSIM(window_size=11)

    def forward(self, img, t, noise):
        img_pred, cond = self.encoder(img, True)
        noise_pred = self.denoise(noise, t, cond)

        return {
            'img_pred': img_pred,
            'noise_pred': noise_pred
        }


class Diffuser(object):
    def __init__(self, rf_size=(128, 2944), timesteps=25, beta_schedule='linear', device='cuda'):
        super().__init__()
        self.device = device
        self.noise_steps = timesteps
        self.beta_schedule = beta_schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.rf_size = rf_size

    def prepare_noise_schedule(self):
        if self.beta_schedule == 'linear':
            beta_start = 1e-4
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, self.noise_steps)
        elif self.beta_schedule == 'cosine':
            return torch.from_numpy(cosine_beta_schedule(self.noise_steps).astype(np.float32))

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, model, cond, n=1, disable=False):
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, 1, self.rf_size[0], self.rf_size[1])).to(self.device)  # 生成高斯噪声
            with tqdm(total=self.noise_steps - 1, desc=f'Epoch 1 / 1', unit='step', disable=disable) as pbar:
                for i in reversed(range(1, self.noise_steps)):
                    t = (torch.ones(n) * i).long().to(self.device)
                    predicted_noise = model(cond, t, x)['noise_pred']
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                        beta) * noise
                    pbar.update()

        return x

    def sample_ddim(self,
                    model,
                    cond,
                    n=1,
                    simple_var=False,
                    ddim_step=20,
                    eta=1,
                    disable=False):
        if simple_var:
            eta = 1
        ts = torch.linspace(self.noise_steps, 0,
                            (ddim_step + 1)).to(self.device).to(torch.long)
        x = torch.randn((n, 1, self.rf_size[0], self.rf_size[1])).to(self.device)  # 生成高斯噪声
        batch_size = x.shape[0]
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}', disable=disable):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_hat[cur_t]
            ab_prev = self.alpha_hat[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([cur_t] * batch_size, dtype=torch.long).to(self.device)
            eps = model(cond, t_tensor, x)['noise_pred']
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur) ** 0.5 * x
            second_term = ((1 - ab_prev - var) ** 0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise
            else:
                third_term = var ** 0.5 * noise
            x = first_term + second_term + third_term

        return x


if __name__ == '__main__':
    x = torch.ones(1, 1, 128, 2944)
    net = HFNet((1, 2, 4, 8))
    flops, params = profile(net, inputs=(x, torch.tensor([1]), x))
    print(flops / (1000 ** 3))
    print(params / (1000 ** 2))
