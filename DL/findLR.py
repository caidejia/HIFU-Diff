import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from HIFU_Diff import Diffuser
from init_hparams import *


class ScheduledOptim(object):
  '''A wrapper class for learning rate scheduling'''

  def __init__(self, optimizer):
    self.optimizer = optimizer
    self.lr = self.optimizer.param_groups[0]['lr']
    self.current_steps = 0

  def step(self):
    "Step by the inner optimizer"
    self.current_steps += 1
    self.optimizer.step()

  def zero_grad(self):
    "Zero out the gradients by the inner optimizer"
    self.optimizer.zero_grad()

  def set_learning_rate(self, lr):
    self.lr = lr
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

  @property
  def learning_rate(self):
    return self.lr

if __name__ == '__main__':
  load_hparams()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset, val_set = init_dataset()
  loader_args = dict(batch_size=16, num_workers=os.cpu_count(), pin_memory=True)
  train_loader = DataLoader(dataset, shuffle=True, **loader_args)
  model = init_model(device)
  criterion = init_loss()
  basic_optim = torch.optim.AdamW(lr=1e-5, params=model.parameters())
  optimizer = ScheduledOptim(basic_optim)
  diff = Diffuser(timesteps=hparams['timestep'])
  n_train = len(dataset)
  lr_mult = (1 / 1e-5) ** (1 / 100)
  lr = []
  losses = []
  best_loss = 1e9
  with tqdm(total=n_train, unit='img') as pbar:
    for batch in train_loader:
      images, labels = batch[0], batch[1]
      images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
      labels = labels.to(device=device, dtype=torch.float32)
      t = diff.sample_timesteps(images.shape[0]).to(device)
      x_0 = labels - images
      x_t, noise = diff.noise_images(x_0, t)
      predicted_noise = model(images, t, x_t)['noise_pred']
      loss = criterion(noise, predicted_noise)
      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr.append(optimizer.learning_rate)
      losses.append(loss.data[0])
      optimizer.set_learning_rate(optimizer.learning_rate * lr_mult)
      pbar.update(images.shape[0])
      if loss.data[0] < best_loss:
          best_loss = loss.data[0]
      if loss.data[0] > 4 * best_loss or optimizer.learning_rate > 1.:
          break

  plt.figure()
  plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
  plt.xlabel('learning rate')
  plt.ylabel('loss')
  plt.plot(np.log(lr), losses)
  plt.show()
  plt.figure()
  plt.xlabel('num iterations')
  plt.ylabel('learning rate')
  plt.plot(lr)
  plt.savefig('LR RATE.png')
