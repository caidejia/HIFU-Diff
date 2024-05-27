import os
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
import torch.distributed as dist
from HIFU_Diff import Diffuser
from evaluate_HIFU import evaluate_HIFU, evaluate_HIFU_1D
from evaluate_rrdb import evaluate_RRDB
from init_hparams import *
from ssim import ssim_loss


dir_checkpoint = Path('./checkpoints/')


def train_model_n(
        device,
        epochs: int = 5,
        batch_size: int = 1,
        amp: bool = False,
        gradient_clipping: float = 1.0,
):

    # 1. Create dataset
    dataset, _ = init_dataset()

    # 2. Split into train / validation partitions
    n_train = len(dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    if hparams['multi_gpu']:
     train_sampler = DistributedSampler(dataset, shuffle=True)
     train_loader = DataLoader(dataset,sampler=train_sampler,**loader_args)
    else:
      train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    # (Initialize logging)
    if dist.get_rank() == 0:
      experiment = wandb.init(project=hparams['project_name'], resume='allow', anonymous='must')
      experiment.config.update(
          dict(device=device,
               Model=hparams['model_name'],
               loss=hparams['loss'],
               dataset_deal=hparams['dataset']['name'],
               randomcrop_size=hparams['dataset']['randomcrop']['size'] if hparams['dataset']['name'] == 'ori_randomcrop' else 'None',
               dataset_dir=hparams['dataset']['dir_train_s'],
               optimizer=hparams['optimizer'],
               scheduler=hparams['scheduler']['name'],
               unet_dim_mults=hparams['unet_dim_mults'],
               rrdb_load=hparams['load_rrdb'],
               rrdb_load_from=hparams['rrdb_pth'],
               Epochs=hparams['epochs'],
               timestep=hparams['timestep'],
               batch_size=hparams['batch_size'],
               lr=hparams['lr'],
               checkpoints_bg=hparams['checkpoint_bg'],
               amp=hparams['amp'],
               n_train=n_train
               ))

    # 4. Set up the model, the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    model = init_model(device)
    if hparams['multi_gpu']:
      world_size = torch.distributed.get_world_size()
      local_rank = int(os.environ["LOCAL_RANK"])
      model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    diff = Diffuser(timesteps=hparams['timestep'],beta_schedule=hparams['beta_schedule'])

    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = init_loss()
    global_step = 0
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        if hparams['multi_gpu']:
          train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img',disable=(dist.get_rank()!=0)) as pbar:
            for batch in train_loader:
                images, labels = batch[0], batch[1]
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                t = diff.sample_timesteps(images.shape[0]).to(device)
                x_0 = labels - images
                x_t, noise = diff.noise_images(x_0, t)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    predicted_noise = model(images, t, x_t)['noise_pred']
                    loss = criterion(noise, predicted_noise)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()  # 计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 梯度裁剪，这可以防止梯度爆炸的情况
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1  # 处理一个batch就是一个step
                if hparams['multi_gpu']:
                  dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                  # 然后除以并行数，就是这个batch的loss值了
                  loss /= world_size
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            scheduler.step()
            # Evaluation round
            # ssim_val = evaluate_ddpm(model,diff,val_loader,device, amp)
            if dist.get_rank()==0:
              logging.info('Epoch mean loss :{}'.format(epoch_loss / len(train_loader)))
            # logging.info('Eva sig RF Loss: {}'.format(eva_loss))
            # logging.info('Eva sig RF ssim_val: {}'.format(ssim_val))
              experiment.log({
                  'learning rate': optimizer.param_groups[0]['lr'],
                  'epoch mean loss': epoch_loss / len(train_loader),
                  'epoch': epoch
              })

        if epoch >= hparams['checkpoint_bg'] and epoch % 10 == 0 and dist.get_rank() == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            if hparams['multi_gpu']:
              state_dict = model.module.state_dict()
            else:
              state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

def train_model_rrdb(
        device,
        epochs: int = 5,
        batch_size: int = 1,
        amp: bool = False,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset, val_set = init_dataset()

    # 2. Split into train / validation partitions
    n_train = len(dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    if hparams['multi_gpu']:
      train_sampler = DistributedSampler(dataset, shuffle=True)
      train_loader = DataLoader(dataset, sampler=train_sampler, **loader_args)
    else:
      train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    # (Initialize logging)
    if dist.get_rank() == 0:
      experiment = wandb.init(project=hparams['project_name'], resume='allow', anonymous='must')
      experiment.config.update(
          dict(device=device,
               Model=hparams['model_name'],
               loss=hparams['loss'],
               dataset_deal=hparams['dataset']['name'],
               dataset_dir=hparams['dataset']['dir_train_s'],
               optimizer=hparams['optimizer'],
               scheduler=hparams['scheduler']['name'],
               unet_dim_mults=hparams['unet_dim_mults'],
               rrdb_load=hparams['load_rrdb'],
               rrdb_load_from=hparams['rrdb_pth'],
               Epochs=hparams['epochs'],
               timestep=hparams['timestep'],
               batch_size=hparams['batch_size'],
               lr=hparams['lr'],
               checkpoints_bg=hparams['checkpoint_bg'],
               amp=hparams['amp'],
               n_train=n_train
               ))

    # 4. Set up the model, the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    model = init_model(device)
    if hparams['multi_gpu']:
      world_size = torch.distributed.get_world_size()
      local_rank = int(os.environ["LOCAL_RANK"])
      model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = init_loss()
    global_step = 0
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', disable=(dist.get_rank()!=0)) as pbar:
            for batch in train_loader:
                images, mask_true = batch[0], batch[1]  # true_mask是一个BHW的张量 images是一个BCHW或者BHW
                images = images.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    mask_pred = model(images)  # BCHW
                    loss = criterion(mask_pred.squeeze(1), images.squeeze(1))
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()  # 计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 梯度裁剪，这可以防止梯度爆炸的情况
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1  # 处理一个batch就是一个step
                if hparams['multi_gpu']:
                  dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                  # 然后除以并行数，就是这个batch的loss值了
                  loss /= world_size
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # if global_step % 20 == 0:
            scheduler.step()
            # Evaluation round
            if dist.get_rank() == 0:
              # eva_loss, ssim_val = evaluate_RRDB(model, val_loader, device, amp)
              # ssim_val = ssim_val / len(val_loader)
              logging.info('Epoch mean loss :{}'.format(epoch_loss / len(train_loader)))
              # logging.info('Eva Epoch Loss: {}'.format(eva_loss))
              # logging.info('Eva Epoch ssim_val: {}'.format(ssim_val))

              experiment.log({
                  'learning rate': optimizer.param_groups[0]['lr'],
                  'epoch mean loss': epoch_loss / len(train_loader),
                  # 'evaluate loss': eva_loss,
                  # 'Eva Epoch ssim_val: {}': ssim_val,
                  'epoch': epoch
                  # **histograms,
              })

        if epoch >= hparams['checkpoint_bg'] and epoch % 10 == 0 and dist.get_rank() == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            if hparams['multi_gpu']:
              state_dict = model.module.state_dict()
            else:
              state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

def train_model_Conv(
        device,
        epochs: int = 5,
        batch_size: int = 1,
        amp: bool = False,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset, val_set = init_dataset()

    # 2. Split into train / validation partitions
    n_train = len(dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    # (Initialize logging)
    experiment = wandb.init(project=hparams['project_name'], resume='allow', anonymous='must')
    experiment.config.update(
       dict(device=device,
            Model=hparams['model_name'],
           loss=hparams['loss'],
           dataset_deal=hparams['dataset']['name'],
           dataset_dir=hparams['dataset']['dir_train_s'],
           optimizer=hparams['optimizer'],
           scheduler=hparams['scheduler']['name'],
           Epochs=hparams['epochs'],
           batch_size=hparams['batch_size'],
           lr=hparams['lr'],
           checkpoints_bg=hparams['checkpoint_bg'],
           amp=hparams['amp'],
           n_train=n_train
           ))

    # 4. Set up the model, the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    model = init_model(device)

    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = init_loss()
    global_step = 0
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, mask_true = batch[0], batch[1]  # true_mask是一个BHW的张量 images是一个BCHW或者BHW
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                mask_true = mask_true.to(device=device, non_blocking=True, dtype=torch.float32)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    mask_pred = model(images)  # BCHW
                    loss = 0.8*criterion(mask_pred.squeeze(1), mask_true.squeeze(1)) + 0.2 * ssim_loss(mask_pred, mask_true)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()  # 计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 梯度裁剪，这可以防止梯度爆炸的情况
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1  # 处理一个batch就是一个step
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # if global_step % 20 == 0:

            # Evaluation round
            eva_loss, ssim_val = evaluate_HIFU(model, val_loader, device, amp)
            ssim_val = ssim_val / len(val_loader)
            if hparams['scheduler']['name'] == 'ReduceLROnPlateau':
                scheduler.step(epoch_loss / len(train_loader))
            else:
                scheduler.step()

            logging.info('Epoch mean loss :{}'.format(epoch_loss / len(train_loader)))
            logging.info('Eva Epoch Loss: {}'.format(eva_loss))
            logging.info('Eva Epoch ssim_val: {}'.format(ssim_val))

            experiment.log({
              'learning rate': optimizer.param_groups[0]['lr'],
              'epoch mean loss': epoch_loss / len(train_loader),
              'evaluate loss': eva_loss,
              'Eva Epoch ssim_val: {}': ssim_val,
              'epoch': epoch
              # **histograms,
          })

        if epoch >= hparams['checkpoint_bg'] and epoch % 5 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

def train_model_Conv_1D(
        device,
        epochs: int = 5,
        batch_size: int = 1,
        amp: bool = False,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset, val_set = init_dataset()

    # 2. Split into train / validation partitions
    n_train = len(dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    # (Initialize logging)
    experiment = wandb.init(project=hparams['project_name'], resume='allow', anonymous='must')
    experiment.config.update(
       dict(device=device,
            Model=hparams['model_name'],
           loss=hparams['loss'],
           dataset_deal=hparams['dataset']['name'],
           dataset_dir=hparams['dataset']['dir_train_s'],
           optimizer=hparams['optimizer'],
           scheduler=hparams['scheduler']['name'],
           Epochs=hparams['epochs'],
           batch_size=hparams['batch_size'],
           lr=hparams['lr'],
           checkpoints_bg=hparams['checkpoint_bg'],
           amp=hparams['amp'],
           n_train=n_train
           ))

    # 4. Set up the model, the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    model = init_model(device)

    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = init_loss()
    global_step = 0
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, mask_true = batch[0], batch[1]  # true_mask是一个BHW的张量 images是一个BCHW或者BHW
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                mask_true = mask_true.to(device=device, non_blocking=True, dtype=torch.float32)
                images_shape = images.shape
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    images = images.view(1,1,-1)
                    mask_pred = model(images)
                    mask_pred = mask_pred.view(images_shape[0], images_shape[1], images_shape[2], images_shape[3])
                    loss = 0.8 * criterion(mask_pred.squeeze(1), mask_true.squeeze(1)) + 0.2 * ssim_loss(mask_pred,mask_true)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()  # 计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 梯度裁剪，这可以防止梯度爆炸的情况
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1  # 处理一个batch就是一个step
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # if global_step % 20 == 0:

            # Evaluation round
            eva_loss, ssim_val = evaluate_HIFU_1D(model, val_loader, device, amp)
            ssim_val = ssim_val / len(val_loader)
            if hparams['scheduler']['name'] == 'ReduceLROnPlateau':
                scheduler.step(epoch_loss / len(train_loader))
            else:
                scheduler.step()

            logging.info('Epoch mean loss :{}'.format(epoch_loss / len(train_loader)))
            logging.info('Eva Epoch Loss: {}'.format(eva_loss))
            logging.info('Eva Epoch ssim_val: {}'.format(ssim_val))

            experiment.log({
              'learning rate': optimizer.param_groups[0]['lr'],
              'epoch mean loss': epoch_loss / len(train_loader),
              'evaluate loss': eva_loss,
              'Eva Epoch ssim_val: {}': ssim_val,
              'epoch': epoch
              # **histograms,
          })

        if epoch >= hparams['checkpoint_bg'] and epoch % 5 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    load_hparams()
    print_params()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        train_model_n(
            epochs=hparams['epochs'],
            batch_size=hparams['batch_size'],
            amp=hparams['amp'],
            device=device
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
