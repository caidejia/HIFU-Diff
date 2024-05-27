import os
import torch.distributed as dist
from trainModel import train_model_n, train_model_rrdb, train_model_Conv, train_model_Conv_1D
from init_hparams import *


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    load_hparams()
    print_params()
    save_parameters()
    device = init_device()
    try:
        if hparams['model_name'] == 'ddpm_n':
            train_model_n(
                epochs=hparams['epochs'],
                batch_size=hparams['batch_size'],
                amp=hparams['amp'],
                device=device
            )
        elif hparams['model_name'] == 'rrdb':
            train_model_rrdb(
                epochs=hparams['epochs'],
                batch_size=hparams['batch_size'],
                amp=hparams['amp'],
                device=device
            )
        elif hparams['model_name'] == 'Unet' or 'FusNet':
            train_model_Conv(
                epochs=hparams['epochs'],
                batch_size=hparams['batch_size'],
                amp=hparams['amp'],
                device=device
            )
        elif hparams['model_name'] == '1D_Unet' or '1D_FusNet':
            train_model_Conv_1D(
                epochs=hparams['epochs'],
                batch_size=hparams['batch_size'],
                amp=hparams['amp'],
                device=device
            )

    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
