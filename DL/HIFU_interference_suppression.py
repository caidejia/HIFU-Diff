from interference_util import *

def get_args():
  parser = argparse.ArgumentParser(description='HIFU Interference Suppression')
  parser.add_argument('--model', '-m', default='HIFU_Diff', metavar='FILE',
                      help='Name of the pre-training parameter to be loaded')
  parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1,
                      help='Number of batch for HIFU interference suppression for the whole dataset, default is 1 no need to modify')
  parser.add_argument('--rf', '-r', default='1',
                      help='The name of the RF data file that needs to suppress HIFU interference (xx_xx for single angle, xx for multi-angle)')
  parser.add_argument('--num', '-n', default=1, type=int,
                      help='Number of repetitions of interference suppression for the same RF (repeated tests)')
  parser.add_argument('--Anum', '-an', default=21, type=int, help='Number of angles required for imaging')
  parser.add_argument('--timestep', '-ts', default=200, type=int, help='Sample timestep')
  parser.add_argument('--ts_ddim', '-dits', default=5, type=int, help='DDIM sample timestep')
  parser.add_argument('--eta_ddim', '-eta', default=0.5, type=int, help='DDIM eta')
  parser.add_argument('--epoches', '-e', default=1, type=int,
                      help='Number of repeated predictions for the dataset (repeated experiments)')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  load_predict_hparams()
  if hparams['predict']['mgpu']:
    device = init_device()
  else:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[
      logging.FileHandler('../store_pre/output1.txt'),
      logging.StreamHandler()  #
    ])
  logging.info(f'Using device {device}')
  # load encoder cond DP
  DP = init_model(device, do_predict=True)
  model_path = hparams['predict']['predict_checkpoints']
  model_name = args.model + '.pth'
  DP_state_dict = torch.load(model_path + model_name, map_location=device)
  # DP_state_dict = torch.load(model_path + model_name, map_location=device)['state_dict']
  DP.load_state_dict(DP_state_dict)
  DP.eval()

  # load noised RF as condition
  input_root = hparams['dataset']['dir_val'] + '/input/'
  label_root = hparams['dataset']['dir_val'] + '/label/'
  dir_input = input_root + args.rf
  dir_label = label_root + args.rf
  diffuser = Diffuser(timesteps=args.timestep, beta_schedule=hparams['beta_schedule'])
  path_save = hparams['predict']['path_save']


  # logging
  logging.info(f'Load cond DP model {args.model} success!')
  mode = hparams['predict']['mode']
  if mode != 2:
    logging.info(f'rf load from {args.rf}.mat success!')
  else:
    pp = hparams['dataset']['dir_val']
    logging.info(f'dataset path {pp}')
  if hparams['predict']['ddpmMode']:
    logging.info(f'total num {args.num}!')
    logging.info(f'total epoches {args.epoches}!')
    logging.info(f'total diffusion step {args.timestep}!')
  if hparams['predict']['ddpmMode']:
    if mode == 3:
      val_set = HF_n_PreWithoutLabel(hparams['dataset']['dir_val'])
      loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
      if hparams['predict']['mgpu']:
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(val_set, sampler=val_sampler, **loader_args)
        denoise_datasetWithoutlabel_mg(DP, diffuser, path_save, val_loader, args.epoches, device, val_sampler)
      else:
        input_root = hparams['dataset']['dir_val']
        denoise_datasetWithoutlabel(DP, diffuser, input_root, path_save, device)
    elif mode == 2:
      val_set = HF_n_pre(hparams['dataset']['dir_val'])
      loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
      if hparams['predict']['mgpu']:
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(val_set, sampler=val_sampler, **loader_args)
        denoise_dataset_mg(DP, diffuser, path_save, val_loader, args.epoches, device, val_sampler,hparams['predict']['use_ddim'],args.ts_ddim,args.eta_ddim)
      else:
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
        denoise_dataset(DP, diffuser, path_save, val_loader, args.epoches, device,hparams['predict']['use_ddim'],args.ts_ddim,args.eta_ddim)
    elif mode == 1:
      if hparams['predict']['MA_predict']:
        denoise_MA(DP, diffuser, dir_label, dir_input, device, args.num, args.Anum,hparams['predict']['use_ddim'],args.ts_ddim,args.eta_ddim)
      else:
        denoise(DP, diffuser, dir_label, dir_input, device, args.num,hparams['predict']['use_ddim'],args.ts_ddim,args.eta_ddim)
  else:
    if hparams['model_name']=='1D_Unet' or hparams['model_name']=='1D_FusNet':
      if mode == 1:
        if hparams['predict']['MA_predict']:
          predict_conv_MA_1D(DP, dir_label, dir_input, device, args.Anum)
        else:
          predict_conv_1D(DP, dir_label, dir_input, device)
      elif mode == 2:
        val_set = HF_n_pre(hparams['dataset']['dir_val'])
        loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
        predict_dataset_conv_1D(DP, path_save, val_loader, args.epoches, device)
    else:
      if mode == 2:
        val_set = HF_n_pre(hparams['dataset']['dir_val'])
        loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
        predict_dataset_conv(DP, path_save, val_loader, args.epoches, device)
      elif mode == 1:
        if hparams['predict']['MA_predict']:
          predict_conv_MA(DP, dir_label, dir_input, device, args.Anum)
        else:
          predict_conv(DP, dir_label, dir_input, device)
  logging.info('denoise successfully!')
  logging.info('  ')
