## Training of the model

You can use the command: `CUDA_VISIBLE_DEVICES="0,...,x" python -m torch.distributed.run --nproc_per_node x train_main.py` train the model with multiple GPUs; if you only want to use 1 GPU for training use `CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.run --nproc_per_node 1 train_main.py `. The parameters for training are set in the config.ymal file.

## HIFU interference suppression

The file HIFU_interference_suppression.py offers a diverse range of methods for suppressing HIFU interference. It encompasses single-angle, multi-angle, and interference suppression for datasets. Additionally, we have implemented a multi-GPU approach to effectively generate Interference-free RF datas. You can modify these parameters in the config.yaml file. You can run the HIFU_interference_suppression.py like `python HIFU_interference_suppression.py -m=total_epoch_2000 -ts=100 -r=30_1` for single GPU。 If you want to speed up inference on a dataset you can use the command like `CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.run --nproc_per_node 2 HIFU_interference_suppression.py -m=total_epoch_2000 -ts=100 -e=3` for using multiple GPUs.



## Introduction to File Structure

DL/checkpoints: Pretrained model parameters

DL/configs: Parameter configurations of different models

DL/pretrained : Pretrained RRDB model parameters

DL/dataset: Partial training and testing dataset

store_pre/: Default path for saving model inference results

