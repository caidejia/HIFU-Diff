## Training of the model

You can use the command: `CUDA_VISIBLE_DEVICES="0,...,x" python -m torch.distributed.run --nproc_per_node x train_main.py` train the model with multiple GPUs; if you only want to use 1 GPU for training use `CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.run --nproc_per_node 1 train_main.py `. The parameters for training are set in the config.ymal file.

## HIFU interference suppression

The script HIFU_interference_suppression.py offers a diverse range of methods for suppressing HIFU interference. It encompasses single-angle, multi-angle, and interference suppression for datasets. Additionally, we have implemented a multi-GPU approach to effectively generate Interference-free RF datas. 

You can execute the HIFU_interference_suppression.py script on a single GPU as follows:

`python HIFU_interference_suppression.py -m=HIFU_Diff -ts_ddim=5 -r=1`  

If you want to speed up inference on a dataset, you can execute the HIFU_interference_suppression.py script on multiple GPUs as follows:

`CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.run --nproc_per_node 2 HIFU_interference_suppression.py -m=HIFU_Diff -ts_ddim=5`

Please be aware that for the script to function properly, it should be executed alongside the config.yaml file located within the same directory.The root directory for the data slated for prediction should be specified using the "dir_val_f" or "dir_val_s" fields within the config.yaml file. Furthermore, adjust the other parameters under the "predict" section of the config.yaml file to tailor the prediction behavior of the model. This includes settings for multi-angle prediction, the deployment of multiple GPUs, the use of ddim acceleration, and other related configuration options.

metric

## Ultrasound imaging

We can use the "Imaging" script for image reconstruction, allowing us to process RF channel data directly for reconstructing images, which includes both single-angle and multiple-angle composite imaging. For single-angle imaging, the data should be formatted as (number of time sequences, number of channels). In the case of multiple-angle composite imaging, the format should be (number of time sequences, number of channels, number of angles). Furthermore, to perform multiple-angle composite imaging, we can concatenate single-angle .mat data using the "connectdata_MA" script to create multi-angle data sets. It is crucial to ensure that the angles of the data being concatenated are sequential.

## Introduction to File Structure

DL/checkpoints/: Pretrained model parameters

DL/configs/: Parameter configurations of different models

DL/pretrained_HIFU_interference_encoder/: Parameters for the pretrained HIFU interference encoder

DL/dataset/: Test data examples

store_pre/: Default path for saving model inference results

Imaging/: Ultrasound imaging scripts
