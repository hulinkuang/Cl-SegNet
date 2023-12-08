# Cl-SegNet
The code for the paper "Hybrid CNN-Transformer Network with Circular Feature Interaction for Acute Ischemic Stroke Lesion Segmentation on Non-contrast CT Scans" submitted to IEEE TMI. <br />


## Requirements
CUDA 11.4<br />
Python 3.8.12<br /> 
Pytorch 1.11.0<br />
Torchvision 0.12.0<br />
batchgenerators 0.21<br />
SimpleITK 2.1.1 <br />
scipy 1.8.0 <br />

## Usage

### 0. Installation
* Install nnUNet and ClSeg as below
  
```
cd nnUNet
pip install -e .

cd ClSeg_package
pip install -e .
```
### 1 Acute Ischemic Stroke Dataset (AISD)
### 1.1 Dataset access
AISD dataset can be downloaded from (https://github.com/griffinliang/aisd). Pre-process the datasets using the preprocess codes in nnUNet/nnunet/dataset_conversion.

### 1.2 Pre-processing
The documentation of the pre-processing can be found at [[DOC]](./nnUNet/documentation) <br />

### 1.3 Training
cd ClSeg_package/ClSeg/run

* Run `python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu='0' -task={task_id} -outpath='AISD'` for training.

### 1.4 Testing 
* Run `python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu='0' -task={task_id} -outpath='AISD' -val --val_folder='validation_output'` for testing.

### 2.1 Pre-trained model
The pre-trained model of AISD dataset can be downloaded from [[Baidu YUN]](https://pan.baidu.com/s/1RmswEZsQewr7UcC14UCKMA) with the password "4phx".

### 2.2 Reproduction codes 
For the methods (i.e. LambdaUNet [1], UNet-AM [2], UNet-GC [3]) that do not publish their codes, we reproduce these methods by ourselves, which can be found at [[DOC]](./ClSeg_package/ClSeg/network_architecture)

### 2.3 Open-source codes
The open-source codes can are as follows: <br />

[[AttnUnet2D]](https://github.com/sfczekalski/attention_unet) </br>
[[Swin-Unet]](https://github.com/HuCaoFighting/Swin-Unet) </br>
[[TransUNet]](https://github.com/Beckschen/TransUNet) </br>
[[FAT-Net]](https://github.com/SZUcsh/FAT-Net) </br>
[[AttnUNet3D]](https://github.com/mobarakol/3D_Attention_UNet) </br>
[[nnFormer]](https://github.com/282857341/nnFormer) </br>
[[UNETR]](https://github.com/282857341/nnFormer) </br>
[[CoTr]](https://github.com/YtongXie/CoTr) </br>
[[nnUNet]](https://github.com/MIC-DKFZ/nnUNet) </br>
[[UNet-RF]](https://github.com/WuChanada/Acute-ischemic-lesion-segmentation-in-NCCT)


## Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.
