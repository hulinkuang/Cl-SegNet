# Cl-SegNet
The code for the paper "CNN and Transformer Hybrid Network with Circular Feature Interaction for Acute Ischemic Stroke Lesion Segmentation on Non-contrast CT Scans" submitted to IEEE TMI. <br />


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
### 1.1 Dataset access and pre-processing
AISD dataset can be downloaded from (https://github.com/griffinliang/aisd). Pre-process the datasets using the preprocess codes in nnUNet/nnunet/dataset_conversion.
### 1.2 Training
cd ClSeg_package/ClSeg/run

* Run `python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu='0' -task={task_id} -outpath='AISD'` for training.

### 1.3 Testing 
* Run `python run_training.py -network_trainer nnUNetTrainerV2_AISD -gpu='0' -task={task_id} -outpath='AISD' -val --val_folder='validation_output'` for testing.

## Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.
