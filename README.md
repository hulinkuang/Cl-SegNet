# Cl-SegNet
The code for the AAAI 2023 paper "Combining CNN and Transformer via circular feature interaction for Stroke Lesion Segmentation on non-contrast CT scans" <br />


## Requirements
CUDA 10.2<br />
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

### 2 ISLES 2018
### 2.1 Dataset access and pre-processing
ISLES 2018 dataset can be downloaded from (http://www.isles-challenge.org/ISLES2018/). Pre-process the datasets using the preprocess codes in nnUNet/nnunet/dataset_conversion.
### 2.2 Training
cd ClSeg_package/ClSeg/run

* Run `python run_training.py -network_trainer nnUNetTrainerV2_ISLES -gpu='0' -task={task_id} -outpath='ISLES'` for training.

### 2.3 Testing 
* Run `python run_training.py -network_trainer nnUNetTrainerV2_ISLES -gpu='0' -task={task_id} -outpath='ISLES' -val --val_folder='validation_output'` for testing. 


## Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.
