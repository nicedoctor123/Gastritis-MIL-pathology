# Deep learning for comprehensively assessing chronic gastritis: a multicenter, retrospective cohort study.

This repository provides scripts to reproduce the results in the paper "Deep learning for comprehensively assessing chronic gastritis from whole-slide images: a multicenter, retrospective cohort study".

GastritisMIL performed similarly to the two senior pathologists and outperformed the junior pathologist by a large extent, efficiently identifying abnormal alterations with a WSI-level interpretation heatmap and reducing the risk of missed diagnoses.
![image](https://github.com/nicedoctor123/Gastritis-MIL-pathology/blob/main/example.png)

## Pubilications
Submitted to the jounal

## Requirements
### Hardware Requirements
Make sure the system contains adequate amount of main memory space (minimal: 20 GB) to prevent out-of-memory error.

### Software Stacks
You should install the following native libraries manually in advance.

- CUDA 12.2

CUDA is essential for PyTorch to enable GPU-accelerated deep neural network training. Please see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ .

- Python 3.11

The development kit should be installed.
```
sudo apt install python3.11-dev
```

- OpenSlide 1.3.1
OpenSlide is a library for reading whole slide image files (also known as virtual slides).
Please see the installation guide in https://github.com/openslide/openslide.

### Python Packages

- h5py 3.10.0
- numpy 1.26.3
- openslide-python 1.3.1
- pandas 2.2.0
- scikit-learn 1.4.0
- scipy 1.12.0
- torchvision 0.16.2
- tensorboard 2.15.1
- torch 2.1.2
- torch_geometric 2.4.0
- utils 1.0.2

## Usage

### Whole-slide images model (GastritisMIL)

Our model requires slide-level labels to be trained and tested








