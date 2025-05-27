# PrivDiffuser
[![PoPETS](https://img.shields.io/badge/PoPETs-2025-blue?style=flat)]()
[![arXiv](https://img.shields.io/badge/arXiv-2209.12046-b31b1b.svg)](https://arxiv.org/abs/2412.14499)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/sustainable-computing/PrivDiffuser/blob/main/LICENSE)

This repository contains the implementation of the paper entitled "PrivDiffuser: Privacy-Guided Diffusion Model for Data Obfuscation in Sensor Networks.


## Datasets
PrivDiffuser is evaluated on three Human Activity Recognition (HAR) datasets: MotionSense, MobiAct, and WiFi-HAR. 

The datasets and the preprocessing script (required for MobiAct) are available at:

- MobiAct V2.0: [Dataset](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2)
    - Preprocessing script: [dataset_builder.py](https://github.com/sustainable-computing/ObscureNet/blob/master/Dataset%26Models/MobiAct%20Dataset/dataset_builder.py)

- MotionSense: [Dataset](https://github.com/mmalekzadeh/motion-sense/tree/master/data)

- WiFi-HAR: [Dataset](https://data.mendeley.com/datasets/v38wjmz6f6/1)


[dataset_loader.py](https://github.com/sustainable-computing/PrivDiffuser/blob/main/dataset_loader.py): contains the code to load preprocessed dataset, change the path to your local dataset before running the notebook.


## How to Use
The jupyter notebook contians the code for obfuscating the gender attribute using the MotionSense dataset.

To use a different dataset, change `args.dataset` to `mobi` / `wifi` to use the MobiAct dataset or the WiFi-HAR dataset. `args.private` specifies the private attribute, the default value is `gender`, change to `weight` for weight obfuscation used in MobiAct or WiFi-HAR.

Due to the file size limit, we compressed the datasets, pre-trained models, and evaluation models into a zip file (DatasetsAndModels.zip) and uploaded to an anonymous Google Drive: https://drive.google.com/file/d/1168ZSbA4CjzZ8YLkGr-wE-u-gBVfV9jN/view?usp=sharing

After downloading the zip file, unzip to get 3 folders named `eval_models`, `datasets`, and `models`. Then move them to the root directly and run the Jupyter Notebook. 
The default path in the notebook should point to the corresponding models and datasets correctly.


`eval_models`: contains pre-trained evaluation models.

`models`: saves trained models, pre-trained model checkpoints included.

`datasets`: contains pre-processed datasets for MobiAct, MotionSense, and WiFi-HAR.

`dataset_loader.py`: contains the code to load the pre-processed datasets.


## Dependencies
| Package           | Version       |
| :----------------:|:-------------:| 
| python3           | 3.8.18        |
| datasets          | 3.0.1         |
| matplotlib        | 3.3.4         |
| numpy             | 1.22.0        |
| pandas            | 1.1.4         |
| pytorch_lightning | 2.3.3         |
| scikit_learn      | 1.5.2         |
| tensorflow        | 2.11.0        |
| torch             | 2.2.0         |
| torchvision       | 0.17.0        |
| tqdm              | 4.66.1        |


## Acknowledgement
- [guided-diffusion](https://github.com/openai/guided-diffusion): OpenAI's implementation for guided diffusion models.
- [mine-pytorch](https://github.com/gtegner/mine-pytorch): a PyTorch implementation for MINE (Mutual Information Neural Estimation).