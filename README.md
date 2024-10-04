# PrivDiffuser
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/sustainable-computing/PrivDiffuser/blob/main/LICENSE)

This repository contains the implementation of the paper entitled "PrivDiffuser: Privacy-Guided Diffusion Model for Data Obfuscation in Sensor Networks".


## Datasets
PrivDiffuser is evaluated on three Human Activity Recognition (HAR) datasets: MotionSense, MobiAct, and WiFi-HAR. 

The datasets and the preprocessing script (required for MobiAct) are available at:

- MobiAct V2.0: [Dataset](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2)
    - Preprocessing script: [dataset_builder.py](https://github.com/sustainable-computing/ObscureNet/blob/master/Dataset%26Models/MobiAct%20Dataset/dataset_builder.py)

- MotionSense: [Dataset](https://github.com/mmalekzadeh/motion-sense/tree/master/data)

- WiFi-HAR: [Dataset](https://data.mendeley.com/datasets/v38wjmz6f6/1)


[dataset_loader.py](https://github.com/sustainable-computing/PrivDiffuser/blob/main/dataset_loader.py): contains the code to load preprocessed dataset, change the path to your local dataset before running the notebook.

[Eval_Models](https://github.com/sustainable-computing/PrivDiffuser/tree/main/Eval_Models): contains pre-trained evaluation models.


## How to Use
The jupyter notebook contians the code for obfuscating the gender attribute using the MotionSense dataset.

To use a different dataset, change `args.dataset` to `mobi` / `wifi` to use the MobiAct dataset or the WiFi-HAR dataset. `args.private` specifies the private attribute, the default value is `gender`, change to `weight` for weight obfuscation used in MobiAct or WiFi-HAR.


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
