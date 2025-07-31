# PrivDiffuser
[![PoPETS](https://img.shields.io/badge/PoPETs-2025-blue?style=flat)]()
[![arXiv](https://img.shields.io/badge/arXiv-2209.12046-b31b1b.svg)](https://arxiv.org/abs/2412.14499)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/sustainable-computing/PrivDiffuser/blob/main/LICENSE)

This repository contains the implementation of the paper entitled "PrivDiffuser: Privacy-Guided Diffusion Model for Data Obfuscation in Sensor Networks.


## Datasets
PrivDiffuser is evaluated on three Human Activity Recognition (HAR) datasets: MotionSense, MobiAct, and WiFi-HAR. 

The datasets and the preprocessing script (required for MobiAct) are available at:

- MobiAct V2.0: [Dataset](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2), Preprocessing script: [dataset_builder.py](https://github.com/sustainable-computing/ObscureNet/blob/master/Dataset%26Models/MobiAct%20Dataset/dataset_builder.py)

- MotionSense: [Dataset](https://github.com/mmalekzadeh/motion-sense/tree/master/data)

- WiFi-HAR: [Dataset](https://data.mendeley.com/datasets/v38wjmz6f6/1)

We provide the datasets, pre-trained models, and evaluation models using Git LFS. You need to first [install Git LFS](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing).

After Git LFS is installed, you can clone this repo and enter the working directory: `cd PrivDiffuser`, then run `git lfs fetch --all` to download the 3 folders: `eval_models`, `datasets`, and `models`. They should be placed under the root directory of this repo.

The notebook should load the corresponding models and datasets correctly by default.

## Setup
We provided `requirements.txt` for dependencies installed via pip. In addition, we provided `environment.yml` for conda environments. 

Note: the `environment.yml` is generated for building our Docker image, hence it does NOT install GPU-related packages. You may need to install GPU-enabled PyTorch and TensorFlow if you want to use GPU acceleration.

### Docker
We provided a pre-built Docker image that contains our code, datasets, pre-trained models, and all required dependencies to run the code (without GPU acceleration). The Docker image is pre-configured with a `base` conda environment and will automatically launch Jupyter Lab on port `8889`.

You can pull our Docker image from the Docker Hub: `docker pull neilyxin/privdiffuser`.

Run the Docker image: `docker run -it --rm -p 8889:8889 neilyxin/privdiffuser`. 

You can find the link to the Jupyter Lab with authentication token in the terminal: `http://127.0.0.1:8889/lab?token=replace_with_your_token`. You can paste this into your browser to open Jupyter Lab. The code base, datasets, and models are located under the default work directory. Open `PrivDiffuser.ipynb` to run the code.

Note: This Docker image is built for and tested on Ubuntu (20.04). Using this image on other OS or architecture, such as macOS with Apple silicon chips, may require additional setup. 

## How to Use
The Jupyter notebook contains the code for obfuscating the gender attribute using the MobiAct dataset.

To use a different dataset, change `args.dataset` to `mobi` / `motion` / `wifi` to use the MobiAct/MotionSense/WiFi-HAR dataset. `args.private` specifies the private attribute, the default value is `gender`, change to `weight` for weight obfuscation used in MobiAct or WiFi-HAR. 

Below we list the private attribute(s) supported in each dataset:

| Dataset         | Supported Private Attribute |
| :--------------:|:---------------------------:| 
| MobiAct         | gender, weight              |
| MotionSense     | gender                      |
| WiFi-HAR        | weight                      |

`PrivDiffuser.ipynb`: jupyter notebook for running the PrivDiffuser code base.

`eval_models`: contains pre-trained evaluation models.

`models`: saves trained models, pre-trained model checkpoints included.

`datasets`: contains pre-processed datasets for MobiAct, MotionSense, and WiFi-HAR.

`dataset_loader.py`: contains the code to load the pre-processed datasets. If you placed the datasets in a different directory, you may need to change the path to the datasets here.


### Reproduce Results on MobiAct
The default configuration in `PrivDiffuser.ipynb` will load the pre-trained models under the `models` folder to perform gender obfuscation on the MobiAct dataset. It will generate obfuscated data and evaluate data utility and privacy.
Running the default notebook will generate results for PrivDiffuser in Table 1 and Figure 4 (a). 

Change `self.private = 'gender'` into `self.private = 'weight'` in the `Args` class, then re-run the notebook to obtain weight obfuscation results on MobiAct, as presented in Table 1 and Figure 4 (b).

### Reproduce Results on MotionSense
To reproduce the results on the MotionSense dataset using pre-trained models, as presented in Table 2 and Figure 5, set `self.dataset='motion'` and `self.private='gender'` in the `Args` class, then re-run the notebook.

### Reproduce Results on MotionSense
To reproduce the results on the Wifi-HAR dataset using pre-trained models, as presented in Table 3 and Figure 6, set `self.dataset='motion'` and `self.private='gender'` in the `Args` class, then re-run the notebook.

Note: the sampling process can be interrupted if at least one batch of obfuscated data is generated. Running the remaining code after the interruption will report the data obfuscation performance on this generated portion of the test set.

### Reproduce Results for Diffusion baseline and Diffusion with Negation baseline
- Diffusion baseline: set `w2=0` and re-run the sampling to disable negative conditioning.
- Diffusion with Negation baseline: set `w3=0` (w2 should be consistent as used in PrivDiffuser), change `train_priv = False` to `train_priv = True`, re-run this cell to train the auxiliary privacy model, and then re-run the remaining cells for sampling. This disables MI-based regularization while still using the negative conditioning.





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
