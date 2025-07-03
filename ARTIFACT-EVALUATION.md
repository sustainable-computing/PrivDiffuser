# Artifact Appendix

Paper title: PrivDiffuser: Privacy-Guided Diffusion Model for Data Obfuscation in Sensor Networks

Artifacts HotCRP Id: 5

Requested Badge: Reproduced, Functional, and Available

## Description
This artifact provides source code, datasets, and pre-trained models for PrivDiffuser, a sensor data obfuscation model. It generates obfuscated data using PrivDiffuser and evaluates the utility loss and privacy loss using pre-trained desired inference models and intrusive inference models. This artifact supports all 3 public datasets as discussed in the paper: MobiAct, MotionSense, and WiFi-HAR.

### Security/Privacy Issues and Ethical Concerns (All badges)
No security/privacy issues and ethical concerns.

## Basic Requirements (Only for Functional and Reproduced badges)
At least 20 GB of storage (40 GB of storage if using Docker image) and 16 GB of RAM. NVIDIA GPU is recommended, but not necessary to reproduce the results. 

The setup process described in the README is tested on Ubuntu 20.04 with x86-64 architecture with Python 3.8. 

The estimated time to perform data obfuscation on all 3 datasets will take 1-2 days on a single NVIDIA RTX 2080 Ti GPU. However, the reviewer can choose to interrupt the data obfuscation process and only evaluate on a reasonable number of batches of the dataset to reduce the computational cost, as discussed in the README.

### Hardware Requirements
We recommend using a computer with x86-64 architecture. ARM architecture, such as Apple Silicon Macs, might require additional configuration for PyTorch and TensorFlow.

### Software Requirements
We recommend using Ubuntu 20.04 and Python 3.8 to evaluate our artifact. We provide all datasets and pre-trained models on Google Drive (https://drive.google.com/file/d/1Hwjhe6v0ZfoSshPA7CIXjwzST9CeSkRD/view?usp=sharing), as described in the README. 

For the ease of evaluation, we recommend using an x86-64 machine to install our Docker image (this does not support GPU acceleration). 
**Note that our Docker image already contains all datasets and pre-trained models.**

### Estimated Time and Storage Consumption
The datasets and pre-trained models require at least 10 GB of storage. The storage consumption for using our Docker image is ~35 GB. 

The estimated time for evaluating all 3 datasets would take around 2 days with GPU acceleration. Evaluating all 3 datasets on CPU would take multiple days, depending on the computational power of the CPU.
Ranking the time needed by each dataset from low to high: MotionSense < MobiAct < WiFi-HAR.

However, reviewers do not need to perform data obfuscation on the entire dataset and can interrupt the sampling process for generating obfuscated data and obtain results on a few batches of the dataset. We anticipate the results will not have significant changes compared to the results reported in the paper. 

## Environment 
In the following, describe how to access our artifact and all related and necessary data and software components.
Afterward, describe how to set up everything and how to verify that everything is set up correctly.

### Accessibility (All badges)
The source code to our implementation is available on GitHub: https://github.com/sustainable-computing/PrivDiffuser

We provide the datasets and pre-trained models via a public Google Drive: https://drive.google.com/file/d/1Hwjhe6v0ZfoSshPA7CIXjwzST9CeSkRD/view?usp=sharing

We also provide a pre-built Docker image on Docker Hub with all datasets and models included, which can be obtained by running: `docker pull neilyxin/privdiffuser`

### Set up the environment (Only for Functional and Reproduced badges)

#### Using Docker Image:
We recommend using our Docker image as it has all the necessary dependencies installed and contains all the source code, datasets, and pre-trained models. 
The Docker image is pre-configured with a `base` conda environment and will automatically launch Jupyter Lab on port `8889`.

The reviewers need to first install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on their machine.

You can pull our Docker image from the Docker Hub: `docker pull neilyxin/privdiffuser`.

Run the Docker image: `docker run -it --rm -p 8889:8889 neilyxin/privdiffuser`.

You can find the link to the Jupyter Lab with authentication token in the terminal: http://127.0.0.1:8889/lab?token=replace_with_your_token. You can paste this into your browser to open Jupyter Lab. The code base, datasets, and models are located under the default work directory. Open PrivDiffuser.ipynb to run the code.

Note: This Docker image is built for and tested on Ubuntu (20.04). Using this image on other OS or architecture, such as macOS with Apple silicon chips, may require additional setup.


#### Non-Docker Setup:
The reviewers can first clone our repo:
```bash
git clone git@github.com:sustainable-computing/PrivDiffuser.git
```

We recommend installing [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) and creating a conda environment with Python 3.8: 
```bash
conda create -n "myenv" python=3.8
```

We provide a list of dependencies with their versions in `requirements.txt`. Reviewers need to install the listed dependencies using `conda install` or `pip install`. The reviewer might refer to the official websites of TensorFlow and PyTorch to install GPU-accelerated versions.
We also provide `environment.yml` used to build our Docker image for reference.

### Testing the Environment (Only for Functional and Reproduced badges)
If the Docker image is successfully configured, running the Docker image should automatically start Jupyter Lab on port 8889. 

Similarly, if the reviewers configured their own environment, the reviewers should be able to open the Jupyter notebook provided in our GitHub repo.

The reviewers can then run all cells. The data sampling should start without error if the environment is set up correctly.

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims

#### Main Result 1: Data Obfuscation on MobiAct Dataset
The result on the MobiAct dataset is presented in Section 6.1, Table 1, and Figure 4.

When gender is the private attribute, data obfuscated by PrivDiffuser yields an average activity recognition accuracy of 97.40% (Table 1) and an average intrusive gender classification accuracy of 51.43% (Figure 4 (a)). 

When weight is the private attribute, data obfuscated by PrivDiffuser yields an average activity recognition accuracy of 97.03% (Table 1) and an average intrusive weight classification accuracy of 36.61% (Figure 4 (b)). 


#### Main Result 2: Data Obfuscation on MotionSense Dataset
The result on the MotionSense dataset is presented in Section 6.2, Table 2, and Figure 5.

When gender is the private attribute, data obfuscated by PrivDiffuser yields an average activity recognition accuracy of 96.32% (Table 2) and an average intrusive gender classification accuracy of 49.96% (Figure 5). 

#### Main Result 3: Data Obfuscation on WiFi-HAR Dataset
The result on the WiFi-HAR dataset is presented in Section 6.3, Table 3, and Figure 6.

When weight is the private attribute, data obfuscated by PrivDiffuser yields an average activity recognition accuracy of 88.18% (Table 3) and an average intrusive gender classification accuracy of 49.21% (Figure 6). 

### Experiments 

#### Experiment 1: Data Obfuscation on MobiAct Dataset
1.1 Gender Obfuscation:

Running the default Jupyter notebook will reproduce the results for gender obfuscation.

1.2 Weight Obfuscation:

Change `self.private='gender'` in the `Args` class into `self.private='weight'`, then restart the kernel and run the Jupyter notebook will reproduce the results for weight obfuscation.

The results should align with Main Result 1. 

#### Experiment 2: Data Obfuscation on MotionSense Dataset
Change `self.dataset='mobi'` in the `Args` class into `self.dataset='motion'`, set `self.private='gender'`, then restart the kernel and run the Jupyter notebook will reproduce the results for gender obfuscation on MotionSense.

The results should align with Main Result 2. 
#### Experiment 3: Data Obfuscation on WiFi-HAR Dataset 
Set `self.dataset='wifi'` and `self.private='weight'` in the `Args` class, then restart the kernel and run the Jupyter notebook will reproduce the results for weight obfuscation on WiFi-HAR.

The results should align with Main Result 3. 
## Limitations (Only for Functional and Reproduced badges)
This source code provides the main performance evaluation for PrivDiffuser on 3 public datasets, which is discussed in Section 6.1, 6.2, and 6.3. These results are critical as they demonstrate the data obfuscated by PrivDiffuser can reduce intrusive inference accuracy on the private attribute with minimum sacrifice on data utility, outperforming state-of-the-art GAN-based obfuscation models.

We do not provide the source code for Section 6.4 to 6.7, given that Section 6.4 can be achieved using our code base by adjusting the hyperparameters w1 and w2; Section 6.5 and 6.6 study the scalability of PrivDiffuser and can be implemented on top of the provided source code with minor adjustments; Section 6.7 can also be implemented on top of the provided source code by introducing another MINE module.

## Notes on Reusability (Only for Functional and Reproduced badges)
The configurations of PrivDiffuser are set in the `Args` and `DiffusionArgs` classes, where users can easily adjust. PrivDiffuser already supports 3 datasets, and the script for loading the datasets is provided in `dataset_loader.py`. Users can refer to this script to support more datasets.
