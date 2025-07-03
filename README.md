# Prediction Modeling in Transdiagnostic Risk: Results from the PROCAN study

This repo contains the code for the machine learning analysis section of the paper. This work is an extension of prior work by <a href=https://link.springer.com/article/10.1007/s11682-024-00953-z>Shakeel et al.</a> We reuse most of the code from the prior work, which can be found <a href=https://github.com/mklasby/brain-connectome-longitudinal>here</a>.

## Data

The data used in this study may be available upon request. The imaging and behavioral data span multiple csv files. The directory structure should be:
```
data/raw_data/t12-updated/
├── DTI
│   ├── DTI_ID_CV.csv
│   ├── DTI_ID_CV.sav
│   ├── Density_intensity
│   │   └── Harmzd_DTI_density_intensity.csv
│   ├── Global
│   │   └── Harmzd_DTI_global.csv
│   ├── Harmzd_DTI_Sync.csv
│   ├── Modular_interactions
│   │   └── Harmzd_DTI_MI.csv
│   └── Nodal
│       ├── Harmzd_DTI_Bc.csv
│       ├── Harmzd_DTI_Dc.csv
│       ├── Harmzd_DTI_Ncc.csv
│       ├── Harmzd_DTI_Ne.csv
│       ├── Harmzd_DTI_Nle.csv
│       ├── Harmzd_DTI_Nsp.csv
│       └── Harmzd_DTI_Pc.csv
├── cognitive
│   └── cognitive.csv
└── fMRI
    ├── Density_intensity
    │   └── Harmzd_fMRI_density_intensity.csv
    ├── Global
    │   └── Harmzd_fMRI_global.csv
    ├── Harmzd_fMRI_Sync.csv
    ├── Modular_interactions
    │   └── Harmzd_fMRI_MI.csv
    ├── Nodal
    │   ├── Harmzd_fMRI_Bc.csv
    │   ├── Harmzd_fMRI_Dc.csv
    │   ├── Harmzd_fMRI_Ncc.csv
    │   ├── Harmzd_fMRI_Ne.csv
    │   ├── Harmzd_fMRI_Nle.csv
    │   ├── Harmzd_fMRI_Nsp.csv
    │   └── Harmzd_fMRI_Pc.csv
    └── fMRI_ID_CV_HM.csv
```

## Setup

1. Initialize a virtual environment:
```sh
$ python3 -m venv .procan
$ source .procan/bin/activate
```
2. Install dependencies:
```sh
$ pip install -r requirements.txt
$ pip install -e .
```
3. Create a `.env` file and enter Weights and Biases credentials and the rest of environment variables:
```
WANDB_API_KEY=<WANDB_API_KEY>
WANDB_PROJECT=<WANDB_PROJECT_NAME>
WANDB_ENTITY=<WANDB_ENTITY>
BASE_PATH=<PATH_TO_PROJECT>
HYDRA_FULL_ERROR=1
```
4. Load environment variables
```sh
$ source .env
```
5. Initialize and run hyperparameter sweep, make sure you create a project on Weights and Biases before runnning the commands:
```sh
$ source init_sweep.sh <SWEEP_NAME>
...
wandb: Creating sweep with ID: <SWEEP_ID>
...
$ source run_sweep.sh <SWEEP_ID>
```
5. Download sweep results from wandb:
```sh
$ python procan_connectome/utils/download_wandb_run_table.py
```   
6. Summarize the results and generate figures by running the `sweep_summary.ipynb` notebook.
