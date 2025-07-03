# Prediction Modeling in Transdiagnostic Risk: Results from the PROCAN study

This repo contains the code for the machine learning analysis section of the paper. This work is an extension of previously published work by <a href=https://link.springer.com/article/10.1007/s11682-024-00953-z>Shakeel et al.</a> We reuse most of the code from the previous work, which can be found <a href=https://github.com/mklasby/brain-connectome-longitudinal>here</a>.

## Setup

The data used in this study may be available upon request.

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
