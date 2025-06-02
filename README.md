# Prediction Modeling in Transdiagnostic Risk: Results from the PROCAN study

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
5. Summarize the results by running the `sweep_summary.ipynb` notebook.