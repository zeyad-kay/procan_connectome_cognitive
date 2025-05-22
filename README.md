## Setup

1. Initialize a virtual environment
```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
```
2. Install dependencies
```sh
$ pip install -r requirements.txt
$ pip install -e .
```
3. Create a `.env` file and enter Weights and Biases credentials and the rest of environment variables:
```
WANDB_API_KEY=<API_KEY>
DATA_PATH=<path to dataset>
HYDRA_FULL_ERROR=1
```
4. Initialize and run hyperparameter sweep, make sure you create a project on Weights and Biases before runnning the commands
```sh
$  wandb sweep configs/sweep.yaml
...
wandb: Creating sweep from: configs\sweep.yaml
wandb: Creating sweep with ID: <SWEEP_ID>
...

$ wandb agent --count 1 <WANDB_USERNAME>/<WANDB_PROJECT_NAME>/<SWEEP_ID>
```