#!/bin/bash

source .env
wandb sweep configs/sweep.yaml -e ${WANDB_ENTITY} -p ${WANDB_PROJECT} --name ${1}
