#!/bin/bash

############
# settings #
############

### EP_GP 

# MODEL="ep_gp"
# N_EPOCHS=1000
# LR=0.01

### SVI_GP 

# MODEL="svi_gp"
# VARIATIONAL_DISTRIBUTION="cholesky"
# VARIATIONAL_STRATEGY="default"
# BATCH_SIZE=100
# N_EPOCHS=2000
# LR=0.01
# N_POSTERIOR_SAMPLES=100

### SVI_BNN 

MODEL="svi_bnn"
LIKELIHOOD="binomial"
ARCHITECTURE="3L"
BATCH_SIZE=100
N_EPOCHS=3000
LR=0.01
N_HIDDEN=30
N_POSTERIOR_SAMPLES=100

#######
# run #
#######

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"


if [ $MODEL = "ep_gp" ]; then

	python EP_GPs/train.py --n_epochs=$N_EPOCHS --lr=$LR >> $OUT 2>&1

elif [ $MODEL = "svi_gp" ]; then

	python SVI_GPs/train.py --variational_distribution=$VARIATIONAL_DISTRIBUTION \
		--variational_strategy=$VARIATIONAL_STRATEGY --batch_size=$BATCH_SIZE --n_epochs=$N_EPOCHS --lr=$LR \
		--n_posterior_samples=$N_POSTERIOR_SAMPLES >> $OUT 2>&1

elif [ $MODEL = "svi_bnn" ]; then

	python SVI_BNNs/train.py --architecture=$ARCHITECTURE --batch_size=$BATCH_SIZE --n_epochs=$N_EPOCHS --lr=$LR \
		--n_hidden=$N_HIDDEN --n_posterior_samples=$N_POSTERIOR_SAMPLES >> $OUT 2>&1

fi