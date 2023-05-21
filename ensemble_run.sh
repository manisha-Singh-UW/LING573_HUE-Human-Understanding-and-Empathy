#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /nopt/dropbox/22-23/575j/env # the environment from Ling 575j can be reused for this shell file

#python src/ensemble_svr_train.py

python src/ensemble_inference.py empathy outputs/ensemble

python src/ensemble_inference.py distress outputs/ensemble

python src/ensemble_eval.py outputs/ensemble results/ensemble
