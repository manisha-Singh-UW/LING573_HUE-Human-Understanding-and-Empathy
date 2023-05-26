#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /nopt/dropbox/22-23/575j/env # the environment from Ling 575j can be reused for this shell file

#SVR training
#python src/ensemble_svr_train.py #does not need to be run -- produces pickled SVR models included in models folder

#devtest inference
python src/ensemble_inference.py empathy outputs/D4/primary/devtest/D4 dev
python src/ensemble_inference.py distress outputs/D4/primary/devtest/D4 dev

#devtest evaluation
python src/ensemble_eval.py outputs/D4/primary/devtest/D4 results/D4/primary/devtest/D4

#evaltest inference
python src/ensemble_inference.py empathy outputs/D4/primary/evaltest/D4 test
python src/ensemble_inference.py distress outputs/D4/primary/evaltest/D4 test

#see readme for evaltest evaluation
