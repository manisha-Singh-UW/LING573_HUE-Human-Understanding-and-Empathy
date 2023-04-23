#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/22-23/575j/env # the environment from Ling 575j can be reused for this shell file

# preprocessing is not requried on patas / condor since the embedding files have been provided
# python src/preprocess.py

# modeling is not reqired on patas / condor since the pytorch model along with its weights have been provided
# python src/model_empathy.py
# python src/model_distress.py

python src/inference_empathy.py
python src/inference_distress.py

python src/eval.py
