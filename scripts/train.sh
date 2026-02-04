#!/bin/bash
# Training script for probabilistic models

cd src/

python train.py --model mbpol --size 16 --n_feat 5 --include 0.999
python train.py --model mbpol --size 16 --n_feat 5 --include 0.99
python train.py --model mbpol --size 16 --n_feat 5 --include 0.98
python train.py --model scan --size 16 --n_feat 5 --include 0.999
python train.py --model scan --size 16 --n_feat 5 --include 0.99
python train.py --model scan --size 16 --n_feat 5 --include 0.98
python train.py --model mbpol --size 3 --n_feat 5 --include 0.999
python train.py --model scan --size 3 --n_feat 5 --include 0.999

cd ..
