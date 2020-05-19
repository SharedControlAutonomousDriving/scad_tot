#!/bin/zsh

python3 verification/sensitivity.py -m ./network/models/v3.2.0/model.h5 -d ./data/data.csv -df 0.01 -dmin 0.00001 -dmax 100 -mt -sr -ss -sl -v 1
