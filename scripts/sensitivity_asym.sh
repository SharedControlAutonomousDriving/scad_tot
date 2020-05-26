#!/bin/zsh

# run asymmetric sensitivity test
python3 verification/sensitivity.py -n ./network/models/v3.2.0/model.nnet -d ./data/v3.2.0/test.csv -df 0.01 -emin 0.00001 -emax 1000 -eprec 0.000001 -a -sr -ss -sl -v 1 -o ./logs/sensitivity/asymmetric
