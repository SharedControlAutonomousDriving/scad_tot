#!/bin/zsh

# run symmetric local robustness test
python3 verification/robustness.py -n ./network/models/v3.2.0/model.nnet -d ./data/v3.2.0/test.csv -df 0.01 -emin 0.00001 -emax 100 -eprec 0.000001 -sr -ss -sl -v 1 -o ./logs/robustness/symmetric -ck

