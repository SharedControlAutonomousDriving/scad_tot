#!/bin/zsh

# v3.2.0_train: https://drive.google.com/file/d/1GF9I0XnhztkYR1ZPNcsc6Mdq-xg2MP93/view?usp=sharing
# v3.2.0_test: https://drive.google.com/file/d/1KkoEaLUsmV916TO9FOAuGmN_uy2o9e-t/view?usp=sharing

DATA_PATH=/home/marabou/scad_tot/data
TRAIN_DATA_URL=https://drive.google.com/file/d/1GF9I0XnhztkYR1ZPNcsc6Mdq-xg2MP93
TEST_DATA_URL=https://drive.google.com/file/d/1KkoEaLUsmV916TO9FOAuGmN_uy2o9e-t

mkdir -p $DATA_PATH
wget -O $DATA_PATH/data/test.csv $TEST_DATA_URL
wget -O $DATA_PATH/data/train.csv $TRAIN_DATA_URL
