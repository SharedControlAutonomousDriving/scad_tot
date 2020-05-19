#!/bin/zsh

# v3.2.0_train: https://drive.google.com/file/d/1GF9I0XnhztkYR1ZPNcsc6Mdq-xg2MP93/view?usp=sharing
# v3.2.0_test: https://drive.google.com/file/d/1KkoEaLUsmV916TO9FOAuGmN_uy2o9e-t/view?usp=sharing
# public folder: https://drive.google.com/drive/folders/1ZWyP6BiUhch94XKGMDs2B1L2T-KUwS_I

DATA_PATH=./data

BASE_DL_URL="https://drive.google.com/uc?authuser=0&export=download"
TEST_DATA_URL="$BASE_DL_URL&id=1KkoEaLUsmV916TO9FOAuGmN_uy2o9e-t"
TRAIN_DATA_URL="$BASE_DL_URL&id=1GF9I0XnhztkYR1ZPNcsc6Mdq-xg2MP93"

mkdir -p $DATA_PATH

# uncomment to download both training and test data.
# wget -O $DATA_PATH/train.csv $TRAIN_DATA_URL
# wget -O $DATA_PATH/test.csv $TEST_DATA_URL

wget -O $DATA_PATH/data.csv $TEST_DATA_URL
