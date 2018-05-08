#!/bin/bash

TENSORFLOW_BIN_DIR=/home/ubuntu/hooman/tf/bin/
# your-dir/bin/
N2V_SCRIPT_DIR=/home/ubuntu/hooman/n2v/
#your-dir/

# N2V parameters
TRAIN_SPLIT=1.0             # train validation split

METHOD=m1
CONFIG="wl10-nw1"
DIR_SUFFIX="$METHOD-is0.5-$CONFIG-p0.25-q0.25-ss0.0001"
SUFFIX="$METHOD-$CONFIG-0-0"
FILE_SUFFIX="w8-s4-$SUFFIX"
BASE_LOG_DIR="/home/ubuntu/hooman/output/$DIR_SUFFIX/emb"               # base directory for logging and saving embeddings
INPUT_DIR="/home/ubuntu/hooman/output/$DIR_SUFFIX/emb/"                  # input data directory
EMB_FILE="embeddings0.pkl"                 # embeddings file name
LABEL_FILE=labels.txt           # label file
DEGREES_FILE="degrees-$SUFFIX.txt"       # node degrees file name
DELIMITER="\\t"
FORCE_OFFSET=-1                      # Offset to adjust node IDs
SEED=1234

source $TENSORFLOW_BIN_DIR/activate tensorflow
cd $N2V_SCRIPT_DIR

COMMAND="-m ml_classifier --base_log_dir $BASE_LOG_DIR --input_dir $INPUT_DIR --emb_file $EMB_FILE --degrees_file $DEGREES_FILE --delimiter $DELIMITER --force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT --label_file $LABEL_FILE"

echo $COMMAND

python3 $COMMAND

deactivate