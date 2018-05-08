#!/bin/bash

TENSORFLOW_BIN_DIR=/home/ubuntu/hooman/tf/bin/
N2V_SCRIPT_DIR=/home/ubuntu/hooman/n2v/

# N2V parameters
TRAIN_SPLIT=1.0             # train validation split
LEARNING_RATE=0.2
EMBEDDING_SIZE=128
VOCAB_SIZE=10400            # Size of vocabulary
NEG_SAMPLE_SIZE=5

N_EPOCHS=5
BATCH_SIZE=20               # minibatch size
FREEZE_EMBEDDINGS=True     #If true, the embeddings will be frozen otherwise the contexts will be frozen.
METHOD=m1
CONFIG="wl10-nw1"
DIR_SUFFIX="$METHOD-is0.5-$CONFIG-p0.25-q0.25-ss0.0001"
SUFFIX="$METHOD-$CONFIG-0-0"
FILE_SUFFIX="w8-s4-$SUFFIX"
BASE_LOG_DIR="/home/ubuntu/hooman/output/$DIR_SUFFIX/emb"               # base directory for logging and saving embeddings

INPUT_DIR="/home/ubuntu/hooman/output/$DIR_SUFFIX/"                  # input data directory
TRAIN_FILE="gPairs-$FILE_SUFFIX.txt"                 # train file name
DEGREES_FILE="degrees-$SUFFIX.txt"       # node degrees file name
CHECKPOINT_FILE=""                    # tf checkpoint file name
AFFECTED_VERTICES_FILE="gPairs-vocabs-$FILE_SUFFIX.txt"     # affected vertices file name
DELIMITER="\\t"
FORCE_OFFSET=-1                      # Offset to adjust node IDs
SEED=1234

source $TENSORFLOW_BIN_DIR/activate tensorflow
cd $N2V_SCRIPT_DIR

COMMAND="-m node2vec_pregen --base_log_dir $BASE_LOG_DIR --input_dir $INPUT_DIR --train_file $TRAIN_FILE --degrees_file $DEGREES_FILE --affected_vertices_file $AFFECTED_VERTICES_FILE --delimiter $DELIMITER --force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT --learning_rate $LEARNING_RATE --embedding_size $EMBEDDING_SIZE --vocab_size $VOCAB_SIZE --neg_sample_size $NEG_SAMPLE_SIZE --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --freeze_embeddings $FREEZE_EMBEDDINGS"

if [ "$LABEL_FILE" != "" ]; then
        COMMAND="$COMMAND --label_file $LABEL_FILE"
fi

if [ "$CHECKPOINT_FILE" != "" ]; then
        COMMAND="$COMMAND --checkpoint_file $CHECKPOINT_FILE"
fi

echo $COMMAND

python3 $COMMAND

deactivate