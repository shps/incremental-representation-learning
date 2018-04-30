#!/bin/bash

TENSORFLOW_BIN_DIR=
# your-dir/bin/
N2V_SCRIPT_DIR=./
#your-dir/

# N2V parameters
TRAIN_SPLIT=0.8             # train validation split
LEARNING_RATE=0.2
EMBEDDING_SIZE=20
VOCAB_SIZE=34            # Size of vocabulary
NEG_SAMPLE_SIZE=2
N_EPOCHS=4
BATCH_SIZE=20               # minibatch size
FREEZE_EMBEDDINGS=False      #If true, the embeddings will be frozen otherwise the contexts will be frozen.

BASE_LOG_DIR=./out               # base directory for logging and saving embeddings
INPUT_DIR=./karate                  # input data directory
TRAIN_FILE=gPairs-w3-s6.txt                 # train file name
LABEL_FILE=karate-labels.txt           # label file
DEGREES_FILE=karate-degree.txt       # node degrees file name
CHECKPOINT_FILE=""                    # tf checkpoint file name
AFFECTED_VERTICES_FILE=""     # affected vertices file name
DELIMITER="\t"
FORCE_OFFSET=-1                      # Offset to adjust node IDs
SEED=58125312

if [ "$TENSORFLOW_BIN_DIR" != "" ]; then
    source $TENSORFLOW_BIN_DIR/activate tensorflow
fi

cd $N2V_SCRIPT_DIR


COMMAND="node2vec_pregen.py --base_log_dir $BASE_LOG_DIR --input_dir $INPUT_DIR
--train_file $TRAIN_FILE --degrees_file $DEGREES_FILE  --delimiter $DELIMITER
--force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT
--learning_rate $LEARNING_RATE --embedding_size $EMBEDDING_SIZE
--vocab_size $VOCAB_SIZE --neg_sample_size $NEG_SAMPLE_SIZE
--n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --freeze_embeddings $FREEZE_EMBEDDINGS"

if [ "$LABEL_FILE" != "" ]; then
        COMMAND="$COMMAND --label_file $LABEL_FILE"
fi

if [ "$CHECKPOINT_FILE" != "" ]; then
        COMMAND="$COMMAND --checkpoint_file $CHECKPOINT_FILE"
fi

echo $COMMAND

python $COMMAND

if [ "$TENSORFLOW_BIN_DIR" != "" ]; then
    source $TENSORFLOW_BIN_DIR/deactivate
fi
