#!/bin/bash

TENSORFLOW_BIN_DIR=your-dir/bin/
N2V_SCRIPT_DIR=your-dir/

# N2V parameters
TRAIN_SPLIT=0.8             # train validation split
LEARING_RATE=0.2
EMBEDDING_SIZE=20
VOCAB_SIZE=10400            # Size of vocabulary
NEG_SAMPLE_SIZE=2
N_EPOCHS=10
BATCH_SIZE=20               # minibatch size
FREEZE_EMBEDDINGS=True      #If true, the embeddings will be frozen otherwise the contexts will be frozen.

BASE_LOG_DIR=your-dir               # base directory for logging and saving embeddings
INPUT_DIR=your-dir                  # input data directory
TRAIN_FILE=blah.txt                 # train file name
LABEL_FILE=label_file.txt           # label file
DEGREES_FILE=degrees_file.txt       # node degrees file name
CHECKPOINT_FILE=                    # tf checkpoint file name
AFFECTED_VERTICES_FILE=blah.txt     # affected vertices file name
DELIMITER="\\t"
FORCE_OFFSET=0                      # Offset to adjust node IDs
SEED=58125312


source $TENSORFLOW_BIN_DIR/activate tensorflow  --base_log_dir $BASE_LOG_DIR --input_dir $INPUT_DIR \
    --train_file $TRAIN_FILE --label_file $LABEL_FILE --degrees_file $DEGREES_FILE \
    --checkpoint_file $CHECKPOINT_FILE --affected_vertices_file $AFFECTED_VERTICES_FILE \
    --delimiter $DELIMITER --force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT \
    --LEARING_RATE $learning_rate --embedding_size $EMBEDDING_SIZE --vocab_size $VOCAB_SIZE \
    --neg_sample_size $NEG_SAMPLE_SIZE --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE \
    --freeze_embeddings $FREEZE_EMBEDDINGS




python -m $N2V_SCRIPT_DIR/node2vec_pregen