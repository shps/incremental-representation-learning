#!/bin/bash

TENSORFLOW_BIN_DIR=/Users/Ganymedian/Desktop/tensorflow/bin
# your-dir/bin/
N2V_SCRIPT_DIR=/Users/Ganymedian/Desktop/Projects/node2vec_experiments
#your-dir/

# N2V parameters
TRAIN_SPLIT=0.8             # train validation split
LEARNING_RATE=0.2
EMBEDDING_SIZE=20
VOCAB_SIZE=10400            # Size of vocabulary
NEG_SAMPLE_SIZE=2
N_EPOCHS=10
BATCH_SIZE=20               # minibatch size
FREEZE_EMBEDDINGS=True      #If true, the embeddings will be frozen otherwise the contexts will be frozen.

BASE_LOG_DIR=your-dir               # base directory for logging and saving embeddings
INPUT_DIR=your-dir                  # input data directory
TRAIN_FILE=blah.txt                 # train file name
LABEL_FILE=""           # label file
DEGREES_FILE=degrees_file.txt       # node degrees file name
CHECKPOINT_FILE=""                    # tf checkpoint file name
AFFECTED_VERTICES_FILE=blah.txt     # affected vertices file name
DELIMITER="\\t"
FORCE_OFFSET=0                      # Offset to adjust node IDs
SEED=58125312


source $TENSORFLOW_BIN_DIR/activate tensorflow
cd $N2V_SCRIPT_DIR


COMMAND="-m node2vec_pregen --base_log_dir $BASE_LOG_DIR --input_dir $INPUT_DIR --train_file $TRAIN_FILE --degrees_file $DEGREES_FILE --affected_vertices_file $AFFECTED_VERTICES_FILE --delimiter $DELIMITER --force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT --learning_rate $LEARNING_RATE --embedding_size $EMBEDDING_SIZE --vocab_size $VOCAB_SIZE --neg_sample_size $NEG_SAMPLE_SIZE --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --freeze_embeddings $FREEZE_EMBEDDINGS"

if [ "$LABEL_FILE" ne "" ]; then
        COMMAND="$COMMAND --label_file $LABEL_FILE"
fi

if [ "$CHECKPOINT_FILE" ne "" ]; then
        COMMAND="$COMMAND --checkpoint_file $CHECKPOINT_FILE"
fi

echo $COMMAND

python $COMMAND


deactivate