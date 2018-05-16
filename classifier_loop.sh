#!/bin/bash

# target-context pair generator config
WINDOW_SIZE=8
SKIP_SIZE=16

# random walk configurations for naming conventions
METHODS=(m1 m3)
INIT_EDGE_SIZE=0.5
STREAM_SIZE=0.001
NUM_WALKS_ARR=(5)
WALK_LENGTH_ARR=(5)
P=0.25
Q=0.25
DATASET=cora
MAX_STEPS=5
NUM_RUNS=4   # counting from zero

# Tensorflow configurations
TENSORFLOW_BIN_DIR=/home/ubuntu/hooman/tf/bin/
N2V_SCRIPT_DIR=/home/ubuntu/hooman/n2v/

# N2V parameters
TRAIN_SPLIT=1.0             # train validation split
EMBEDDING_SIZE=128
N_EPOCHS=4                  # starts from zero
DELIMITER="\\t"
FORCE_OFFSET=0                      # Offset to adjust node IDs
SEED=1234

LABELS_DIR=/home/ubuntu/hooman/dataset/cora/
LABEL_FILE=cora_labels.txt           # label file

source $TENSORFLOW_BIN_DIR/activate tensorflow
cd $N2V_SCRIPT_DIR

trap "exit" INT

for METHOD_TYPE in ${METHODS[*]}
do
    for NUM_WALKS in ${NUM_WALKS_ARR[*]}
    do
        for WALK_LENGTH in ${WALK_LENGTH_ARR[*]}
        do
            for RUN in $(seq 0 $NUM_RUNS)
            do
                for STEP in $(seq 0 $MAX_STEPS)
                do
                    for EPOCH in $(seq 0 $N_EPOCHS)
                    do
                        printf "Run generator for method type %s\n" $METHOD_TYPE
                        printf "    Num Walks: %s\n" $NUM_WALKS
                        printf "    Walk Length: %s\n" $WALK_LENGTH
                        printf "    Run number: %s\n" $RUN
                        printf "    Step number: %s\n" $STEP
                        printf "    Epoch number: %s\n" $EPOCH

                        CONFIG=wl$WALK_LENGTH-nw$NUM_WALKS
                        SUFFIX="$METHOD_TYPE-$CONFIG-$STEP-$RUN"
                        DIR_SUFFIX="$METHOD_TYPE-is$INIT_EDGE_SIZE-$CONFIG-p$P-q$Q-ss$STREAM_SIZE"
                        BASE_LOG_DIR="/home/ubuntu/hooman/output/$DATASET/train/$DIR_SUFFIX/emb-$STEP-$RUN"
                        INPUT_DIR="/home/ubuntu/hooman/output/$DATASET/emb/$DIR_SUFFIX/emb-$STEP-$RUN"
                        DEGREES_DIR="/home/ubuntu/hooman/output/$DATASET/rw/$DIR_SUFFIX/"                  # input data directory
                        DEGREES_FILE="degrees-$SUFFIX.txt"       # node degrees file name

                        EMB_FILE="embeddings$EPOCH.pkl"                # embeddings file name
                        COMMAND="-m ml_classifier --base_log_dir $BASE_LOG_DIR --output_index $EPOCH --input_dir $INPUT_DIR --emb_file $EMB_FILE --degrees_dir $DEGREES_DIR --degrees_file $DEGREES_FILE --delimiter $DELIMITER --force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT --label_dir $LABELS_DIR --label_file $LABEL_FILE"

                        echo $COMMAND

                        python3 $COMMAND
                    done
                done
            done
        done
    done
done

deactivate