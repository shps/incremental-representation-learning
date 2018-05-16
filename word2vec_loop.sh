#!/bin/bash

# target-context pair generator config
WINDOW_SIZE=8
SKIP_SIZE=16
PAIR_FILE="gPairs-w$WINDOW_SIZE-s$SKIP_SIZE"

# random walk configurations for naming conventions
METHODS=(m1)
INIT_EDGE_SIZE=0.5
STREAM_SIZE=0.001
NUM_WALKS_ARR=(1)
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
LEARNING_RATE=0.2
EMBEDDING_SIZE=128
VOCAB_SIZE=2708            # Size of vocabulary
NEG_SAMPLE_SIZE=5
N_EPOCHS=5
BATCH_SIZE=200               # minibatch size
FREEZE_EMBEDDINGS=True     #If true, the embeddings will be frozen otherwise the contexts will be frozen.
DELIMITER="\\t"
FORCE_OFFSET=0                      # Offset to adjust node IDs
SEED=1234


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
                    printf "Run generator for method type %s\n" $METHOD_TYPE
                    printf "    Num Walks: %s\n" $NUM_WALKS
                    printf "    Walk Length: %s\n" $WALK_LENGTH
                    printf "    Run number: %s\n" $RUN
                    printf "    Step number: %s\n" $STEP

                    CONFIG=wl$WALK_LENGTH-nw$NUM_WALKS
                    SUFFIX="$METHOD_TYPE-$CONFIG-$STEP-$RUN"
                    FILE_SUFFIX="w$WINDOW_SIZE-s$SKIP_SIZE-$SUFFIX"
                    DIR_SUFFIX="$METHOD_TYPE-is$INIT_EDGE_SIZE-$CONFIG-p$P-q$Q-ss$STREAM_SIZE"
                    BASE_LOG_DIR="/home/ubuntu/hooman/output/$DATASET/emb/$DIR_SUFFIX/emb-$STEP-$RUN"
                    INPUT_DIR="/home/ubuntu/hooman/output/$DATASET/rw/$DIR_SUFFIX/"                  # input data directory
                    TRAIN_FILE="gPairs-$FILE_SUFFIX.txt"                 # train file name
                    DEGREES_FILE="degrees-$SUFFIX.txt"       # node degrees file name

                    COMMAND="-m node2vec_pregen --base_log_dir $BASE_LOG_DIR --input_dir $INPUT_DIR --train_file $TRAIN_FILE --degrees_file $DEGREES_FILE --delimiter $DELIMITER --force_offset $FORCE_OFFSET --seed $SEED --train_split $TRAIN_SPLIT --learning_rate $LEARNING_RATE --embedding_size $EMBEDDING_SIZE --vocab_size $VOCAB_SIZE --neg_sample_size $NEG_SAMPLE_SIZE --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --freeze_embeddings $FREEZE_EMBEDDINGS"

                    if [ "$METHOD_TYPE" != "m1" ] && [ $STEP -gt 0 ]; then
                        AFFECTED_VERTICES_FILE="$sca-afs-$SUFFIX.txt"     # affected vertices file name
                        COMMAND="$COMMAND --affected_vertices_file $AFFECTED_VERTICES_FILE"
                    fi

                    if [ $STEP -gt 0 ]; then
                        COMMAND="$COMMAND --checkpoint_file model-epoch-$(($N_EPOCHS-1)) --checkpoint_dir /home/ubuntu/hooman/output/$DATASET/emb/$DIR_SUFFIX/emb-$(($STEP-1))-$RUN"
                    fi

                    echo $COMMAND

                    python3 $COMMAND
                done
            done
        done
    done
done

deactivate