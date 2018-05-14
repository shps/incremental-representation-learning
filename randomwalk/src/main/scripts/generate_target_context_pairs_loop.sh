#!/bin/bash

# target-context pair generator config
RW_JAR_FILE=/home/ubuntu/hooman/rw/randomwalk-0.0.1-SNAPSHOT.jar

DELIMITER="\\t"    # e.g., space-separated ("\ "), or comma-separated (",").
WINDOW_SIZE=8
SKIP_SIZE=16
SELF_CONTEXT=false  # whether allows target == context pairs.


# random walk configurations for naming conventions
METHODS=(m1 m2 m3 m4)
INIT_EDGE_SIZE=0.5
STREAM_SIZE=0.001
NUM_WALKS_ARR=(1 5 10 15 20)
WALK_LENGTH_ARR=(5 10 15 20)
P=0.25
Q=0.25
DATASET=cora
MAX_STEPS=10
NUM_RUNS=4   # counting from zero

# You can customize the JVM memory size by modifying -Xms.
# To run the script on the background: nohup sh generate_target_context_pairs.sh > log.txt &

PAIR_FILE="gPairs-w$WINDOW_SIZE-s$SKIP_SIZE"
VOCAB_FILE="gPairs-vocabs-w$WINDOW_SIZE-s$SKIP_SIZE"


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
                    EXPERIMENT_TYPE="$METHOD_TYPE-$CONFIG-$STEP-$RUN"
                    RW_FILE="sca-$EXPERIMENT_TYPE.txt"
                    DIR_NAME="$METHOD_TYPE-is$INIT_EDGE_SIZE-$CONFIG-p$P-q$Q-ss$STREAM_SIZE"
                    INPUT_EDGE_LIST="/home/ubuntu/hooman/output/$DATASET/$DIR_NAME/$RW_FILE"
                    OUTPUT_DIR="/home/ubuntu/hooman/output/$DATASET/$DIR_NAME/"

                    java -Xmx100g -Xms40g -jar $RW_JAR_FILE  --cmd gPairs --input $INPUT_EDGE_LIST --output $OUTPUT_DIR \
                        --d "$DELIMITER"  --w2vWindow $WINDOW_SIZE --w2vSkip $SKIP_SIZE \
                        --selfContext $SELF_CONTEXT

                    mv "$OUTPUT_DIR$PAIR_FILE.txt" "$OUTPUT_DIR$PAIR_FILE-$EXPERIMENT_TYPE.txt"
                    mv "$OUTPUT_DIR$VOCAB_FILE.txt" "$OUTPUT_DIR$VOCAB_FILE-$EXPERIMENT_TYPE.txt"
                done
            done
        done
    done
done