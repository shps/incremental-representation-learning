#!/bin/bash

RW_JAR_FILE=/home/ubuntu/hooman/rw/randomwalk-0.0.1-SNAPSHOT.jar
INPUT_EDGE_LIST=/home/ubuntu/hooman/dataset/cora/cora_edgelist.txt
INIT_EDGE_SIZE=0.5
METHOD_TYPE="m3"
NUM_WALKS=10
WALK_LENGTH=10
P=0.25
Q=0.25
STREAM_SIZE=0.001
DATASET=cora
OUTPUT_DIR="/home/ubuntu/hooman/output/$DATASET/$METHOD_TYPE-is$INIT_EDGE_SIZE-wl$WALK_LENGTH-nw$NUM_WALKS-p$P-q$Q-ss$STREAM_SIZE/"
NUM_RUNS=1
DIRECTED=true    # tested on undirected graphs only.
SEED=1234
WALK_TYPE=secondorder
DELIMITER="\\s+"    # e.g., tab-separated ("\\t"), or comma-separated (",").
LOG_PERIOD=1      # after what number of steps log the output
LOG_ERRORS=false  # Should it compute and log transition probability errors (computation intensive)   # portion of edges to be used for streaming at each step
MAX_STEPS=5        # max number of steps to run the experiment
GROUPED=false         # whether the edge list is already tagged with group number (e.g., year)

# You can customize the JVM memory size by modifying -Xms.
# To run the script on the background: nohup sh random_walk.sh > log.txt &
#-Xmsg
java -Xmx100g -Xms40g -jar $RW_JAR_FILE  --cmd sca --walkLength $WALK_LENGTH --numWalks $NUM_WALKS \
    --input $INPUT_EDGE_LIST --output $OUTPUT_DIR --nRuns $NUM_RUNS --directed $DIRECTED --p $P \
    --q $Q --seed $SEED --d "$DELIMITER" --rrType $METHOD_TYPE --wType $WALK_TYPE --save $LOG_PERIOD \
    --logErrors $LOG_ERRORS --initEdgeSize $INIT_EDGE_SIZE --edgeStreamSize $STREAM_SIZE \
    --maxSteps $MAX_STEPS --grouped $GROUPED

cp ~/hooman/output/log.txt $OUTPUT_DIR
cp ~/hooman/rw/random_walk.sh $OUTPUT_DIR