#!/bin/bash

RW_JAR_FILE=your-dir/randomwalk-0.0.1-SNAPSHOT.jar
INPUT_EDGE_LIST=your-dir/coauthors-edge-list.txt
OUTPUT_DIR=your-dir
WALK_LENGTH=10
NUM_WALKS=80
NUM_RUNS=1
DIRECTED=false    # tested on undirected graphs only.
P=0.25
Q=0.25
SEED=1234
METHOD_TYPE=m1
WALK_TYPE=secondorder
DELIMITER="\ "    # e.g., tab-separated ("\\t"), or comma-separated (",").
LOG_PERIOD=1      # after what number of steps log the output
LOG_ERRORS=false  # Should it compute and log transition probability errors (computation intensive)
INIT_EDGE_SIZE=0.5    # portion of edges to be used to construct the initial graph
STREAM_SIZE=0.0001    # portion of edges to be used for streaming at each step
MAX_STEPS=20          # max number of steps to run the experiment
GROUPED=false         # whether the edge list is already tagged with group number (e.g., year)

# You can customize the JVM memory size by modifying -Xms.
# To run the script on the background: nohup sh random_walk.sh > log.txt &

java -Xms5g -jar $RW_JAR_FILE  --cmd sca --walkLength $WALK_LENGTH --numWalks $NUM_WALKS \
    --input $INPUT_EDGE_LIST --output $OUTPUT_DIR --nRuns $NUM_RUNS --directed $DIRECTED --p $P \
    --q $Q --seed $SEED --d "$DELIMITER" --rrType $METHOD_TYPE --wType $WALK_TYPE --save $LOG_PERIOD \
    --logErrors $LOG_ERRORS --initEdgeSize $INIT_EDGE_SIZE --edgeStreamSize $STREAM_SIZE \
    --maxSteps $MAX_STEPS --grouped $GROUPED