#!/bin/bash

RW_JAR_FILE=your-dir/randomwalk-0.0.1-SNAPSHOT.jar
INPUT_EDGE_LIST=your-dir/a-random-walk-result.txt
OUTPUT_DIR=your-dir
DELIMITER="\\t"    # e.g., space-separated ("\ "), or comma-separated (",").
WINDOW_SIZE=3
SKIP_SIZE=6
SELF_CONTEXT=false  # whether allows target == context pairs.

# You can customize the JVM memory size by modifying -Xms.
# To run the script on the background: nohup sh generate_target_context_pairs.sh > log.txt &

java -Xms5g -jar $RW_JAR_FILE  --cmd gPairs --input $INPUT_EDGE_LIST --output $OUTPUT_DIR \
    --d "$DELIMITER"  --w2vWindow $WINDOW_SIZE --w2vSkip $SKIP_SIZE \
    --selfContext $SELF_CONTEXT