# Dynamic Random Walk (node2vec/DeepWalk) #


## Features ##
* node2vec's second-order random walk

## Requirements ##
* Scala 2.11 or later.
* Maven 3+
* Java 8+

## Quick Setup ##

## Application Options ##
The following options are available:

```
   --walkLength <value>     walkLength: 80
   --numWalks <value>       numWalks: 10
   --p <value>              return parameter p: 1.0
   --q <value>              in-out parameter q: 1.0
   --rddPartitions <value>  Number of RDD partitions in running Random Walk and Word2vec: 200
   --weighted <value>       weighted: true
   --directed <value>       directed: false
   --w2vPartitions <value>  Number of partitions in word2vec: 10
   --input <value>          Input edge-file/paths-file: empty
   --output <value>         Output path: empty
   --cmd <value>            command: randomwalk/embedding/node2vec (to run randomwalk + embedding)
   --partitioned <value>    Whether the graph is partitioned: false
   --lr <value>             Learning rate in word2vec: 0.025
   --iter <value>           Number of iterations in word2vec: 10
   --dim <value>            Number of dimensions in word2vec: 128
   --window <value>         Window size in word2vec: 10
```
     
## Graph Input File Format ##

## Output Format ##

```

## References ##
1. (Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016.).
2. Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).
3. [Random Walks on Large Scale Graphs with Apache Spark](https://spark-summit.org/2017/events/random-walks-on-large-scale-graphs-with-apache-spark/)









