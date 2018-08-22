# Dynamic node2vec #


## Features ##
* dynamic algorithms for node2vec's second-order random walk
* Corpus generator, i.e., context-target pairs generator from given random walks

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
   --weighted <value>       weighted: true
   --directed <value>       directed: false
   --input <value>          Input edge-file/paths-file: empty
   --output <value>         Output path: empty
   --cmd <value>            command: randomwalk/embedding/node2vec (to run randomwalk + embedding)
```
     
## Graph Input File Format ##

## Output Format ##

```

## References ##
1. (Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016.).
2. Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).









