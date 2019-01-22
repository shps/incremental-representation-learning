# incremental-representation-learning
Incremental vertex representation learning using random walks and skip-gram model. For the detail of the algorithms please refer to:

```
Sajjad, Hooman Peiro, Andrew Docherty, and Yuriy Tyshetskiy. 
"Efficient Representation Learning Using Random Walks for Dynamic graphs." 
arXiv preprint arXiv:1901.01346 (2019).
```

# Dynamic DeepWalk/node2vec #
This is the implementation of our algorithms for unsupervised representation learning using random walks for dynamic networks. It includes four different algorithms based on DeepWalk's first order and node2vec's second order random walk. Algorithm M1 is our implementation of the vanilla DeepWalk/node2vec.

## Features ##
* dynamic algorithms for the DeepWalk/node2vec network representation learning
* Corpus generator, i.e., context-target pairs generator.

## Requirements ##
* Scala 2.12 or later.
* Maven 3+
* Java 8+

## Application Options ##
You can run the application using the run_all.sh script file. You need to configure the script file before running your experiments.

## References ##
1. Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014.
2. (Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016.).
3. Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).
