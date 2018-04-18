# node2vec_experiments

Computes and evaluates node2vec embeddings from context pairs file.

## How to run the code:

The code takes a text file containing target-context pairs of node
indices:

```
1 3
40 2
4 5
...
```

Currently, the filenames are hard coded in the file. It expects two files:
 * `gPairs-w3-s6.txt`
 * `karate-labels.txt`

To change this, edit the following lines:

```
 ds = PregeneratedDataset(<target context pairs filename>,
                          n_nodes=<number of nodes in graph>,
                          delimiter="\t",
                          force_offset=-1,
                          splits=[0.8,0.2])
```

Additionally changing the delimiter and offset to match the file.
The `force_offset` flag is added to the node indices in the target-context
file to map the file node indices to the range 0 to (n_nodes-1).

## Training using previous node embeddings:

Node embeddings are automatically stored in tensorflow checkpoint files,
they can be reloaded in the code by specifying the `checkpoint_file`.
The checkpoints files are named `checkpoint...` for incremental checkpoints and
`model_epoch_<n>` saved after epoch n.

e.g.
```
checkpoint_file = "n2v_2018-04-18/checkpoint-170"
```

## Freezing node embeddings

Training can be performed with some embeddings frozen.

To freeze a set of node embeddings use:
```
freeze_indices = [199, 200, 399, 400]
```

To freeze a set of context embeddings (but allow embeddings to change) use:
```
freeze_context_indices = [199, 200, 399, 400]
```
