import tensorflow as tf
import math
import numpy as np
import os
from pathlib import Path
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn import model_selection, linear_model, preprocessing, svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
# pylint: disable=g-import-not-at-top
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
# current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# Settings
from src.main.python.minibatch import MiniBatch

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate.')
flags.DEFINE_string('train_prefix', '',
                    'name of the object file that stores the training data. must be specified.')

flags.DEFINE_integer('dim', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('vocab_size', 10400, 'Size of vocabulary.')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 400, 'minibatch size.')
flags.DEFINE_integer('sim_comp_freq', 10000, 'Frequency of computing similarity.')
flags.DEFINE_integer('avg_loss_comp_freq', 1000, 'Frequency of average loss computation.')

flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('input_dir', '.', 'Input data directory.')
flags.DEFINE_string('train_file', '', 'Input train file name.')
flags.DEFINE_string('existing_vocabs_file', '', 'Input existing vocab file name.')
flags.DEFINE_string('delimiter', '\t', 'Delimiter.')
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('start_id', 1, "Start ID for the vertices.")
flags.DEFINE_integer('seed', 1234, "Seed for random generator.")


def read_data(fname):
    rws = np.concatenate([
        np.loadtxt(f.open(), delimiter=flags.delimiter, dtype=int)
        for f in Path(flags.input_dir).glob(fname)
        if f.stat().st_size > 0])
    return rws


def load_data(file_name):
    np.random.seed(seed=flags.seed)
    data = read_data(file_name) - flags.start_id
    print('Data size', len(data))
    return data


def read_vocabs(fname):
    rws = np.concatenate([
        np.loadtxt(f.open(), dtype=int)  # accepts only the first label
        for f in Path(flags.input_dir).glob(fname)
        if f.stat().st_size > 0
    ])

    print('Existing Vocab size: ' + str(len(rws)))
    return rws


def choose_valid_examples(existing_vocabs):
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    np.random.seed(flags.seed)
    valid_examples = np.random.choice(existing_vocabs, flags.valid_size, replace=False)
    return valid_examples


def train(minibatch, valid_examples):
    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[flags.batch_size])
            train_contexts = tf.placeholder(tf.int32, shape=[flags.batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([flags.vocab_size, flags.dim], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [flags.vocab_size, flags.dim],
                        stddev=1.0 / math.sqrt(flags.dim)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([flags.vocab_size]))
                # Compute the average NCE loss for the batch.
                # tf.nce_loss automatically draws a new sample of the negative labels each
                # time we evaluate the loss.
                # Explanation of the meaning of NCE loss:
                #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_contexts,
                        inputs=embed,
                        num_sampled=flags.neg_sample_size,
                        num_classes=flags.vocab_size))
            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)
            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(flags.learning_rate).minimize(loss)
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                      valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()

    minibatch.reset_batch_gen()

    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(flags.log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(minibatch.num_batches()):
            #         print(step)
            inputs, contexts = minibatch.next_batch()
            feed_dict = {train_inputs: inputs, train_contexts: contexts}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (minibatch.num_batches() - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % flags.avg_loss_comp_freq == 0:
                    if step > 0:
                        average_loss /= 50
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % flags.sim_comp_freq == 0:
                    sim = similarity.eval()
                    for i in range(len(valid_examples)):
                        valid_word = valid_examples[i]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = nearest[k]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)

        final_embeddings = normalized_embeddings.eval()
        # Write corresponding labels for the embeddings.

        with open(flags.log_dir + '/metadata.tsv', 'w') as f:
            for i in range(flags.vocab_size):
                f.write(str(i + flags.start_id) + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(flags.log_dir, 'model.ckpt'))

        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(flags.log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

        writer.close()
    return final_embeddings


def main(argv=None):
    print("Loading the input data..")
    train_data = load_data(flags.train_file)
    print("Finished loading.")
    minibatch = MiniBatch(train_data, flags.batch_size, flags.seed)
    existing_vocabs = read_vocabs(flags.existing_vocabs_file)
    valid_examples = choose_valid_examples(existing_vocabs)
    final_embeddings = train(minibatch, valid_examples)


if __name__ == '__main__':
    tf.app.run()
