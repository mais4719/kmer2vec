#!/usr/bin/env python3
"""
Vector Representations of genomic k-mers
(Skip-gram Model)
"""
import os
import pysam
import gzip
import math
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from random import shuffle

REF_GENOME_FA = '/sas/seq5/StaticFiles/genomes/hs38DH/hs38DH.fa'
REF_GENOME_FAI = '/sas/seq5/StaticFiles/genomes/hs38DH/hs38DH.fa.fai'

BASE_TO_NUMBER = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
NUMBER_TO_BASE = ('A', 'C', 'G', 'T')

VALID_PATH = '/sas/seq5/Isaksson/SSEL_ML_investigation/hg38_Vocabulary'


# Creating command line options.
flags = tf.app.flags

flags.DEFINE_string('fa_file', REF_GENOME_FA,
                    'Fasta file with accompanied samtool index [{}].'.format(REF_GENOME_FA))
flags.DEFINE_string('fai_file', REF_GENOME_FAI,
                    'Fasta index file with accompanied fasta [{}].'.format(REF_GENOME_FAI))

flags.DEFINE_string('valid_path', VALID_PATH,
                    'Path to validation files, sorted on ' +
                    'frequency within fasta file ({}).'.format(VALID_PATH))
flags.DEFINE_string('save_path', '.', 'Output path for logs and results [current path].')

flags.DEFINE_integer('kmer_size', 6, 'Kmer size in nucleotides [6].')
flags.DEFINE_integer('padding', 1,
                     'Number of kmers used on each side as ' +
                     'the context for this skip grama model [1].')

flags.DEFINE_integer('batch_size', 128,
                     'Number of training examples processed per step [128]')
flags.DEFINE_integer('epochs', 10,
                     'Number of traning epochs [10].')
flags.DEFINE_float('learning_rate', 0.2,
                   'Initial learning rate [0.2].')
flags.DEFINE_integer('embedding_size', 200,
                     'The embedding dimension size [200].')
flags.DEFINE_integer('num_neg_samples', 100,
                     'Negative samples per training example [100].')

FLAGS = flags.FLAGS


class Kmer2Vec(object):
    """ Word2Vec model with skipgram. """

    def __init__(self, flags, session):
        self._flags = flags
        self._session = session

        self.VOCABULARY_SIZE = 4**flags.kmer_size

        self.VALID_WINDOW = 200
        self.VALID_SET_SIZE = 10
        self.valid_examples = self._get_valid_examples()

        self.allowed_chrs, self.kmers_to_train = self._get_chromosomes()
        self.number_of_chrom = len(self.allowed_chrs)

        self.build_graph()

    def _get_valid_examples(self):
        """ Picking a random validation set out of top kmers. """

        valid_random = np.random.choice(self.VALID_WINDOW,
                                        self.VALID_SET_SIZE,
                                        replace=False)
        valid_examples = []
        valid_file = os.path.join(self._flags.valid_path,
                                  'vocabulary_{}mers.sorted.tsv.gz'.format(self._flags.kmer_size))
        if os.path.isfile(valid_file):
            for i, kmer in enumerate(gzip.open(valid_file)):
                if i in valid_random:
                    valid_examples.append(patten2number(kmer.split()[0]))
                if i == self.VALID_WINDOW - 1:
                    break

        return valid_examples

    def _get_chromosomes(self):
        """ Collect a list of chromosome names and the number of possible
            kmers from the fasta index file '*.fai'
        """
        allowed_chrs = []
        numbers_of_kmers_to_train = 0
        filter_chrom_strs = ('decoy', 'chrEBV')

        for row in open(REF_GENOME_FAI, 'r'):
            chrom_str, chom_len = row.split()[:2]
            if not any([s in chrom_str for s in filter_chrom_strs]):
                allowed_chrs.append(chrom_str)
                numbers_of_kmers_to_train += int(chom_len) - self._flags.kmer_size + 1

        numbers_of_kmers_to_train *= self._flags.epochs

        return allowed_chrs, numbers_of_kmers_to_train

    def build_graph(self):
        """ Build the graph for the model. """
        f = self._flags

        # Input data
        self.train_inputs = tf.placeholder(tf.int32, shape=[f.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[f.batch_size, 1])
        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # Look up embeddings for inputs
            embeddings = tf.Variable(tf.random_uniform([self.VOCABULARY_SIZE, f.embedding_size],
                                                       -1.0, 1.0))
            #embeddings = tf.Variable(pickle.load(open('10-mers/10mers_1padding_200embedding_start.pickle', 'rb')))
            embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([self.VOCABULARY_SIZE, f.embedding_size],
                                                      stddev=1.0 / math.sqrt(f.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.VOCABULARY_SIZE]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        self.loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights,
                                                  nce_biases,
                                                  embed,
                                                  self.train_labels,
                                                  f.num_neg_samples,
                                                  self.VOCABULARY_SIZE))

        # Construct the SGD optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = f.learning_rate * tf.maximum(
                0.0001, 1.0 - tf.cast(self.VOCABULARY_SIZE, tf.float32) / self.kmers_to_train)
        # self._learning_rate = learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescent')
        self.train_opt = optimizer.minimize(self.loss,
                                            global_step=self.global_step,
                                            gate_gradients=optimizer.GATE_NONE)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

        # Properly initialize all variables.
        tf.initialize_all_variables().run()
        print('Build and Initialized Model...')

    def save_vocab(self, filename):
        """ Evaluate and saves current embeddings to a pickle file. """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.normalized_embeddings.eval(),
                            f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', filename, ':', e)

    def train(self):
        """ Train the model. """
        f = self._flags

        average_loss = 0
        output_file_template = os.path.join(f.save_path,
                                            '{}mers_{}padding_{}embedding_epoch{}_batch{}.pickle')

        for epoch in range(1, f.epochs + 1):

            old_chrom = None
            chroms_done = 0
            shuffle(self.allowed_chrs)

            for index, (chrom, batch, labels) in enumerate(generate_batch(f.batch_size,
                                                                          self.allowed_chrs,
                                                                          f.kmer_size,
                                                                          f.padding)):

                feed_dict = {self.train_inputs: batch,
                             self.train_labels: labels}

                _, step, loss_val = self._session.run([self.train_opt,
                                                       self.global_step,
                                                       self.loss],
                                                      feed_dict=feed_dict)
                average_loss += loss_val

                if old_chrom != chrom:
                    print('Starting traning on {}...'.format(chrom))
                    chroms_done += 1
                    old_chrom = chrom

                if index % 20000 == 0:
                    if index > 0:
                        average_loss /= 20000
                    info_str = '{} ({}/{} done) Avg. loss epoch: {} batch: {} -> {}'
                    print(info_str.format(chrom, chroms_done,
                                          self.number_of_chrom,
                                          epoch, index,
                                          average_loss))
                    average_loss = 0

                if index % 100000 == 0:
                    sim = self.similarity.eval()
                    for i in xrange(self.VALID_SET_SIZE):
                        valid_kmer = number2patten(self.valid_examples[i], f.kmer_size)
                        top_k = 4  # Number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        close_kmers = [number2patten(nearest[k], f.kmer_size,) for k in xrange(top_k)]
                        print('Nearest to {} -> ({})'.format(valid_kmer, ', '.join(close_kmers)))

                if index % 1000000 == 0:
                    self.save_vocab(output_file_template.format(f.kmer_size,
                                                                f.padding,
                                                                f.embedding_size,
                                                                epoch,
                                                                index))

        self.save_vocab(output_file_template.format(f.kmer_size,
                                                    f.padding,
                                                    f.embedding_size,
                                                    epoch,
                                                    index))


def patten2number(sequence):
    """ Converts DNA sequence into an int. """
    if len(sequence) == 0:
        return 0
    last_base = sequence[-1]
    prefix = sequence[:-1]
    return 4 * patten2number(prefix) + BASE_TO_NUMBER[last_base]


def number2patten(number, kmer_size):
    """ Converts an int into a string with k nucleotides. """
    if kmer_size == 1:
        return NUMBER_TO_BASE[number]

    prefix_index = number // 4
    base = NUMBER_TO_BASE[number % 4]
    return number2patten(prefix_index, kmer_size - 1) + base


def create_int_kmers(sequence, kmer_size, jump_size=1):
    """ Creates kmers from a sequence """
    assert(jump_size < len(sequence) - kmer_size + 1)
    for i in xrange(0, len(sequence) - kmer_size + 1, jump_size):
        yield sequence[i:i + kmer_size]


def nearby(chroms, kmer_size=10, padding=1):

    with pysam.FastaFile(REF_GENOME_FA) as ref:
        for chrom in chroms:
            chr_seq = ref.fetch(chrom)
            for subseq_pos in xrange(0, len(chr_seq)):
                subseq = chr_seq[subseq_pos:subseq_pos + kmer_size + (kmer_size * padding * 2)]
                try:
                    num_kmers = []
                    for i in range(0, len(subseq), kmer_size):
                        num_kmers.append(patten2number(subseq[i:i + kmer_size]))

                    neighbours = tuple(num_kmers[:padding] + num_kmers[-padding:])
                    center = num_kmers[padding]

                    yield (chrom, center, neighbours)

                except (KeyError, IndexError):  # as e:
                    pass  # Was not able to convert patten to number or
                          # we are at the end of the chromosome.


def generate_batch(batch_size, chroms, kmer_size=10, padding=1):
    """ Batching yields from the nearby generator """

    batch = np.ndarray(shape=(batch_size), dtype=np.uint32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.uint32)
    i = 0

    for chrom, center, neighbours in nearby(chroms, kmer_size, padding):
        for neighbour in neighbours:
            batch[i] = center
            labels[i] = neighbour
            if i % (batch_size - 1) == 0 and i > 0:
                np.random.shuffle(labels)  # Avoid gram bias
                yield chrom, batch, labels
                batch = np.ndarray(shape=(batch_size), dtype=np.uint32)
                labels = np.ndarray(shape=(batch_size, 1), dtype=np.uint32)
                i = 0
            else:
                i += 1


def _start_shell(local_ns=None):
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
    with tf.Graph().as_default(), tf.Session() as session:
        model = Kmer2Vec(FLAGS, session)
        model.train()


if __name__ == '__main__':
    tf.app.run()
