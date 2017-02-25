#!/usr/bin/env python3
"""
Vector Representations of genomic k-mers
(Skip-gram Model)
"""
import os
import pysam
import math
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from random import shuffle
from utils import multisize_patten2number, number2multisize_patten
from utils import number2patten
from utils import tsv_file2kmer_vector
from utils import read_faidx
from utils import reverse_complement

# REF_GENOME_FA = '/sas/seq5/StaticFiles/genomes/hs38DH/hs38DH.fa'
# REF_GENOME_FAI = '/sas/seq5/StaticFiles/genomes/hs38DH/hs38DH.fa.fai'


# Creating command line options.
flags = tf.app.flags

flags.DEFINE_string('fa_file', default_value='',
                    docstring='Fasta file with accompanied samtool index.')
flags.DEFINE_string('chr_filter', default_value='decoy,chrEBV',
                    docstring='Comma-separated list of strings not allowed ' +
                              'within the chromosome/parent name. [decoy,chrEBV].')

flags.DEFINE_string('validation_vocabulary', default_value='',
                    docstring='Path to validation vocabulary file.')

flags.DEFINE_string('save_path', default_value='.',
                    docstring='Output path for logs and results [current path].')

flags.DEFINE_string('kmer_sizes', default_value='3,4,5',
                    docstring='Comma-separated list of kmer sizes in ' +
                              'nucleotides [3,4,5].')
flags.DEFINE_integer('padding', default_value=1,
                     docstring='Number of kmers used on each side as ' +
                               'the context for this skip grama model [1].')

flags.DEFINE_integer('batch_size', default_value=128,
                     docstring='Number of training examples processed per ' +
                               'step [128]')
flags.DEFINE_integer('epochs', default_value=10,
                     docstring='Number of traning epochs [10].')
flags.DEFINE_float('learning_rate', default_value=0.2,
                   docstring='Initial learning rate [0.2].')
flags.DEFINE_integer('embedding_size', default_value=300,
                     docstring='The embedding dimension size [300].')
flags.DEFINE_integer('num_neg_samples', default_value=100,
                     docstring='Negative samples per training example [100].')

flags.DEFINE_boolean('interactive', default_value=False,
                     docstring='Jumps into an iPython shell for debugging.')

FLAGS = flags.FLAGS


class Kmer2Vec(object):
    """ Kmer2Vec Skipgram Model """

    def __init__(self, flags, session):
        self._flags = flags
        self._session = session

        self.KMER_SIZES = set([int(s) for s in flags.kmer_sizes.split(',')])
        self.VOCABULARY_SIZE = sum([4**s for s in self.KMER_SIZES])

        self.VALID_WINDOW = 200
        self.VALID_SET_SIZE = 10
        self.valid_kmers, self.valid_kmer_size = self._get_valid_examples()

        assert self.valid_kmer_size in self.KMER_SIZES

        self.ALLOWED_CHRS = set([chr_name for chr_name in
                                 read_faidx(flags.fa_file, flags.chr_filter)])

        self.NUMBER_OF_CHRS = len(self.ALLOWED_CHRS)

        self.build_graph()

    def _get_valid_examples(self):
        """ Picking a random validation set out of top frequencly
        occuring kmers within the reference sequence.
        """
        valid_file = self._flags.validation_vocabulary

        try:
            kmer_vector, valid_kmer_size = tsv_file2kmer_vector(valid_file)

            valid_kmers = np.argpartition(kmer_vector,
                                          -self.VALID_WINDOW)[-self.VALID_WINDOW:]
            valid_kmers = np.random.choice(valid_kmers,
                                           self.VALID_SET_SIZE,
                                           replace=False)
        except Exception:
            print('Not able to fetch validation examples ' +
                  'from validation vocabulary file ' +
                  '[{}].'.format(valid_file))
            raise

        return valid_kmers, valid_kmer_size

    def build_graph(self):
        """ Build Model Graph """
        f = self._flags

        # Input data
        self.train_inputs = tf.placeholder(tf.int32, shape=[f.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[f.batch_size, 1])
        valid_dataset = tf.constant(self.valid_kmers, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # Ceating word embeddings for kmers
            embeddings = tf.Variable(tf.random_uniform([self.VOCABULARY_SIZE, f.embedding_size],
                                                       -1.0, 1.0))
            #embeddings = tf.Variable(pickle.load(open('10-mers/10mers_1padding_200embedding_start.pickle', 'rb')))
            embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([self.VOCABULARY_SIZE, f.embedding_size],
                                                      stddev=1.0 / math.sqrt(f.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.VOCABULARY_SIZE]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of
        # the negative labels each time we evaluate the loss.
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                  biases=nce_biases,
                                                  inputs=embed,
                                                  labels=self.train_labels,
                                                  num_sampled=f.num_neg_samples,
                                                  num_classes=self.VOCABULARY_SIZE))

        # Construct the SGD optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # TODO estimate self.kmers_to_train
        #learning_rate = f.learning_rate * tf.maximum(
        #        0.0001, 1.0 - tf.cast(self.VOCABULARY_SIZE, tf.float32) / self.kmers_to_train)
        learning_rate = f.learning_rate

        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescent')
        self.train_opt = optimizer.minimize(self.loss,
                                            global_step=self.global_step,
                                            gate_gradients=optimizer.GATE_NONE)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                  valid_dataset)
        self.similarity = tf.matmul(valid_embeddings,
                                    self.normalized_embeddings,
                                    transpose_b=True)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
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

        output_file_template = '{}mers_{}padding_{}embedding_epoch{}_batch{}.pickle'
        output_file_template = os.path.join(f.save_path, output_file_template)

        for epoch in range(1, f.epochs + 1):

            old_chrom = None
            chroms_done = 0
            shuffle(self.allowed_chrs)

            for index, (chrom, batch, labels) in enumerate(batch_generator(f.batch_size,
                                                                           f.fa_file,
                                                                           self.ALLOWED_CHRS,
                                                                           self.KMER_SIZES,
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
                    print(info_str.format(chrom,
                                          chroms_done,
                                          self.NUMBER_OF_CHRS,
                                          epoch,
                                          index,
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

        self.save_vocab(output_file_template.format('-'.join(self.KMER_SIZES),
                                                    f.padding,
                                                    f.embedding_size,
                                                    epoch,
                                                    index))


def context_generator(fa_file, chroms, min_length=3, max_length=5, padding=1):
    """ Creates context and target k-mers using provided fasta and
    fasta index file. Using a 1 base sliding window approach with
    random k-mer sizes between min and max length. Both polarities
    are sampled randomly.

    E.g. min_length=3, max_length=5, padding=1

        rnd_kmer_sizes = [4, 3, 5]
        CATATCA -> ['CATA', 'ATA', 'TATCA']

        -> ('chr?', 'ATA', ['CATA', 'TATCA'])

        DNA sequences will be converted into ints for the final result

        -> ('chr?', 12, [140, 1140])

    Args:
          fa_file (str): Path to fasta file with with accompanying
                         Samtools index file (*.fai).
          chroms (list): Orded list of chromosome/parent ids which will
                         be included when iterating over the fasta file.
       min_length (int): Minimal allowed kmer size (nt).
       max_length (int): Maximum allowed kmer size (nt).
          padding (int): Number of kmers, on each side, added to the context.

    Yields:
        chromosom_id (str), target_seq (int), list(context_seqs (ints))
    """
    kmer_sizes = np.arange(min_length, max_length + 1)

    with pysam.FastaFile(fa_file) as ref:
        for chrom in chroms:
            chr_seq = ref.fetch(chrom)
            for subseq_pos in range(0, len(chr_seq)):

                # Create random kmer sizes.
                rnd_kmer_sizes = np.random.choice(kmer_sizes, padding * 2 + 1)

                # Extract sub-sequence from provided fasta file.
                subseq = chr_seq[subseq_pos:subseq_pos +
                                 rnd_kmer_sizes.size + rnd_kmer_sizes.max()]

                if len(subseq) < rnd_kmer_sizes.size + rnd_kmer_sizes.max():
                    continue

                # Randomly use both strand for learning.
                if np.random.randint(2):
                    subseq = reverse_complement(subseq)

                try:
                    num_kmers = []

                    for i, pos in enumerate(rnd_kmer_sizes):
                        kmer_seq = subseq[i:i + rnd_kmer_sizes[i]]
                        number_seq = multisize_patten2number(kmer_seq,
                                                             min_length,
                                                             max_length)
                        num_kmers.append(number_seq)

                    context = np.array(num_kmers[:padding] +
                                       num_kmers[-padding:])
                    # np.random.shuffle(context)

                    target = num_kmers[padding]

                    yield chrom, target, context

                except (KeyError, IndexError, ValueError):  # as e:
                    pass  # Was not able to convert patten to number or
                          # we are at the end of the chromosome.


def batch_generator(batch_size, fa_file, chroms,
                    min_length=3, max_length=5, padding=1):
    """ Target and context k-mer batch generator for a reference fasta file.

    Args:
        batch_size (int): Size of each yield batch.
           fa_file (str): Path to fasta file with with accompanying
                          Samtools index file (*.fai).
           chroms (list): Orded list of chromosome/parent ids which will
                          be included when iterating over the fasta file.
        min_length (int): Minimal allowed kmer size (nt).
        max_length (int): Maximum allowed kmer size (nt).
           padding (int): Number of kmers, on each side, added to the context.

    Yields:
        chromosom_id (str),
        y_batch (numpy uint32 array shape=(batch_size)),
        label_batch (numpy uint32 array shape=(batch_size, 1))

    """

    y_batch = np.ndarray(shape=(batch_size), dtype=np.uint32)
    label_batch = np.ndarray(shape=(batch_size, 1), dtype=np.uint32)
    i = 0

    for chrom, target, context in context_generator(fa_file, chroms,
                                                    min_length, max_length,
                                                    padding):
        for neighbour in context:
            y_batch[i] = target
            label_batch[i] = neighbour
            if i % (batch_size - 1) == 0 and i > 0:
                yield chrom, y_batch, label_batch
                y_batch = np.ndarray(shape=(batch_size), dtype=np.uint32)
                label_batch = np.ndarray(shape=(batch_size, 1),
                                         dtype=np.uint32)
                i = 0
            else:
                i += 1


def _start_shell(local_ns=None):
    """ Enable interactive mode via iPython for development and debugging. """
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
    with tf.Graph().as_default(), tf.Session() as session:
        model = Kmer2Vec(FLAGS, session)
        if FLAGS.interactive:
            _start_shell(local_ns=globals().update({'model': model}))
        else:
            model.train()


if __name__ == '__main__':
    tf.app.run()
