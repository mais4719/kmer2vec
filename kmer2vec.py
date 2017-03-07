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
from tensorflow.contrib.tensorboard.plugins import projector
from six.moves import cPickle as pickle
from random import shuffle
from tempfile import mkdtemp
from utils import multisize_patten2number, number2multisize_patten
from utils import tsv_file2kmer_vector
from utils import read_faidx
from utils import reverse_complement, gc, sequence_entropy


# REF_GENOME_FA = '/sas/seq5/StaticFiles/genomes/hs38DH/hs38DH.fa'
# REF_GENOME_FAI = '/sas/seq5/StaticFiles/genomes/hs38DH/hs38DH.fa.fai'


# Creating command line options.
flags = tf.app.flags

flags.DEFINE_string('fa_file', default_value='',
                    docstring='Fasta file with accompanied samtool index.')
flags.DEFINE_string('chr_filter', default_value='decoy,chrEBV',
                    docstring=('Comma-separated list of strings not allowed '
                               'within the chromosome/parent name.'
                               '[decoy,chrEBV].'))

flags.DEFINE_string('validation_vocabulary', default_value='',
                    docstring='Path to validation vocabulary file.')

flags.DEFINE_string('save_path', default_value='.',
                    docstring=('Output path for logs and results '
                               '[current path].'))

flags.DEFINE_integer('min_kmer_size', default_value=3,
                     docstring=('Minimum kmer size in nucleotides used during '
                                'traning [3].'))
flags.DEFINE_integer('max_kmer_size', default_value=5,
                     docstring=('Maximum kmer size in nucleotides used during '
                                'traning [5].'))
flags.DEFINE_integer('padding', default_value=1,
                     docstring='Number of kmers used on each side as ' +
                               'the context for this skip grama model [1].')

flags.DEFINE_integer('batch_size', default_value=128,
                     docstring='Number of training examples processed per ' +
                               'step [128]')
flags.DEFINE_integer('epochs', default_value=10,
                     docstring='Number of traning epochs [10].')
flags.DEFINE_float('learning_rate', default_value=0.002,
                   docstring='Initial learning rate [0.2].')
flags.DEFINE_integer('embedding_size', default_value=100,
                     docstring='The embedding dimension size [100].')
flags.DEFINE_integer('num_neg_samples', default_value=100,
                     docstring='Negative samples per training example [100].')

flags.DEFINE_boolean('interactive', default_value=False,
                     docstring='Jumps into an iPython shell for debugging.')

flags.DEFINE_string('log_dir', default_value=None,
                    docstring='Path to validation vocabulary file.')

flags.DEFINE_string('embeddings_file', default_value=None,
                    docstring='Path to previously pickled embeddings.')

FLAGS = flags.FLAGS


class Kmer2Vec(object):
    """ Kmer2Vec Skipgram Model """

    def __init__(self, flags, session):
        self._flags = flags
        self._session = session

        assert flags.min_kmer_size <= flags.max_kmer_size
        self.KMER_SIZES = np.arange(flags.min_kmer_size,
                                    flags.max_kmer_size + 1)

        self.VOCABULARY_SIZE = np.sum(4**self.KMER_SIZES)

        self.VALID_WINDOW = 200
        self.VALID_SET_SIZE = 10
        self.valid_kmers = self._get_valid_examples()

        self.ALLOWED_CHRS = [chr_name for chr_name in
                             read_faidx(flags.fa_file,
                                        flags.chr_filter.split(','))]

        self.NUMBER_OF_CHRS = len(self.ALLOWED_CHRS)

        self.build_graph()

        self.metadata_file = create_metadata(os.path.join(flags.log_dir,
                                                          'metadata.tsv'),
                                             flags.min_kmer_size,
                                             flags.max_kmer_size)
        self.setup_summaries()

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

    def _get_valid_examples(self):
        """ Picking a random validation set out of top frequencly
        occuring kmers within the reference sequence.
        """
        valid_file = self._flags.validation_vocabulary

        try:
            # Make sure to convert kmer seqs into same max min space
            # used in the run.
            kmer_vector = tsv_file2kmer_vector(valid_file,
                                               self._flags.min_kmer_size,
                                               self._flags.max_kmer_size)

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

        return valid_kmers

    def build_graph(self):
        """ Build Model Graph """
        f = self._flags

        # Input data
        self.train_inputs = tf.placeholder(tf.int32, shape=[f.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[f.batch_size, 1])
        valid_dataset = tf.constant(self.valid_kmers, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # Ceating, or load, kmer embeddings.
            if f.embeddings_file:
                print('Loading embeddings from [%s]' % f.embeddings_file)
                self.embeddings = tf.Variable(pickle.load(open(f.embeddings_file, 'rb')),
                                              name='embeddings')
            else:
                self.embeddings = tf.Variable(tf.random_uniform([self.VOCABULARY_SIZE,
                                                                 f.embedding_size],
                                              -1.0, 1.0),
                                              name='embeddings')

            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        # Construct the variables for the NCE loss
        self.nce_weights = tf.Variable(tf.truncated_normal([self.VOCABULARY_SIZE,
                                                            f.embedding_size],
                                       stddev=1.0 / math.sqrt(f.embedding_size)),
                                       name='nce_weights')
        self.nce_biases = tf.Variable(tf.zeros([self.VOCABULARY_SIZE]),
                                      name='nce_biases')

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of
        # the negative labels each time we evaluate the loss.
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                                                  biases=self.nce_biases,
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
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = tf.divide(self.embeddings, norm,
                                               name='norm_embeddings')
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                  valid_dataset,
                                                  name='valid_embeddings')
        self.similarity = tf.matmul(valid_embeddings,
                                    self.normalized_embeddings,
                                    transpose_b=True,
                                    name='similarity')

        print('Build Model...')

    def setup_summaries(self):
        """ Setting up logging and summary output. """
        f = self._flags

        # Summary variables
        loss_op = tf.summary.scalar('Batch average NCE loss',
                                    self.loss)
        nce_weights_op = tf.summary.histogram('NCE weights',
                                              self.nce_weights)
        nce_biases_op = tf.summary.histogram('NCE biases',
                                             self.nce_biases)

        embeddings_op = tf.summary.histogram('embeddings',
                                             self.embeddings)
        norm_embeddings_op = tf.summary.histogram('norm embeddings',
                                                  self.normalized_embeddings)

        self.merged_op = tf.summary.merge([loss_op,
                                           nce_weights_op,
                                           nce_biases_op,
                                           embeddings_op,
                                           norm_embeddings_op])
        train_summary_dir = os.path.join(f.log_dir, 'train')
        self.summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                    self._session.graph)

        # Embedding visualization
        # Format:
        #     tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        embedding.metadata_path = self.metadata_file

        # Saves a configuration file that TensorBoard will read.
        projector.visualize_embeddings(self.summary_writer, config)

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
        # embedding_saver = tf.train.Saver([self.embeddings])
        embedding_saver = tf.train.Saver()

        output_file_template = ('max{max_size}_min{min_size}_mers'
                                '_{padding}padding_{emb_size}embedding_'
                                'epoch{epoch}_batch{index}.pickle')
        output_file_template = os.path.join(f.save_path, output_file_template)

        for epoch in range(1, f.epochs + 1):

            old_chrom = None
            chroms_done = 0
            shuffle(self.ALLOWED_CHRS)

            for index, (chrom, batch, labels) in enumerate(batch_generator(f.batch_size,
                                                                           f.fa_file,
                                                                           self.ALLOWED_CHRS,
                                                                           f.min_kmer_size,
                                                                           f.max_kmer_size,
                                                                           f.padding)):

                feed_dict = {self.train_inputs: batch,
                             self.train_labels: labels}

                if old_chrom != chrom:
                    print('Starting traning on {}...'.format(chrom))
                    chroms_done += 1
                    old_chrom = chrom

                if index % 20000 == 0:
                    _, summary, step, loss_val = self._session.run([self.train_opt,
                                                                    self.merged_op,
                                                                    self.global_step,
                                                                    self.loss],
                                                                   feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary, step)

                    info_str = '{} ({}/{} done) Avg. loss epoch: {} batch: {}'
                    print(info_str.format(chrom,
                                          chroms_done,
                                          self.NUMBER_OF_CHRS,
                                          epoch,
                                          index))
                else:
                    _, step, loss_val = self._session.run([self.train_opt,
                                                           self.global_step,
                                                           self.loss],
                                                          feed_dict=feed_dict)

                # Show nearest kmers to each kmers in the validation set (cosine distance).
                # Save model to ckpt-file for TensorBoard embedding visualization.
                if index % 100000 == 0:
                    sim = self.similarity.eval()
                    for i in range(self.VALID_SET_SIZE):
                        valid_kmer = number2multisize_patten(self.valid_kmers[i],
                                                             f.min_kmer_size,
                                                             f.max_kmer_size)
                        top_k = 4  # Number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        close_kmers = [number2multisize_patten(nearest[k],
                                                               f.min_kmer_size,
                                                               f.max_kmer_size)
                                       for k in range(top_k)]
                        print('Nearest to {} -> ({})'.format(valid_kmer, ', '.join(close_kmers)))

                    # Save embeddings
                    embedding_saver.save(self._session,
                                         os.path.join(f.log_dir, 'model.ckpt'),
                                         step)

                # Save embeddings to pickle file.
                if index % 1000000 == 0:
                    self.save_vocab(output_file_template.format(min_size=f.min_kmer_size,
                                                                max_size=f.max_kmer_size,
                                                                padding=f.padding,
                                                                emb_size=f.embedding_size,
                                                                epoch=epoch,
                                                                index=index))

        self.save_vocab(output_file_template.format(min_size=f.min_kmer_size,
                                                    max_size=f.max_kmer_size,
                                                    padding=f.padding,
                                                    emb_size=f.embedding_size,
                                                    epoch=epoch,
                                                    index=index))


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


def create_metadata(filename, min_length=3, max_length=5):
    """ Target and context k-mer batch generator for a reference fasta file.

    Args:
          filename (str): Path to output tsv-file.
        min_length (int): Minimal allowed kmer size (nt).
        max_length (int): Maximum allowed kmer size (nt).

    Return:
        filename (str): If successful, path to created metadata tsv file.
    """
    kmer_sizes = np.arange(min_length, max_length + 1)
    vocabulary_size = np.sum(4**kmer_sizes)
    tmpl_str = '{name}\t{length}\t{gc}\t{ent}\n'

    with open(filename, 'w') as f:
        f.write(tmpl_str.format(name='name', length='length',
                                gc='gc', ent='entropy'))
        for seq_numb in range(0, vocabulary_size):
            seq = number2multisize_patten(seq_numb, min_length, max_length)
            seq_length = len(seq)
            seq_gc = str(round(gc(seq), 2))
            seq_ent = str(round(sequence_entropy(seq), 2))
            f.write(tmpl_str.format(name=seq, length=seq_length,
                                    gc=seq_gc, ent=seq_ent))

    return filename


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

        # Log file directory
        if not FLAGS.log_dir:
            FLAGS.log_dir = mkdtemp(prefix='kmer2vec_')
            print('Created new log directory [%s]' % FLAGS.log_dir)
        else:
            if tf.gfile.Exists(FLAGS.log_dir):
                print('Log directory already exist [%s]' % FLAGS.log_dir)
                return
            tf.gfile.MakeDirs(FLAGS.log_dir)

        model = Kmer2Vec(FLAGS, session)

        if FLAGS.interactive:
            _start_shell(local_ns=globals().update({'model': model}))
        else:
            model.train()


if __name__ == '__main__':
    tf.app.run()
