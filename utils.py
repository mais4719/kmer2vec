"""
Utilities Module for DNA Sequence.
"""
import os
import gzip
import numpy as np
from collections import Counter

__author__ = 'Magnus Isaksson'
__credits__ = ['Magnus Isaksson']
__version__ = '0.1.0'

BASE_TO_NUMBER = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
NUMBER_TO_BASE = ('A', 'C', 'G', 'T')
COMPLEMENTARY_BASE = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
                      'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W',
                      'K': 'M', 'M': 'K', 'B': 'V', 'D': 'H',
                      'H': 'D', 'V': 'B', 'N': 'N'}


def reverse_complement(sequence):
    """ Creates reverse complement of provided DNA sequence.

    Args:
        sequence (str): DNA sequence (allowing A, C, G, T uppercase only)

    Returns:
        str: DNA sequence string.
    """
    try:
        return ''.join([COMPLEMENTARY_BASE[base]
                        for base in reversed(sequence)])
    except KeyError:
        raise ValueError('Not able to reverse complement: %s' % sequence)


def patten2number(sequence):
    """ Recurrent function for converting DNA sequence to an interger.

    Args:
        sequence (str): DNA sequence (allowing A, C, G, T uppercase only)

    Returns:
        int: Interger reprencitation for a four bases sequence.
    """
    try:
        if len(sequence) == 0:
            return 0
        last_base = sequence[-1]
        prefix = sequence[:-1]
        return 4 * patten2number(prefix) + BASE_TO_NUMBER[last_base]
    except KeyError:
        raise ValueError('Not able to convert nucleotide: %s' % last_base)


def number2patten(number, length):
    """ Recurrent function for converting interger to DNA sequence.

    Args:
        number (int): Interger created by patten2number.
        length (int): Original sequence length provided to patten2number.

    Returns:
        str: DNA sequence string.
    """
    if length == 1:
        return NUMBER_TO_BASE[number]
    prefix_index = number // 4
    base = NUMBER_TO_BASE[number % 4]
    return number2patten(prefix_index, length - 1) + base


def multisize_patten2number(sequence, min_length, max_length):
    """ Converts kmers with heterogeneous size into intergers.

    Args:
          sequence (str): DNA sequence (allowing A, C, G, T uppercase only)
        min_length (int): Mininal kmer sizes.
        max_length (int): Maximum kmer sizes.

    Returns:
        int: Interger reprencitation for a four bases sequence.
    """
    lengths = np.arange(min_length, max_length + 1)
    offsets = np.concatenate(([0], 4**lengths))

    try:
        index = np.where(lengths == len(sequence))[0][0]
        return patten2number(sequence) + offsets[index]
    except IndexError:
        raise ValueError('Provided sequence length (%d nt) ' % len(sequence) +
                         'not in list of provided lengths %s nt.' % lengths)


def number2multisize_patten(number, min_length, max_length):
    """ Converts kmers with heterogeneous size into intergers.

    Args:
            number (int): Interger created by multisize_patten2number.
        min_length (int): Mininal kmer sizes.
        max_length (int): Maximum kmer sizes.

    Returns:
        str: DNA sequence string.
    """
    lengths = np.arange(min_length, max_length + 2)  # +2 Include last interval
    offsets = 4**lengths

    try:
        index = np.where((offsets - number) > 0)[0][0]
        org_length = lengths[index - 1]
        number -= np.concatenate(([0], offsets))[index - 1]
        return number2patten(number, org_length)
    except IndexError:
        raise ValueError('Provided number (%d) do not match ' % number +
                         'list of provided lengths %s nt.' % lengths)


def gc(sequence):
    """ Computes GC-ratio for a DNA sequence.

    Args:
        sequence (str): DNA sequence string

    Returns:
        float: Ratio of Gs + Cs within provided DNA string.
    """
    sequence = sequence.upper()
    return (sequence.count('G') + sequence.count('C')) / float(len(sequence))


def sequence_entropy(sequence):
    """ Computes Shannon entropy for provided DNA sequence.

    S = -sum( p_i * log2(p_i) )

    where p_i is s the probability of character number i
    showing up in sequence.

    Args:
        sequence (str): DNA sequence string

    Returns:
        float: Shannon entropy value.
    """
    c = Counter(sequence.upper())
    tot = float(sum(c.values()))
    c = {k: v / tot for k, v in c.items()}
    return -1 * sum(c * np.log2(c) for b, c in c.items())


def isgzip(filename):
    """ Function checks gzip files magic number

    Args:
        filename (str): Path to file to test.

    Returns:
        boolean: True = gzip magic number exist (this is a gzip file).
    """
    magic_number = b'\x1f\x8b\x08'
    with open(filename, 'rb') as f:
        file_start = f.read(len(magic_number))

    if magic_number == file_start:
        return True
    return False


def kmer_vector2tsv_file(filename, kmer_vector, length, enable_gzip=False):
    """ Converting the data structure kmer-vector to a human readable
    tsv file.

    Args:
            filename (str): Output tsv filename.
        kmer_vector (iter): Iterable object where index correspond to kmer
                            sequence and value equals kmer frequency.
                            (Use number2patten to convert index into sequence)
              length (int): Original sequence length of each kmer used to
                            create the vector. (see patten2number)
     enable_gzip (boolean): If true gzip compress output file.

    Returns:
        filename (str): Return full file path if successfull.
    """
    try:
        fh = gzip.open if enable_gzip else open
        with fh(filename, 'wt') as out:
            for index, count in enumerate(kmer_vector):
                seq = number2patten(index, length)
                out.write('{seq}\t{count}\n'.format(seq=seq,
                                                    count=count))
        return filename
    except Exception:
        print('Not able to create [%s]\n' % filename)
        raise


def tsv_file2kmer_vector(filename):
    """ Reads a kmer count tsv file into a numpy array object.

    Args:
        filename (str): Input kmer count tsv filename.

    Returns:
        kmer_vector (numpy): Numpy array object where index correspond to kmer
                             sequence, and value equals kmer frequency.
    """
    try:
        fh = gzip.open if isgzip(filename) else open
        with fh(filename, 'rt') as f:
            for i, rec in enumerate(f):
                seq, count = rec.strip().split()

                if i == 0:
                    length = len(seq)
                    kmer_vector = np.array([0] * (4**length),
                                           dtype=np.uint32)

                assert len(seq) == length
                kmer_vector[patten2number(seq)] += int(count)

        return kmer_vector, length

    except Exception:
        print('Not able to read file [%s]\n' % filename)
        raise


def read_faidx(filename, filter_str):
    """ Parsing fasta index file.

    Args:
           filename (str): Path to fasta index file or fasta file.
                           This function tries to guess the correct
                           path to the fasta index file if fasta file
                           path is provided.
        filter_str (list): List of strings to ignore in parent
                           name.
    Returns:
        dict: With parents/chromosome id as key and size (bp) as value.
    """
    if not filename.endswith('.fai'):
        # Try to find fai file.
        fai_candidates = [filename + '.fai',
                          filename.rstrip('.fa') + '.fai',
                          filename.rstrip('.gz') + '.fai']
        try:
            faidx_file = [os.path.isfile(c)
                          for c in fai_candidates].index(True)
            faidx_file = fai_candidates[faidx_file]
        except ValueError:
            raise IOError('Could not find valid faidx for [%s]i\n' % filename)
    else:
        faidx_file = filename

    try:
        chroms = {}
        with open(faidx_file) as faidx:
            for record in faidx:
                col = record.strip().split()
                name, length = col[0], int(col[1])
                if not any([ignore in name for ignore in filter_str]):
                    chroms[name] = length
        return chroms
    except Exception:
        print('Could not read faidx file [%s]' % faidx_file)
        raise
