"""
Utilities Module for DNA Sequence.
"""
import numpy as np
from collections import Counter

__author__ = 'Magnus Isaksson'
__credits__ = ['Magnus Isaksson']
__version__ = '0.1.0'

BASE_TO_NUMBER = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
NUMBER_TO_BASE = ('A', 'C', 'G', 'T')


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

    S = -∑ p_i * log2(p_i)

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


