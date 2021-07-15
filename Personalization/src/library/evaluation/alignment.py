import re

from Bio.Seq import Seq
from Bio import Align


def compute_alignment(s1, s2):
    """
    Computes the best alignment for the given strings.
    :param s1: The first string. This should be the
    :param s2:
    :return:
    """
    free_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    free_chars -= set(s1) | set(s2)

    # Replace all non-alphabetic characters by unused alphabetic characters.
    repl_table = dict()
    matches = re.findall('[\W0-9]', s1 + s2)

    for match in matches:
        if match not in repl_table:
            repl = free_chars.pop()
            repl_table[match] = repl

        s1 = s1.replace(match, repl_table[match])
        s2 = s2.replace(match, repl_table[match])

    seq1 = Seq(s1)
    seq2 = Seq(s2)

    aligner = Align.PairwiseAligner()
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    alignments = aligner.align(seq1, seq2)

    alignments = alignments[0].format().split('\n')
    s1_aligned = alignments[0].replace('-', '#')
    s2_aligned = alignments[2].replace('-', '#')

    for char, repl in repl_table.items():
        s1_aligned = s1_aligned.replace(repl, char)
        s2_aligned = s2_aligned.replace(repl, char)

    return s1_aligned, s2_aligned


def main():
    s1, s2 = compute_alignment('reference', 'refolance')
    print(s1, s2, sep='\n')

    s1, s2 = compute_alignment('reference', 'refolce')
    print(s1, s2, sep='\n')


if __name__ == '__main__':
    main()
