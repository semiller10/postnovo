'''
Find the minimum de novo sequence length to uniquely match a reference proteome.
'''

from postnovo.classifier import load_fasta_ref_file

#from classifier import load_fasta_ref_file

import argparse
import re
import scipy.stats

from bisect import bisect_left
from functools import partial
from multiprocessing import Pool
from random import randint
from statsmodels.stats.proportion import proportion_confint

def find_min_seq_len(fasta_ref_path = None, fasta_ref = None, target_confidence_level = 0.95, number_subseqs = 1000, cores = 1):

    if fasta_ref == None:
        fasta_ref = load_fasta_ref_file(fasta_ref_path)
    fasta_ref_index = make_fasta_ref_index(fasta_ref)
    #min_correct_seq_len, unaltered_subseqs = find_min_correct_seq_len(fasta_ref, fasta_ref_index, target_confidence_level, number_subseqs, cores)
    min_incorrect_seq_len = find_min_incorrect_seq_len(fasta_ref, fasta_ref_index, target_confidence_level, number_subseqs, cores)

    #return max([min_correct_seq_len, min_incorrect_seq_len])
    #return min_correct_seq_len
    return min_incorrect_seq_len

def make_fasta_ref_index(fasta_ref):
    """Compute the position of the first residue of each sequence in the overall sequence list.

    Returns
    -------
    fasta_ref_index : list of int
        Each element is the position of the first residue of each sequence in the overall sequence list,
        and there is an additional element at the end for the total number of residues.

    """
    
    fasta_ref_index = [0]
    for seq in fasta_ref:
        fasta_ref_index.append(fasta_ref_index[-1] + len(seq))

    return fasta_ref_index

#def find_min_correct_seq_len(fasta_ref, fasta_ref_index, target_confidence_level, number_subseqs, cores):
#    """Find the minimum sequence length that uniquely matches the reference

#    Returns
#    -------

#    min_subseq_len : int
    
#    unaltered_subseqs : list of str
#        The subseqs tested at `min_subseq_len`.
    
#    """

#    measured_confidence_level = 0
#    subseq_len = 7
#    while measured_confidence_level < target_confidence_level:

#        subseqs = draw_subseqs(fasta_ref, fasta_ref_index, number_subseqs, subseq_len)
        
#        #subseq_matches = []
#        #for i, subseq in enumerate(subseqs):
#        #    subseq_matches.append(match_subseq(subseq, fasta_ref))

#        #Find the number of matches of each subseq to the fasta ref
#        multiprocessing_pool = Pool(cores)
#        single_var_match_subseq = partial(match_subseq, fasta_ref = fasta_ref)
#        subseq_matches = multiprocessing_pool.map(single_var_match_subseq, subseqs)
#        multiprocessing_pool.close()
#        multiprocessing_pool.join()

#        # Calculate the 95% binomial confidence interval
#        # for the proportion of 1-match subseqs as opposed to >1-match subseqs.
#        one_match_subseqs = subseq_matches.count(1)
#        ci = proportion_confint(one_match_subseqs, len(subseq_matches), method = 'wilson')
#        # The true proportion of 1-match subseqs must be >= 0.95 (95% of the time)
#        measured_confidence_level = ci[0]
        
#        print('seq len = {}, ci = {}-{}'.format(subseq_len, ci[0], ci[1]))

#        subseq_len += 1
#    min_subseq_len = subseq_len - 1
#    return min_subseq_len, subseqs

def find_min_incorrect_seq_len(fasta_ref, fasta_ref_index, target_confidence_level, number_subseqs, cores):
    """Confirm that seqs containing a common type of de novo error that are at least as long as unique correct seqs cannot be found in the ref"""

    measured_confidence_level = 0
    subseq_len = 7
    while measured_confidence_level < target_confidence_level:

        unaltered_subseqs = draw_subseqs(fasta_ref, fasta_ref_index, number_subseqs, subseq_len)
        altered_subseqs = invert_residues(unaltered_subseqs)

        #subseq_matches = []
        #for subseq in altered_subseqs:
        #    subseq_matches.append(match_subseq(subseq, fasta_ref))

        #Find the number of matches of each subseq to the fasta ref
        multiprocessing_pool = Pool(cores)
        single_var_match_subseq = partial(match_subseq, fasta_ref = fasta_ref)
        subseq_matches = multiprocessing_pool.map(single_var_match_subseq, altered_subseqs)
        multiprocessing_pool.close()
        multiprocessing_pool.join()

        # Calculate the 95% binomial confidence interval
        # for the proportion of 0-match subseqs as opposed to >0-match subseqs.
        zero_match_subseqs = subseq_matches.count(0)
        ci = proportion_confint(zero_match_subseqs, len(subseq_matches), method = 'wilson')
        # The true proportion of 0-match subseqs must be >= 0.95 (95% of the time)
        measured_confidence_level = ci[0]

        print('seq len = {}, ci = {}-{}'.format(subseq_len, ci[0], ci[1]))
        
        subseq_len += 1
    min_subseq_len = subseq_len - 1
    return min_subseq_len

def draw_subseqs(fasta_ref, fasta_ref_index, number_subseqs, subseq_len):

    # Keep track of the subseqs drawn
    # Each subseq index has two numbers: [parent seq number, residue number in parent seq]
    # Each subseq index is added to the list of indices
    used_subseq_indices = []
    subseqs = []
    for i in range(number_subseqs):
        subseq = ''
        while subseq == '':
            used_subseq_index = []
            # Randomly choose a residue from the ref file
            rand_residue = randint(0, fasta_ref_index[-1] - 1)
            # Find the seq in which the residue lies
            seq_index = bisect_left(fasta_ref_index, rand_residue)
            if fasta_ref_index[seq_index] != rand_residue:
                seq_index -= 1
            used_subseq_index.append(seq_index)
            seq = fasta_ref[seq_index]
            # Ensure that the protein is long enough to be sampled
            # for a subseq of length `subseq_len`
            try:
                subseq_start = randint(0, len(seq) - subseq_len)
                subseq = seq[subseq_start: subseq_start + subseq_len]
                used_subseq_index.append(subseq_start)
            except ValueError:
                pass
            # Ensure that the subseq has not already been sampled
            if used_subseq_index in used_subseq_indices:
                subseq = ''
            else:
                used_subseq_indices.append(used_subseq_index)
        subseqs.append(subseq)

    return subseqs

def match_subseq(subseq, fasta_ref):

    match_count = 0
    for seq in fasta_ref:
        match_count += len(re.findall('(?={0})'.format(subseq), seq))

    return match_count

def invert_residues(subseqs, inversion_len = 2):

    altered_subseqs = []
    for subseq in subseqs:
        target_residues = ''
        while target_residues == '':
            inversion_start = randint(0, len(subseq) - inversion_len)
            target_residues = subseq[inversion_start: inversion_start + inversion_len]
            # If the order of the target residues is the same when inverted,
            # determine whether all the residues in the subseq are the same,
            # in which case, disregard the subseq,
            # and if not all the residues are the same,
            # keep sampling the subseq for target residues that can be inverted to produce a different subseq.
            if target_residues == target_residues[::-1]:
                first_residue = subseq[0]
                for residue in subseq[1:]:
                    if residue != first_residue:
                        target_residues = ''
                        break
                else:
                    break
            else:
                altered_subseqs.append(
                    subseq[: inversion_start]
                    + target_residues[::-1]
                    + subseq[inversion_start + inversion_len:])

    return altered_subseqs

#def make_l_i_permutations(seq, residue_number = 0, permuted_seqs = []):

#    if residue_number == len(seq):
#        permuted_seqs.append(seq)
#    else:
#        if seq[residue_number] == 'L':
#            permuted_seq = seq[: residue_number] + 'I' + seq[residue_number + 1:]
#            make_l_i_permutations(permuted_seq, residue_number + 1)
#        make_l_i_permutations(seq, residue_number + 1)
#        return permuted_seqs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Find the minimum de novo sequence length to uniquely match the reference proteome')
    parser.add_argument('--fasta_ref_path',
                        help = 'path to reference proteome')
    parser.add_argument('--target_confidence_level', default = 0.95, type = float,
                        help = 'confidence level for unique sequence match')
    parser.add_argument('--cores', default = 1, type = int,
                        help = 'number of cores available for use')
    args = parser.parse_args()

    print('Minimum sequence length required = ' +
          str(find_min_seq_len(fasta_ref_path = args.fasta_ref_path, target_confidence_level = args.target_confidence_level, cores = args.cores)))

    #fasta_ref_path = 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\test\\human.faa'
    #target_confidence_level = 0.95
    #cores = 3
    #print('Minimum sequence length required = ' +
    #      str(find_min_seq_len(fasta_ref_path = fasta_ref_path, target_confidence_level = target_confidence_level, cores = cores)))