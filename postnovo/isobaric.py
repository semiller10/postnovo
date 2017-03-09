''' Identify possible isobaric substitutions in sequences '''

import pandas as pd

from config import *
from utils import *

from multiprocessing import Pool, current_process

multiprocessing_seq_count = 0


def update_prediction_df(prediction_df):
    verbose_print()
    verbose_print('finding possible isobaric substitutions')

    seqs = prediction_df['seq']

    ## single processor method
    #possible_substitution_lists = []
    #child_initialize(isobaric_substitutions, near_isobaric_substitutions)
    #for seq in seqs:
    #    possible_substitution_lists.append(find_possible_substitutions(seq))

    ## multiprocessing method
    one_percent_number_seqs = len(seqs) / cores[0] / 100
    multiprocessing_pool = Pool(cores[0],
                                initializer = child_initialize,
                                initargs = (mono_dipeptide_isobaric_substitutions, dipeptide_isobaric_substitutions,
                                            mono_dipeptide_near_isobaric_substitutions, dipeptide_near_isobaric_substitutions,
                                            cores[0], one_percent_number_seqs)
                                )
    possible_substitution_lists = multiprocessing_pool.map(find_possible_substitutions, seqs)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    isobaric_df = pd.DataFrame(possible_substitution_lists,
                               index = prediction_df.index,
                               columns = ['possible mono-dipeptide isobaric substitutions',
                                          'possible dipeptide isobaric substitutions',
                                          'possible mono-dipeptide near isobaric substitutions',
                                          'possible dipeptide near isobaric substitutions'])
    prediction_df = pd.concat([prediction_df, isobaric_df], axis = 1)

    return prediction_df

def child_initialize(_mono_dipeptide_isobaric_substitutions, _dipeptide_isobaric_substitutions,
                     _mono_dipeptide_near_isobaric_substitutions, _dipeptide_near_isobaric_substitutions,
                     _cores = 1, _one_percent_number_seqs = None):
    global mono_dipeptide_isobaric_substitutions, dipeptide_isobaric_substitutions,\
           mono_dipeptide_near_isobaric_substitutions, dipeptide_near_isobaric_substitutions,\
           cores, one_percent_number_seqs
    mono_dipeptide_isobaric_substitutions = _mono_dipeptide_isobaric_substitutions
    dipeptide_isobaric_substitutions = _dipeptide_isobaric_substitutions
    mono_dipeptide_near_isobaric_substitutions = _mono_dipeptide_near_isobaric_substitutions
    dipeptide_near_isobaric_substitutions = _dipeptide_near_isobaric_substitutions
    cores = _cores
    one_percent_number_seqs = _one_percent_number_seqs

def find_possible_substitutions(seq):

    if current_process()._identity[0] % cores == 1:
        global multiprocessing_seq_count
        multiprocessing_seq_count += 1
        if int(multiprocessing_seq_count % one_percent_number_seqs) == 0:
            percent_complete = int(multiprocessing_seq_count / one_percent_number_seqs)
            if percent_complete <= 100:
                verbose_print_over_same_line('inter-spectrum comparison progress: ' + str(percent_complete) + '%')

    possible_substitution_list = []

    mono_dipeptide_isobaric_substitution_count = 0
    for peptide in mono_dipeptide_isobaric_substitutions:
        if peptide in seq:
            mono_dipeptide_isobaric_substitution_count += 1
    possible_substitution_list.append(mono_dipeptide_isobaric_substitution_count)

    dipeptide_isobaric_substitution_count = 0
    for peptide in dipeptide_isobaric_substitutions:
        if peptide in seq:
            dipeptide_isobaric_substitution_count += 1
    possible_substitution_list.append(dipeptide_isobaric_substitution_count)

    mono_dipeptide_near_isobaric_substitution_count = 0
    for peptide in mono_dipeptide_near_isobaric_substitutions:
        if peptide in seq:
            near_isobaric_substitution_count += 1
    possible_substitution_list.append(mono_dipeptide_near_isobaric_substitution_count)

    dipeptide_near_isobaric_substitution_count = 0
    for peptide in dipeptide_near_isobaric_substitutions:
        if peptide in seq:
            dipeptide_near_isobaric_substitution_count += 1
    possible_substitution_list.append(dipeptide_near_isobaric_substitution_count)

    return possible_substitution_list