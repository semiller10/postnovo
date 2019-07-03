''' Functions used across Postnovo project. '''

import config

import numpy as np
import os
import os.path
import pandas as pd
import pickle as pkl
import sys

from collections import OrderedDict, Counter
from itertools import product, combinations, combinations_with_replacement, permutations
from multiprocessing import current_process

#A global variable for tracking multithreaded process completion.
progress_count = 0

def save_pkl_objects(dir, **kwargs):
    '''
    Serialize objects using pickle dump.

    Parameters
    ----------
    dir : str
        Directory destination.
    kwargs : dict
        Filename string keys and object values.

    Returns
    -------
    None
    '''

    for obj_name, obj in kwargs.items():
        verbose_print('Saving', obj_name)
        with open(os.path.join(dir, obj_name), 'wb') as f:
            pkl.dump(obj, f, 2)

    return

def load_pkl_objects(dir, *args):
    '''
    De-serialize objects using pickle load.

    Parameters
    ----------
    dir : str
        Directory origin of files.
    args : list
        Filename string values.

    Returns
    -------
    Loaded objects, either as a single object for a single input 
    or a tuple of objects for multiple inputs.
    '''

    return_list = []
    for obj_name in args:
        verbose_print('Loading', obj_name)
        with open(os.path.join(dir, obj_name), 'rb') as f:
            return_list.append(pkl.load(f))
    if len(args) == 1:
        return return_list[0]
    else:
        return tuple(return_list)

    return

def verbose_print(*args):
    '''
    Print function that can be turned off when the command line option "verbose" is set to false.

    Parameters
    ----------
    args : list of str
        Each string will be printed on a separate line.
    
    Returns
    -------
    None
    '''

    if config.globals['Verbose']:
        for arg in args:
            print(arg, end=' ', flush=True)
        print(flush=True)

    return

def verbose_print_over_same_line(output_str):
    '''
    Print function that can be turned off when the command line option "verbose" is set to false, 
    with the text stream written over the same line.

    Parameters
    ----------
    output_str : str

    Returns
    -------
    None
    '''

    if config.globals['Verbose']:
        #ANSI escape code to clear to the end of the line in the terminal.
        sys.stdout.write('\033[K')
        sys.stdout.write(output_str + '\r')
        sys.stdout.flush()

    return

def print_percent_progress_singlethreaded(procedure_str, one_percent_total_count):
    '''
    Print the percent progress of single-threaded function calls.

    Parameters
    ----------
    procedure_str : str
        Name of procedure for printing progress of function calls.
    one_percent_total_count : float
        One percent of the number of function calls.

    Returns
    -------
    None
    '''

    global progress_count
    progress_count += 1
    if int(progress_count % one_percent_total_count) == 0:
        percent_complete = int(progress_count / one_percent_total_count)
        if percent_complete <= 100:
            verbose_print_over_same_line(procedure_str + str(percent_complete) + '%')

    return

def print_percent_progress_multithreaded(procedure_str, one_percent_total_count, cores):
    '''
    Print the percent progress of multithreaded (multiprocess) function calls.

    Parameters
    ----------
    procedure_str : str
        Name of procedure for printing progress of function calls.
    one_percent_total_count : float
        One percent of the number of function calls in each thread.
    cores : int
        Number of cores used in multiprocessing.

    Returns
    -------
    None
    '''

    if current_process()._identity[0] % cores == 1:
        global progress_count
        progress_count += 1
        if int(progress_count % one_percent_total_count) == 0:
            percent_complete = int(progress_count / one_percent_total_count)
            if percent_complete <= 100:
                verbose_print_over_same_line(procedure_str + str(percent_complete) + '%')

    return

def check_between_zero_and_one(i, interval_type='open'):
    '''
    Determine whether a number is in the open interval between 0 and 1.

    Parameters
    ----------
    i : float or str
    interval_type : {'open', 'closed'}

    Returns
    -------
    float
        Input value.

    Raises
    ------
    ValueError
        If the input number does not lie in the interval between 0 and 1.
    '''

    try:
        i = float(i)
    except:
        raise ValueError('Input cannot be cast as a float.')

    valid_interval_types = {'open', 'closed'}
    if interval_type not in valid_interval_types:
        raise ValueError('interval_type must be one of {0}'.format(valid_interval_types))

    if interval_type == 'open':
        if 0 < i < 1:
            return i
        else:
            raise ValueError('Number does not lie in open interval between 0 and 1.')
    elif interval_type == 'closed':
        if 0 <= i <= 1:
            return i
        else:
            raise ValueError('Number does not lie in closed interval between 0 and 1')
    
    return

def check_positive_nonzero_int(i):
    '''
    Determine whether an integer is non-zero and positive.

    Parameters
    ----------
    i : int or str

    Returns
    -------
    i : int
        Int-typed input value.

    Raises
    ------
    ValueError
        If the integer is negative or zero.
    '''

    try:
        i = int(i)
    except:
        raise ValueError('Input cannot be cast as an integer.')

    if i > 0:
        return i
    else:
        raise ValueError('Number is not > 0.')

    return

def check_positive_nonzero_float(i):
    '''
    Determine whether a float is non-zero and positive.

    Parameters
    ----------
    i : float or str

    Returns
    -------
    i : float
        Float-typed input value.

    Raises
    ------
    ValueError
        If the float is negative or zero.
    '''

    try:
        i = float(i)
    except:
        raise ValueError('Input cannot be cast as a float.')

    if i > 0:
        return i
    else:
        raise ValueError('Number is not > 0.')

    return

def check_filepaths(fps):
    '''
    Check the existence of multiple filepaths.

    Parameters
    ----------
    list of str

    Returns
    -------
    list of str
        Filepaths that do not exist.
    '''

    bad_fps = []
    for fp in fps:
        if not os.path.exists(fp):
            bad_fps.append(fp)

    return bad_fps

def encode_aas(seq):
    '''
    Encode amino acids of an input peptide string as an integer array.

    Parameters
    ----------
    seq : str
        Input peptide string, with properly formatted modification symbols.

    Returns
    -------
    encoded_seq : numpy array
        Integer-encoded amino acids.
    '''

    standard_aa_chars = list(config.standard_aa_mass_dict.keys())
    aa_code_dict = config.aa_code_dict
    encoded_seq = []
    if seq == '':
        encoded_seq = np.array(encoded_seq)
    else:
        #Chars in seq may include modification symbols.
        prev_char_is_not_standard_aa = False
        start_aa_index = -1
        for char_index, char in enumerate(seq):
            #Record the previously encountered aa.
            if start_aa_index == -1:
                start_aa_index = 0
            elif char in standard_aa_chars:
                encoded_seq.append(
                    aa_code_dict[seq[start_aa_index: char_index]])
                start_aa_index = char_index
                prev_char_is_not_standard_aa = False
            else:
                prev_char_is_not_standard_aa = True
        #Record the last aa.
        encoded_seq.append(aa_code_dict[seq[start_aa_index: ]])
        encoded_seq = np.array(encoded_seq)

    return encoded_seq

def find_subarray(array1, array2):
    '''
    Determine whether the first 1-D array is found in order in the second.

    Parameters
    ----------
    array1 : numpy array
    array2 : numpy array

    Returns
    -------
    subarray_start_index : int
        -1 if no match.
    '''

    last_array1_item = array1[-1]
    first_item_offset = array1.size - 1
    for i in range(array1.size - 1, array2.size):
        if last_array1_item == array2[i]:
            if all(array1 == array2[i - first_item_offset: i + 1]):
                return i - first_item_offset
    
    return -1

def find_isobaric(aa_mass_dict, max_pep_length):
    '''
    Finds isobaric and near-isobaric peptides within the length constraint.

    Parameters:
    aa_mass_dict : Ordered dict of amino acid symbol keys and mass values, 
        e.g., OrderedDict([('C+57.021', 160.03065)])
    max_pep_length : Maximum peptide length to consider, 
        e.g., 2 includes monopeptides and dipeptides

    Returns:
    permuted_isobaric_peps_dict : Dict of lists of peptides with isobaric substitutes
        for each combination of substitution lengths, e.g., 
        OrderedDict((1, 2): [('N', ), ('G', 'G'), ('Q', ), ('A', 'G'), ('G', 'A')], 
                    (2, 2): [('A', 'D'), ('D', 'A'), ('E', 'G'), ('G', 'E'), ...], 
                    ...)
    permuted_near_isobaric_peps_dict : Dict of lists of peptides with near-isobaric substitutes
        for each combination of substitution lengths, e.g., 
        OrderedDict((1, 2): [('R', ), ('G', 'V'), ('V', 'G')], 
                    (2, 2): [('C+57.021', 'L'), ('L', 'C+57.021'), ('S', 'W'), ('W', 'S'), ...], 
                    ...)
    '''

    #Record a list of peptides with each mass.
    isobaric_peps_dict = OrderedDict()
    #Loop through peptide lengths.
    for pep_length in range(1, max_pep_length + 1):
        #Loop through peptides of each length.
        for aa_tuple, mass_tuple in zip(
            combinations_with_replacement(aa_mass_dict, pep_length), 
            combinations_with_replacement(list(aa_mass_dict.values()), pep_length)):
            #Amino acid masses in config are reported to five decimals.
            pep_mass = round(sum(mass_tuple), 5)
            #Record peptides sharing each mass.
            if pep_mass in isobaric_peps_dict:
                isobaric_peps_dict[pep_mass].append(aa_tuple)
            elif pep_mass + 0.00001 in isobaric_peps_dict:
                isobaric_peps_dict[pep_mass + 0.00001].append(aa_tuple)
            elif pep_mass - 0.00001 in isobaric_peps_dict:
                isobaric_peps_dict[pep_mass - 0.00001].append(aa_tuple)
            elif pep_mass + 0.00002 in isobaric_peps_dict:
                isobaric_peps_dict[pep_mass + 0.00002].append(aa_tuple)
            elif pep_mass - 0.00002 in isobaric_peps_dict:
                isobaric_peps_dict[pep_mass - 0.00002].append(aa_tuple)
            else:
                isobaric_peps_dict[pep_mass] = [aa_tuple]

    isobaric_peps = []
    permuted_isobaric_peps_dict = OrderedDict()
    for mass, peps in isobaric_peps_dict.items():
        if len(peps) > 1:
            isobaric_peps += peps
            
            #Determine the combinations of lengths (mono-/di-peptide, di-/di-peptide, etc.)
            #at the given mass.
            pep_len_count_dict = Counter(list(map(len, peps)))
            ##Example: peptides of length 2, 3, and 4 are found 1x, 3x and 2x at the mass.
            len_combos = []
            ##len_combos = [(3, 3), (4, 4)]
            for pep_len, pep_len_count in pep_len_count_dict.items():
                if pep_len_count > 1:
                    len_combos += list(product((pep_len, ), repeat=2))
            ##len_combos = [(3, 3), (4, 4), (2, 3), (2, 4), (3, 4)]
            len_combos += list(combinations(pep_len_count_dict.keys(), 2))

            for len_combo in len_combos:
                if len_combo not in permuted_isobaric_peps_dict:
                    permuted_isobaric_peps_dict[len_combo] = []
                for pep in peps:
                    if len(pep) in len_combo:
                        permuted_isobaric_peps_dict[len_combo].append(pep)

    #There should not be a need to dereplicate for truly isobaric peptides, 
    #but since there is a small amount of rounding in the fifth decimal, do it anyways.
    isobaric_peps = list(set(isobaric_peps))
    for len_combo, peps in permuted_isobaric_peps_dict.items():
        #Permute the peptides.
        permuted_peps = []
        for pep in peps:
            #Dereplicate permutations.
            permuted_peps += list(set(permutations(pep, len(pep))))
        permuted_isobaric_peps_dict[len_combo] = permuted_peps

    permuted_isobaric_peps_dict = OrderedDict(
        sorted(permuted_isobaric_peps_dict.items()))

    #Record a list of (peptide, mass) tuples within the mass window of each (peptide, mass) key.
    near_isobaric_peps_dict = OrderedDict()
    #Loop through peptide lengths.
    for pep_length in range(1, max_pep_length + 1):
        #Loop through peptides of each length.
        for aa_tuple, mass_tuple in zip(
            combinations_with_replacement(aa_mass_dict, pep_length), 
            combinations_with_replacement(list(aa_mass_dict.values()), pep_length)):
            new_mass = sum(mass_tuple)
            #Record peptides within mass window of each peptide.
            new_pep_mass_tuple = (aa_tuple, new_mass)
            near_isobaric_peps_dict[new_pep_mass_tuple] = []
            for old_pep_mass_tuple in near_isobaric_peps_dict.keys():
                old_mass = old_pep_mass_tuple[1]
                delta_mass = round(abs(old_mass - new_mass), 5)
                #Ignore isobaric matches.
                if delta_mass > 0.00002 and delta_mass <= config.near_isobaric_window:
                    near_isobaric_peps_dict[old_pep_mass_tuple].append(new_pep_mass_tuple)
                    near_isobaric_peps_dict[new_pep_mass_tuple].append(old_pep_mass_tuple)

    near_isobaric_peps = []
    permuted_near_isobaric_peps_dict = OrderedDict()
    for pep_mass_tuple, other_pep_mass_tuples in near_isobaric_peps_dict.items():
        if len(other_pep_mass_tuples) > 0:
            near_isobaric_peps.append(pep_mass_tuple[0])
            
            #Determine the combinations of lengths (mono-/di-peptide, di-/di-peptide, etc.)
            #that are near-isobaric to the given peptide.
            pep_len_count_dict = Counter([len(pep_mass_tuple[0])] + list(map(len, [
                other_pep_mass_tuple[0] for other_pep_mass_tuple in other_pep_mass_tuples])))
            ##Example: peptides of length 2, 3, and 4 centered at the given mass 
            #are found 1x, 3x and 2x.
            len_combos = []
            ##len_combos = [(3, 3), (4, 4)]
            for pep_len, pep_len_count in pep_len_count_dict.items():
                if pep_len_count > 1:
                    len_combos += list(product((pep_len, ), repeat=2))
            ##len_combos = [(3, 3), (4, 4), (2, 3), (2, 4), (3, 4)]
            len_combos += list(combinations(pep_len_count_dict.keys(), 2))

            for len_combo in len_combos:
                if len_combo not in permuted_near_isobaric_peps_dict:
                    permuted_near_isobaric_peps_dict[len_combo] = []
                if len(pep_mass_tuple[0]) in len_combo:
                    permuted_near_isobaric_peps_dict[len_combo].append(pep_mass_tuple[0])
                for other_pep, _ in other_pep_mass_tuples:
                    if len(other_pep) in len_combo:
                        permuted_near_isobaric_peps_dict[len_combo].append(other_pep)

    #Remove isobaric peptides from list of near-isobaric peptides.
    #The peptides removed at this stage are those with isobaric matches to one set of peptides 
    #and near-isobaric matches to a different set of peptides.
    near_isobaric_peps = list(set(near_isobaric_peps).difference(set(isobaric_peps)))
    for len_combo, peps in permuted_near_isobaric_peps_dict.items():
        permuted_peps = []
        #Avoid replicate peptides from permuted_near_isobaric_peps_dict.
        for pep in set(peps):
            if pep in near_isobaric_peps:
                permuted_peps += list(set(permutations(pep, len(pep))))
        if permuted_peps == []:
            permuted_near_isobaric_peps_dict.pop(len_combo)
        else:
            permuted_near_isobaric_peps_dict[len_combo] = permuted_peps

    permuted_near_isobaric_peps_dict = OrderedDict(
        sorted(permuted_near_isobaric_peps_dict.items()))

    return permuted_isobaric_peps_dict, permuted_near_isobaric_peps_dict

def get_potential_substitution_info(
    pep, 
    aa_scores, 
    alg, 
    max_subseq_len=config.max_subseq_len, 
    all_isobaric_peps_dict=config.all_permuted_isobaric_peps_dict, 
    all_near_isobaric_peps_dict=config.all_permuted_near_isobaric_peps_dict):
    '''
    Find potential isobaric and near-isobaric substitutions in the input peptide.

    Parameters
    ----------
    pep : numpy array
        Array of encoded amino acids
    alg : str
    max_subseq_len : int
        Maximum length of subsequences to consider
    all_isobaric_peps_dict : OrderedDict object
        Dict of lists of peptides with isobaric substitutes
    all_near_isobaric_peps_dict : OrderedDict object
        Dict of lists of peptides with near-isobaric substitutes

    Returns
    -------
    isobaric_subseqs_dict : OrderedDict object
        Potential isobaric substitutes for each length combo in the sequence
    near_isobaric_subseqs_dict : OrderedDict object
        Potential near-isobaric substitutes for each length combo in the sequence
    '''

    if alg == 'Novor':
        max_score = 100
    elif alg == 'DeepNovo':
        max_score = 1

    subseq_info_dict = OrderedDict()
    #Record subsequences at each length.
    for subseq_length in range(1, max_subseq_len + 1):
        subseq_info_dict[subseq_length] = subseq_info_for_length_dict = OrderedDict()
        #Record information on each unique subsequence.
        for subseq_start_index in range(len(pep) - subseq_length + 1):
            subseq = tuple(pep[subseq_start_index: subseq_start_index + subseq_length])
            if subseq in subseq_info_for_length_dict:
                #Sum the scores from all instances of the subsequence.
                subseq_info_for_length_dict[subseq][1] += max_score - np.average(
                    aa_scores[subseq_start_index: subseq_start_index + subseq_length])
            else:
                #Record the starting position (1-indexed) of the most N-terminal subsequence.
                subseq_info_for_length_dict[subseq] = [
                    subseq_start_index + 1, 
                    np.mean(aa_scores[subseq_start_index: subseq_start_index + subseq_length])]

    isobaric_subseqs_dict = OrderedDict()
    near_isobaric_subseqs_dict = OrderedDict()
    #Consider mono-/monopeptide, mono-/dipeptide, di-/dipeptide, etc. substitutions.
    subseq_length_combos = list(combinations_with_replacement(range(1, max_subseq_len + 1), 2))

    for subseq_length_combo in subseq_length_combos:
        #Record the average N-terminal position and average score of potential substitutions.
        isobaric_subseqs_dict[subseq_length_combo] = isobaric_info_for_length_combo = [0, 0]
        near_isobaric_subseqs_dict[subseq_length_combo] = \
            near_isobaric_info_for_length_combo = [0, 0]
        #The length combo may not have any substitutions.
        if subseq_length_combo in all_isobaric_peps_dict:
            all_isobaric_peps_for_length_combo = all_isobaric_peps_dict[subseq_length_combo]
        else:
            all_isobaric_peps_for_length_combo = []
        if subseq_length_combo in all_near_isobaric_peps_dict:
            all_near_isobaric_peps_for_length_combo = all_near_isobaric_peps_dict[
                subseq_length_combo]
        else:
            all_near_isobaric_peps_for_length_combo = []
        isobaric_subseq_count = 0
        near_isobaric_subseq_count = 0
        for subseq_length in list(set(subseq_length_combo)):
            subseq_info_for_length_dict = subseq_info_dict[subseq_length]
            #Check whether each subsequence of the given length has a potential substitution.
            #To record the average N-terminal position and score, 
            #divide the sum by the count of subsequences.
            #The default N-terminal position (no subsequences found) is 0.
            for subseq, subseq_info in list(subseq_info_for_length_dict.items()):
                if subseq in all_isobaric_peps_for_length_combo:
                    isobaric_info_for_length_combo[0] += subseq_info[0]
                    isobaric_info_for_length_combo[1] += subseq_info[1]
                    isobaric_subseq_count += 1
                    #Reduce the search space for further length combos.
                    subseq_info_for_length_dict.pop(subseq) 
                if subseq in all_near_isobaric_peps_for_length_combo:
                    near_isobaric_info_for_length_combo[0] += subseq_info[0]
                    near_isobaric_info_for_length_combo[1] += subseq_info[1]
                    near_isobaric_subseq_count += 1
                    subseq_info_for_length_dict.pop(subseq)
        if isobaric_subseq_count > 0:
            isobaric_info_for_length_combo[0] /= isobaric_subseq_count
            isobaric_info_for_length_combo[1] /= isobaric_subseq_count
        if near_isobaric_subseq_count > 0:
            near_isobaric_info_for_length_combo[0] /= near_isobaric_subseq_count
            near_isobaric_info_for_length_combo[1] /= near_isobaric_subseq_count

    return isobaric_subseqs_dict, near_isobaric_subseqs_dict

def count_low_scoring_peptides(aa_scores, pep_len):
    '''
    Count the number of isolated, relatively low-scoring amino acid subsequences.
    These often are incorrect due to inversion or isobaric substitution errors.

    Parameters
    ----------
    aa_scores : numpy array
    pep_len : int
        Length of peptide subsequences to search

    Returns
    -------
    low_scoring_pep_count : int
    '''

    #If a subsequence has a score one standard deviation lower than the bounding amino acids, 
    #it is identified as isolated and relatively low-scoring.
    score_stdev = np.std(aa_scores)
    low_scoring_pep_count = 0
    last_pep_start = len(aa_scores) - pep_len
    for i in range(last_pep_start):
        is_low_scoring = False
        pep_score = np.mean(aa_scores[i: i + pep_len])
        if i == 0:
            bounding_score = aa_scores[pep_len]
            bounding_stdev = 0
        #When considering an interior subseq,
        #ensure that the bounding amino acids have similarly high scores, 
        #e.g., bounding Novor amino acid scores of 0 and 100 are almost certainly unacceptable, 
        #but scores of 80 and 90 are likely acceptable.
        elif i < last_pep_start:
            first_bounding_score = aa_scores[i - 1]
            second_bounding_score = aa_scores[i + pep_len]
            bounding_score = np.average([first_bounding_score, second_bounding_score])
            bounding_stdev = np.std([first_bounding_score, second_bounding_score])
        else:
            bounding_score = aa_scores[i - 1]
            bounding_stdev = 0
        if bounding_stdev <= score_stdev:
            if pep_score < bounding_score - score_stdev:
                low_scoring_pep_count += 1

    return low_scoring_pep_count

def calculate_qvalues(db_search_df):
    '''
    Calculate q-values from target/decoy hit frequencies.

    Parameters
    ----------
    db_search_df : pandas DataFrame
        DataFrame of MSGF+ formatted database search results.

    Returns
    -------
    db_search_df : pandas DataFrame
        DataFrame of MSGF+ formatted database search results with a new column of PSM q-values.
    '''
    
    #MSGF+ places the prefix 'XXX_' at the start of protein IDs for decoy hits.
    db_search_df['is_decoy'] = db_search_df['Protein'].apply(
        lambda id: 1 if 'XXX_' in id else 0)
    db_search_df.sort_values('SpecEValue', inplace=True)
    decoy_count = 0
    target_count = 0
    target_count_denom = target_count
    qvalues = []
    for is_decoy in db_search_df['is_decoy'].tolist():
        if is_decoy:
            decoy_count += 1
            target_count_denom = target_count
        else:
            target_count += 1

        if decoy_count == 0:
            #No decoy matches yet encountered.
            qvalues.append(0)
        else:
            try:
                qvalues.append(decoy_count / target_count_denom)
            except ZeroDivisionError:
                #Only decoy matches have been encountered (should not happen with normal datasets).
                qvalues.append(1)
    db_search_df['psm_qvalue'] = qvalues
    db_search_df.sort_values(['ScanNum', 'psm_qvalue'], inplace=True)
    #Remove lower-ranking PSMs from the same spectrum.
    db_search_df.drop_duplicates(subset='ScanNum', inplace=True)

    return db_search_df

##REMOVE: unused
#def invert_dict_of_lists(d):
#    '''
#    For a dict with values that are lists, make a new dict, 
#    with keys being unique elements of the lists 
#    and values being lists of the input keys corresponding to input lists containing the element.

#    Parameters
#    ----------
#    d : dict

#    Returns
#    -------
#    invert_d : dict
#    '''

#    values = set(a for b in d.values() for a in b)
#    values = sorted(list(values))
#    invert_d = OrderedDict((new_k, [k for k, v in d.items() if new_k in v]) for new_k in values)

#    return invert_d