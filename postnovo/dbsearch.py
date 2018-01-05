import argparse
import difflib
import os.path
import pandas as pd
import sys
import time

from collections import OrderedDict
from functools import partial

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.utils as utils
else:
    import config
    import utils

def main():
    postnovo_df = parse_args()
    merged_df = merge_predictions(postnovo_df)
    grouped_df = group_predictions(merged_df)
    retained_seq_dict = lengthen_seqs(grouped_df)
    make_fasta(retained_seq_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--postnovo_df',
                        help='best_predictions.csv file produced by postnovo')
    parser.add_argument('--msgf_tsv_list',
                        nargs='+',
                        help='list the db search output file paths to consider')
    parser.add_argument('--msgf_name_list',
                        nargs='+',
                        help='list the dataset names to associate with each of the db search output files')
    parser.add_argument('--out_dir')
    args=parser.parse_args()
    for i, fp in enumerate(args.msgf_tsv_list):
        config.psm_fp_list.append(fp)
        config.db_name_list.append(args.msgf_name_list[i])
    config.iodir.append(args.out_dir)
    postnovo_df = pd.read_csv(args.postnovo_df, header=0)
    return postnovo_df

def merge_predictions(postnovo_df):

    # Compare postnovo to psm seqs
    merged_df = postnovo_df
    # Loop through each psm dataset
    for i, psm_fp in enumerate(config.psm_fp_list):
        # Use a string to indicate the metagenome based on the filename
        db_name = config.db_name_list[i]
        # Load MSGF tsv output as search df
        psm_df = pd.read_csv(psm_fp, sep='\t', header=0)
        # There is a row of col labels
        # Retain the following cols as search df:
        # ScanNum, Precursor, Charge, Peptide, Protein
        psm_df = psm_df[['ScanNum', 'Precursor', 'Charge', 'Peptide', 'Protein', 'SpecEValue']]
        psm_df['Precursor'] = psm_df['Precursor'] * psm_df['Charge']
        psm_df.drop('Charge', axis=1, inplace=True)
        # Make a new is_decoy col to indicate decoy hits from Protein col (decoy = 1, not = 0)
        psm_df['is_decoy'] = psm_df['Protein'].apply(lambda x: 1 if 'XXX_' in x else 0)
        # Make a new col, protein_ids, to link peptide to proteins
        # Ex. protein id at this stage: 'Protein317(pre=R,post=Q);Protein692(pre=R,post=-)'
        protein_ids = psm_df['Protein'].apply(lambda x: x.split(';')).tolist()
        # Want to retain ['Protein317', 'Protein692']
        psm_df['protein_ids'] = [[id[:id.index('(pre=')] for id in id_list] for id_list in protein_ids]
        # Retain and rename cols: scan_<db name>, precursor_mass_<>, psm_seq_<>, is_decoy_<>, 1-psm_qvalue_<>, protein_ids_<>
        psm_df.drop('Protein', axis=1, inplace=True)

        psm_df.sort('SpecEValue', inplace=True)
        # Loop through rows, calculating q values of PSMs
        decoy_count = 0
        target_count = 0
        target_count_denom = target_count
        psm_df_col_list = psm_df.columns.tolist()
        decoy_col = psm_df_col_list.index('is_decoy')
        spec_evalue_col = psm_df_col_list.index('SpecEValue')
        qvalue_list = []
        for row in psm_df.itertuples(index=False):
            if row[decoy_col]:
                decoy_count += 1
                target_count_denom = target_count
            else:
                target_count += 1

            if decoy_count == 0:
                qvalue_list.append(0)
            else:
                try:
                    qvalue_list.append(decoy_count / target_count_denom)
                except ZeroDivisionError:
                    qvalue_list.append(1)
        psm_df['psm_qvalue'] = qvalue_list 
        psm_df['psm_qvalue'] = 1 - psm_df['psm_qvalue']
        psm_df.drop('SpecEValue', axis=1, inplace=True)
        psm_df.columns = [
            'scan', 
            'precursor_mass_' + db_name, 
            'psm_seq_' + db_name, 
            'is_decoy_' + db_name, 
            'protein_ids_' + db_name,
            '1-psm_qvalue_' + db_name]
        # Retain rows with 1-psm_qvalue >= 0.99
        psm_df = psm_df[psm_df['1-psm_qvalue_' + db_name] >= 0.99]
        # Retain rows where is_decoy == 0
        psm_df = psm_df[psm_df['is_decoy_' + db_name] == 0]

        # Remove PTM characters from PSM seqs and convert all isoleucines to leucines
        psm_df['psm_seq_' + db_name] = psm_df['psm_seq_' + db_name].apply(
            lambda seq: utils.remove_mod_chars(seq=seq).replace('I', 'L')
            )
        psm_df.drop('is_decoy_' + db_name, axis=1, inplace=True)

        # Retrieve full sequences from db files
        # seqs in db file may be separated by line breaks onto multiple lines
        # First simplify files that have line breaks in sequences
        # Just do this automatically to every file
        with open(config.db_fp_list[i]) as f:
            db_lines = f.readlines()
        with open(config.db_fp_list[i], 'w') as f:
            for j, line in enumerate(db_lines):
                if line[0] == '>':
                    f.write(line)
                    protein_seq = ''
                else:
                    protein_seq += line.rstrip()
                if j + 1 == len(db_lines):
                    f.write(protein_seq + '\n')
                elif db_lines[j + 1][0] == '>':
                    f.write(protein_seq + '\n')

        with open(config.db_fp_list[i]) as f:
            db_lines = f.readlines()
        protein_ids = []
        protein_seqs = []
        for i, line in enumerate(db_lines):
            if line[0] == '>':
                # With Graph2Pro, seq length (a number) is separated by the protein ID by a space
                protein_id = line.rstrip().lstrip('>')
                protein_id = protein_id[:protein_id.index(' ')]
                protein_ids.append(protein_id)
            else:
                protein_seqs.append(line.rstrip())

        protein_dict = {}
        for i, protein_id in enumerate(protein_ids):
            protein_dict[protein_id] = protein_seqs[i]

        psm_protein_id_lists = psm_df['protein_ids_' + db_name].tolist()
        psm_protein_lists = []
        for id_list in psm_protein_id_lists:
            psm_protein_list = []
            for id in id_list:
                psm_protein_list.append(protein_dict[id])
            # take the set of non-redundant protein seqs
            psm_protein_lists.append(list(set(psm_protein_list)))
        psm_df['protein_seqs_' + db_name] = psm_protein_lists

        # Merge search and postnovo df's on scan, retaining all rows
        merged_df = merged_df.merge(psm_df, how='outer', on='scan')
    # End loop

    # Make cols initialized with empty lists called 'has_predict_from' and 'best_predicts_from'
    # Lists will be stored in these cols
    db_name_list = config.db_name_list
    contains_predicts_from_list = [OrderedDict() for i in range(len(merged_df))]
    best_predicts_from_list = [OrderedDict() for i in range(len(merged_df))]
    best_predicts_list = [[] for i in range(len(merged_df))]
    best_peptide_list = [None for i in range(len(merged_df))]
    best_score_list = [None for i in range(len(merged_df))]
    # Since we will be iterating through tuples in order to preserve data types,
    # we need to map col names to tuple positions
    col_dict = OrderedDict([(merged_df.columns[i], i+1) for i in range(len(merged_df.columns))])
    # Map dataset names to tuple score and seq positions
    name_list = ['postnovo'] + config.db_name_list
    score_col_dict = OrderedDict([('postnovo', col_dict['probability'])])
    peptide_col_dict = OrderedDict([('postnovo', col_dict['seq'])])
    predict_col_dict = OrderedDict([('postnovo', col_dict['seq'])])
    for db_name in config.db_name_list:
        score_col_dict[db_name] = col_dict['1-psm_qvalue_' + db_name]
        peptide_col_dict[db_name] = col_dict['psm_seq_' + db_name]
        predict_col_dict[db_name] = col_dict['protein_seqs_' + db_name]
    row_count = 0

    # sort db search dicts by score, followed by peptide,
    # so that only the top-scoring peptide is considered if there are conflicting options
    # postnovo is still considered first, because it can have partial seq
    # ex. postnovo -> db peptide1 -> db peptide1 -> db peptide2 ignored
    for row in merged_df.itertuples():

        # sort peptides by score (descending), tacking on postnovo last
        # db search is always trusted more than postnovo
        sorted_peptide_col_dict = OrderedDict(
            sorted(
            [(name, peptide_col_dict[name]) for name in db_name_list],
            key=lambda duple: -row[score_col_dict[duple[0]]]
            ) + 
            [('postnovo', peptide_col_dict['postnovo'])]
            )

        best_predicts_sublist = best_predicts_list[row_count]
        best_predicts_from_dict = best_predicts_from_list[row_count]
        contains_predicts_from_dict = contains_predicts_from_list[row_count]

        for name, peptide_col in sorted_peptide_col_dict.items():
            if pd.notnull(row[peptide_col]):

                if best_peptide_list[row_count] == None:
                    best_peptide_list[row_count] = row[peptide_col]
                    best_score_list[row_count] = row[score_col_dict[name]]
                    if name == 'postnovo':
                        best_predict = row[predict_col_dict[name]]
                        best_predicts_sublist.append(best_predict)
                        best_predicts_from_dict[best_predict] = [name]
                        contains_predicts_from_dict[best_predict] = [name]
                    else:
                        # screen out best predicts that are substrings of others
                        best_predicts = sorted(
                            [best_predict for best_predict in row[predict_col_dict[name]]],
                             key=lambda best_predict: -len(best_predict)
                             )
                        best_predicts_sublist.append(best_predicts[0])
                        best_predicts_from_dict[best_predicts[0]] = [name]
                        contains_predicts_from_dict[best_predicts[0]] = [name]
                        for additional_seq in best_predicts[1:]:
                            already_added = False
                            for seq in best_predicts_sublist:
                                if additional_seq in seq:
                                    already_added = True
                                    break
                            if not already_added:
                                best_predicts_sublist.append(additional_seq)
                                best_predicts_from_dict[additional_seq] = [name]
                                contains_predicts_from_dict[additional_seq] = [name]

                elif row[peptide_col] in best_peptide_list[row_count]:
                    # do not add lower probability postnovo seq to db seqs
                    # but check if it is the same as any db seqs
                    if name == 'postnovo':
                        additional_seq = row[predict_col_dict[name]]
                        for i, seq in enumerate(best_predicts_sublist):
                            if additional_seq == seq:
                                best_predicts_from_dict[seq].append(name)
                                contains_predicts_from_dict[seq].append(name)
                            elif additional_seq in seq:
                                contains_predicts_from_dict[seq].append(name)
                    # treat all db search results sharing the top PSM equally
                    else:
                        additional_seqs = row[predict_col_dict[name]]
                        # check if additional seqs are redundant/substring
                        for additional_seq in additional_seqs:
                            redundant_seqs = []
                            redundant_contains = []
                            already_added = False
                            for i, seq in enumerate(best_predicts_sublist):
                                # if the new seq under consideration is already represented
                                if additional_seq == seq:
                                    best_predicts_from_dict[seq].append(name)
                                    contains_predicts_from_dict[seq].append(name)
                                    already_added = True
                                elif additional_seq in seq:
                                    contains_predicts_from_dict[seq].append(name)
                                    already_added = True
                                # if a previously added seq is a substring of the new seq
                                elif seq in additional_seq:
                                    redundant_seqs.append(seq)
                                    redundant_contains += contains_predicts_from_dict[seq]

                            if not already_added:
                                # Remove "redundant" predicts
                                if redundant_seqs:
                                    for seq in redundant_seqs:
                                        best_predicts_sublist.remove(seq)
                                        del(best_predicts_from_dict[seq])
                                        del(contains_predicts_from_dict[seq])
                                best_predicts_sublist.append(additional_seq)
                                best_predicts_from_dict[additional_seq] = [name]
                                contains_predicts_from_dict[additional_seq] = [name] + list(set(redundant_contains))
        row_count += 1

    # Make a single col of masses, rather than a col for each database name
    merged_mass_col = []
    mass_col_headers = ['measured mass'] + ['precursor_mass_' + name for name in config.db_name_list]
    mass_col_lists = [merged_df[header].tolist() for header in mass_col_headers]
    for row in range(len(mass_col_lists[0])):
        for col in range(len(mass_col_lists)):
            mass = mass_col_lists[col][row]
            if pd.notnull(mass):
                merged_mass_col.append(mass)
                break
    merged_df['measured mass'] = merged_mass_col
    for header in mass_col_headers[1:]:
        merged_df.drop(header, axis=1, inplace=True)

    merged_df['best_predicts'] = best_predicts_list
    merged_df['best_predicts_from'] = best_predicts_from_list
    merged_df['contains_predicts_from'] = contains_predicts_from_list
    merged_df['best_peptide'] = best_peptide_list
    merged_df['best_score'] = best_score_list
    retained_cols = ['scan', 'seq', 'probability', 'measured mass']
    for name in config.db_name_list:
        retained_cols.append('psm_seq_' + name)
        retained_cols.append('1-psm_qvalue_' + name)
    retained_cols += ['best_predicts', 'best_predicts_from', 'contains_predicts_from', 'best_peptide', 'best_score']
    merged_df = merged_df[retained_cols]

    return merged_df

def group_predictions(input_df):

    # after assembling fasta seqs, check to make sure no seqs are in any other seqs
    # remove duplicate shorter seqs

    # If only de novo sequencing and not database search was performed,
    # then merge_predictions was not called.
    # input_df therefore lacks certain columns which are now added
    if 'best_predicts' not in input_df.columns:
        input_df['best_predicts'] = input_df['seq']
        input_df['best_predicts_from'] = [['postnovo']] * len(input_df)
        input_df['contains_predicts_from'] = [['postnovo']] * len(input_df)
        input_df['best_peptide'] = input_df['seq']
        input_df['best_score'] = input_df['probability']
        retained_cols = ['scan', 'seq', 'measured mass', 'best_predicts', 'best_predicts_from', 'contains_predicts_from', 'best_peptide', 'best_score']
        input_df = input_df[retained_cols]

    # make preliminary mass groups
    # have a list of final mass groups
    # keep track of the last assigned final group (bigger than each preliminary mass group)
    # loop through group to determine seq similarity
    # seq similarity > 
    # ex. preliminary mass group
    # scan  seq         prob        mass        error       final group
    # 18920	LTVEEAK	    0.610231138	1075.575276	0.004302301 N
    # 24760	LTEDLGGLEK	0.892213125	1075.575276	0.004302301 N+1
    # 18825	VDALTVEEAK	0.574638	1075.576276	0.004302305 N
    # 24856	LTEDLGGLEK	0.819234005	1075.576276	0.004302305 N+1
    # 24953	LTEDLELNK	0.7073737	1075.576276	0.004302305 N+1
    # assign group N to row 0
    # make a list of chars in seq 0: [L, T, V, E, E, A, K]
    # make a list of chars in seq 1: [L, T, E, D, L, G, G, L, E, K]
    # loop through chars in shorter seq (seq 0)
    # search for char in longer seq (seq 1)
    # count matches and mismatches
    # if found in longer seq list, del char
    # at the end of this example: seq 1 list = [D, L, G, G, L], matches = 5, mismatches = 2
    # if matches / # aa in shorter seq >= 0.77 (7/9), add to group N
    # else do not add to a new final group
    # final groups are recorded in corresponding list
    # proceeding to the next comparison
    # seq 0 list: [L, T, V, E, E, A, K]
    # seq 2 list: [V, D, A, L, T, V, E, E, A, K]
    # matches = 7, mismatches = 0 => add to group N
    # ... moving onto second loop starting with second seq:
    # this seq is assigned to group N+1 (last assigned final group + 1)
    # ignore seq comparisons to seqs with final group assignment
    # first comparison: seq 1, seq 3 => 100% match
    # second (last) comparison: seq 1, seq 4
    # seq 1 list: [L, T, E, D, L, G, G, L, E, K]
    # seq 4 list: [L, T, E, D, L, E, L, N, K]
    # matches = 8, mismatches = 1

    #UNCOMMENT
    input_df['mass error'] = input_df['measured mass'] * (config.precursor_mass_tol[0] * 10**-6)
    # df with cols: scan, seq, prob, mass, mass error, [best_peptide], [etc]
    # sort by mass
    input_df.sort('measured mass', inplace=True)
    # extract mass list, mass error list
    mass_list = input_df['measured mass'].apply(float).tolist()
    mass_error_list = input_df['mass error'].apply(float).tolist()
    # assign mass 0 to group 0
    current_prelim_mass_group = 0
    prelim_mass_group_list = [current_prelim_mass_group]
    # loop through each mass, to mass n-1
    for i in range(len(mass_list[:-1])):
        # if next mass outside mass error of mass
        if mass_list[i] + mass_error_list[i] < mass_list[i+1] - mass_error_list[i+1]:
            current_prelim_mass_group += 1
        prelim_mass_group_list.append(current_prelim_mass_group)
    # add mass group col to df
    input_df['prelim mass group'] = prelim_mass_group_list

    current_final_mass_group = -1
    final_mass_group_list = []
    # min seq similarity = 7/9 rounded down to third digit
    min_seq_similarity = 0.777
    scan_proximity = 300
    # scan proximity bonus = 1/9 rounded down to third digit
    scan_proximity_bonus = 0.111
    # length difference penalty = 1/9 rounded down to third digit
    length_diff_penalty = 0.111
    length_diff_multiplier = 3

    col_list = input_df.columns.tolist()
    seq_col = col_list.index('best_peptide')

    for prelim_mass_group in set(prelim_mass_group_list):
        prelim_mass_group_df = input_df[input_df['prelim mass group'] == prelim_mass_group]

        local_final_mass_group_list = [-1] * len(prelim_mass_group_df)
        for first_row_index in range(len(prelim_mass_group_df)):
            first_scan = prelim_mass_group_df['scan'].iloc[first_row_index]
            # If the current spectrum has not been assigned to a final mass group,
            # assign it to the next incremented final mass group
            if local_final_mass_group_list[first_row_index] == -1:
                current_final_mass_group += 1
            first_seq = prelim_mass_group_df.iloc[first_row_index, seq_col]

            for second_row_index in range(first_row_index + 1, len(prelim_mass_group_df)):
                if (local_final_mass_group_list[first_row_index] == -1 or
                    local_final_mass_group_list[second_row_index] == -1):

                    first_seq_list = list(first_seq)
                    second_seq = prelim_mass_group_df.iloc[second_row_index, seq_col]
                    second_seq_list = list(second_seq)
                    second_scan = prelim_mass_group_df['scan'].iloc[second_row_index]

                    if len(first_seq) <= len(second_seq):
                        shorter_seq_list = first_seq_list
                        longer_seq_list = second_seq_list
                    else:
                        shorter_seq_list = second_seq_list
                        longer_seq_list = first_seq_list

                    match_count = 0
                    mismatch_count = 0
                    for aa in shorter_seq_list:
                        try:
                            del(longer_seq_list[longer_seq_list.index(aa)])
                            match_count += 1
                        except ValueError:
                            mismatch_count += 1

                    seq_similarity = match_count / len(shorter_seq_list)
                    if abs(first_scan - second_scan) <= scan_proximity:
                        adjusted_seq_similarity = (seq_similarity + scan_proximity_bonus - 
                                                    (abs(len(first_seq) - len(second_seq)) // length_diff_multiplier * length_diff_penalty)
                                                    )
                    else:
                        adjusted_seq_similarity = (seq_similarity - 
                                                    (abs(len(first_seq) - len(second_seq)) // length_diff_multiplier * length_diff_penalty)
                                                    )

                    if adjusted_seq_similarity >= min_seq_similarity:
                        if ((local_final_mass_group_list[first_row_index] == -1) and 
                            (local_final_mass_group_list[second_row_index] == -1)):
                            local_final_mass_group_list[first_row_index] = current_final_mass_group
                            local_final_mass_group_list[second_row_index] = current_final_mass_group
                        else:
                            if local_final_mass_group_list[first_row_index] == -1:
                                local_final_mass_group_list[first_row_index] = local_final_mass_group_list[second_row_index]
                            if local_final_mass_group_list[second_row_index] == -1:
                                local_final_mass_group_list[second_row_index] = local_final_mass_group_list[first_row_index]

            if local_final_mass_group_list[first_row_index] == -1:
                local_final_mass_group_list[first_row_index] = current_final_mass_group

        final_mass_group_list += local_final_mass_group_list

    input_df['final mass group'] = final_mass_group_list
    input_df.drop('prelim mass group', axis=1, inplace=True)

    return input_df

def lengthen_seqs(mass_grouped_df):

    min_overlap = 5

    mass_grouped_df_cols = mass_grouped_df.columns.tolist()
    postnovo_seq_col = mass_grouped_df_cols.index('seq')
    score_col = mass_grouped_df_cols.index('best_score')

    def lengthen_postnovo_seq(final_mass_group_df):

        first_seq = final_mass_group_df.iloc[0]['seq']
        avg_score = final_mass_group_df.iloc[0]['best_score']

        # Loop through subsequent postnovo seqs
        for row in final_mass_group_df.iloc[1:].itertuples(index=False):
            second_seq = row[postnovo_seq_col]
            try:
                # Do not consider subseqs or identical seqs
                if second_seq not in first_seq:
                    seq_matcher_obj = difflib.SequenceMatcher(None, first_seq, second_seq)
                    # Find all matching subseqs
                    matching_blocks_list = seq_matcher_obj.get_matching_blocks()
                    for block in matching_blocks_list:
                        # Look for sufficiently strong overlap at the ends -- seq extensions
                        # this means overlap of left side of first seq with second
                        # ex. first seq = 'bcde', second seq = 'abcd'
                        if ((block[0] == 0) and 
                            (block[1] + block[2] == len(second_seq)) and 
                            (block[2] >= min_overlap)):
                            second_seq_score = row[score_col]
                            second_seq_annex_len = len(second_seq) - block[2]
                            merged_seq_len = len(first_seq) + second_seq_annex_len
                            avg_score = ((avg_score * len(first_seq) / merged_seq_len) + 
                                            (second_seq_score * second_seq_annex_len / merged_seq_len))
                            first_seq = second_seq[:block[1]] + first_seq
                        # this means overlap of right side of first seq with second
                        # ex. first seq = 'abcd', second seq = 'bcde'
                        elif ((block[1] == 0) and
                                (block[0] + block[2] == len(first_seq)) and
                                (block[2] >= min_overlap)):
                            second_seq_score = row[score_col]
                            second_seq_annex_len = len(second_seq) - block[2]
                            merged_seq_len = len(first_seq) + second_seq_annex_len
                            avg_score = ((avg_score * len(first_seq) / merged_seq_len) + 
                                            (second_seq_score * second_seq_annex_len / merged_seq_len))
                            first_seq = first_seq + second_seq[block[2]:]
            # This should not occur! Check that and delete
            except TypeError:
                pass

        return first_seq, avg_score

    retained_seq_dict = {
        'mass_group_list': [],
        'mass_list': [],
        'scan_list_of_lists': [],
        'best_predicts': [],
        'best_score': [],
        'best_predicts_from': [],
        'contains_predicts_from': []
        }

    if config.psm_fp_list:
        multiple_origins_possible = True
        mass_grouped_df_cols = mass_grouped_df.columns.tolist()
        best_predicts_col = mass_grouped_df_cols.index('best_predicts')
        best_predicts_from_col = mass_grouped_df_cols.index('best_predicts_from')
        contains_predicts_from_col = mass_grouped_df_cols.index('contains_predicts_from')
    else:
        multiple_origins_possible = False

    final_mass_group_dfs = [
        df for _, df in mass_grouped_df.sort('best_score', ascending=False).groupby('final mass group')]
    for final_mass_group_df in final_mass_group_dfs:

        retained_seq_dict['mass_group_list'].append(final_mass_group_df['final mass group'].iloc[0])
        retained_seq_dict['scan_list_of_lists'].append(final_mass_group_df['scan'].tolist())
        retained_seq_dict['mass_list'].append(final_mass_group_df['measured mass'].iloc[0])

        # loop through each row in the final_mass_group_df
        # all db search proteins are given equal weight
        # first, find the origin of predictions
        # create a full list of has_predicts_from
        # if the only origin is postnovo, then try merging seq predictions
        # else only look for redundant seqs

        if multiple_origins_possible:
            best_predicts_from_list = list(set(
                [db_name for best_predicts_from_dict in final_mass_group_df['best_predicts_from'].tolist()
                 for _, db_name_list in best_predicts_from_dict.items()
                 for db_name in db_name_list]
                ))

            if best_predicts_from_list == ['postnovo']:
                long_seq, avg_score = lengthen_postnovo_seq(final_mass_group_df)
                retained_seq_dict['best_predicts'].append([long_seq])
                retained_seq_dict['best_predicts_from'].append(OrderedDict([(long_seq, ['postnovo'])]))
                retained_seq_dict['best_score'].append(avg_score)
                retained_seq_dict['contains_predicts_from'].append(OrderedDict([(long_seq, ['postnovo'])]))

            else:
                consolid_predicts = final_mass_group_df.iloc[0]['best_predicts']
                consolid_best_predicts_from_dict = final_mass_group_df.iloc[0]['best_predicts_from']
                consolid_contains_predicts_from_dict = final_mass_group_df.iloc[0]['contains_predicts_from']
                for row in final_mass_group_df.iloc[1:].itertuples(index=False):
                    for additional_seq in row[best_predicts_col]:
                        unique_predict = True
                        for seq in consolid_predicts:
                            if additional_seq == seq:
                                unique_predict = False
                                # the identical seq may arise from different db's
                                additional_best_predicts_from_set = set(row[best_predicts_from_col][additional_seq])
                                consolid_best_predicts_from_set = set(consolid_best_predicts_from_dict[seq])
                                additional_contains_predicts_from_set = set(row[contains_predicts_from_col][additional_seq])
                                consolid_contains_predicts_from_set = set(consolid_contains_predicts_from_dict[seq])
                                consolid_best_predicts_from_dict[seq] = list(consolid_best_predicts_from_set.union(additional_best_predicts_from_set))
                                consolid_contains_predicts_from_dict[seq] = list(consolid_contains_predicts_from_set.union(additional_contains_predicts_from_set))
                            elif additional_seq in seq:
                                unique_predict = False
                                additional_contains_predicts_from_set = set(row[contains_predicts_from_col][additional_seq])
                                consolid_contains_predicts_from_set = set(consolid_contains_predicts_from_dict[seq])
                                consolid_contains_predicts_from_dict[seq] = list(consolid_contains_predicts_from_set.union(additional_contains_predicts_from_set))
                                
                        if unique_predict:
                            consolid_predicts.append(additional_seq)
                            consolid_best_predicts_from_dict[additional_seq] = row[best_predicts_from_col][additional_seq]
                            consolid_contains_predicts_from_dict[additional_seq] = row[contains_predicts_from_col][additional_seq]
                retained_seq_dict['best_predicts'].append(consolid_predicts)
                retained_seq_dict['best_predicts_from'].append(consolid_best_predicts_from_dict)
                retained_seq_dict['contains_predicts_from'].append(consolid_contains_predicts_from_dict)
                retained_seq_dict['best_score'].append(final_mass_group_df.iloc[0]['best_score'])
                
        else:
            # COME BACK TO THIS!
            long_seq, avg_score = lengthen_postnovo_seq(final_mass_group_df)
            retained_seq_dict['best_predicts'].append([long_seq])
            retained_seq_dict['best_predicts_from'].append(OrderedDict([(long_seq, ['postnovo'])]))
            retained_seq_dict['contains_predicts_from'].append(OrderedDict([(long_seq, ['postnovo'])]))
            retained_seq_dict['best_score'].append(avg_score)

    return retained_seq_dict

def make_fasta(retained_seq_dict):

    scan_list_of_lists = retained_seq_dict['scan_list_of_lists']
    best_predicts_list = retained_seq_dict['best_predicts']
    fasta_best_predicts_list = [seq for seq_list in best_predicts_list for seq in seq_list]
    best_score_list = retained_seq_dict['best_score']
    best_predicts_from_list = retained_seq_dict['best_predicts_from']
    contains_predicts_from_list = retained_seq_dict['contains_predicts_from']
    fasta_scan_lists_list = []
    fasta_best_score_list = []
    fasta_best_predicts_from_list = []
    fasta_contains_predicts_from_list = []
    for i, seq_list in enumerate(best_predicts_list):
        scan_list = scan_list_of_lists[i]
        best_score = best_score_list[i]
        best_predicts_from_dict = best_predicts_from_list[i]
        contains_predicts_from_dict = contains_predicts_from_list[i]
        for seq in seq_list:
            fasta_scan_lists_list.append(scan_list)
            fasta_best_score_list.append(best_score)
            fasta_best_predicts_from_list.append(best_predicts_from_dict[seq])
            fasta_contains_predicts_from_list.append(contains_predicts_from_dict[seq])

    # Eliminate redundancy in seqs:
    # this is chiefly (if not entirely) caused by protein-level redundancy,
    # i.e., spectra being associated with different peptides from the same protein
    # Make a list of seqs slated for removal due to redundancy
    removal_list = [0 for i in range(len(fasta_best_predicts_list))]
    # Loop through each seq
    for i, first_seq in enumerate(fasta_best_predicts_list):
        if removal_list[i] == 0:
            # Loop through each subsequent seq to see if first seq is subseq of second
            for j in range(i + 1, len(fasta_best_predicts_list)):
                if removal_list[j] == 0:
                    second_seq = fasta_best_predicts_list[j]
                    if first_seq == second_seq:
                        removal_list[i] = 1
                        fasta_scan_lists_list[j] = list(set(fasta_scan_lists_list[j]).union(set(fasta_scan_lists_list[i])))
                        fasta_best_score_list[j] = max([fasta_best_score_list[i], fasta_best_score_list[j]])
                        fasta_best_predicts_from_list[j] = list(set(fasta_best_predicts_from_list[i]).union(set(fasta_best_predicts_from_list[j])))
                        fasta_contains_predicts_from_list[j] = list(set(fasta_contains_predicts_from_list[i]).union(set(fasta_contains_predicts_from_list[j])))
                        break
                    elif first_seq in second_seq:
                        removal_list[i] = 1
                        fasta_scan_lists_list[j] = list(set(fasta_scan_lists_list[j]).union(set(fasta_scan_lists_list[i])))                 
                        fasta_best_score_list[j] = max(fasta_best_score_list[i], fasta_best_score_list[j])
                        fasta_best_predicts_from_list[j] = list(set(fasta_best_predicts_from_list[i]).union(set(fasta_best_predicts_from_list[j])))
                        fasta_contains_predicts_from_list[j] = list(set(fasta_contains_predicts_from_list[i]).union(set(fasta_contains_predicts_from_list[j])))
                        break
                    elif second_seq in first_seq:
                        removal_list[j] = 1
                        fasta_scan_lists_list[i] = list(set(fasta_scan_lists_list[j]).union(set(fasta_scan_lists_list[i])))
                        fasta_best_score_list[i] = max(fasta_best_score_list[i], fasta_best_score_list[j])
                        fasta_best_predicts_from_list[i] = list(set(fasta_best_predicts_from_list[i]).union(set(fasta_best_predicts_from_list[j])))
                        fasta_contains_predicts_from_list[i] = list(set(fasta_contains_predicts_from_list[i]).union(set(fasta_contains_predicts_from_list[j])))

    # The origin of all of the seqs in the mass group are recorded,
    # but individual seqs should be matched to individual datasets.
    # It would difficult and tedious to change the earlier code
    # to keep track of each seq by origin.
    # Open all of the database files, placing in a dict keyed by db name.
    # Find the db's from best_predicts_from list containing the seq
    # Place these db names at the start of the list of db's corresponding to the seq
    # Search the remaining db's for the 

    info_table = [['seq_number', 'scan_list', 'seq_score', 'best_predicts_from', 'also_contains_predicts_from']]
    seq_number = 0
    with open(os.path.join(config.iodir[0], 'postnovo_seqs.faa'), 'w') as fasta_file:
        for i, remove in enumerate(removal_list):
            if not remove:
                fasta_seq = fasta_best_predicts_list[i]
                if len(fasta_seq) >= config.min_blast_query_len:
                    best_predicts_from = fasta_best_predicts_from_list[i]
                    also_contains_predicts_from = list(set(fasta_contains_predicts_from_list[i]).difference(set(fasta_best_predicts_from_list[i])))
                    info_table.append([
                        str(seq_number),
                        ','.join(sorted(map(str, fasta_scan_lists_list[i]))),
                        str(round(fasta_best_score_list[i], 2)),
                        ','.join(sorted(best_predicts_from)),
                        ','.join(sorted(also_contains_predicts_from))
                        ])

                    fasta_header = ('>' + 
                                    '(seq_number)' + str(seq_number) + 
                                    '\n')
                    fasta_file.write(fasta_header)
                    fasta_file.write(fasta_seq + '\n')

                    seq_number += 1
    pd.DataFrame(info_table[1:], columns=info_table[0]).to_csv(os.path.join(config.iodir[0], 'postnovo_seqs_info.tsv'), sep='\t', index=False)

def make_xle_permutations(seq, residue_number = 0, permuted_seqs = None):

    if permuted_seqs == None:
        permuted_seqs = []

    if residue_number == len(seq):
        permuted_seqs.append(seq)
        return permuted_seqs
    else:
        if seq[residue_number] == 'L':
            permuted_seq = seq[: residue_number] + 'I' + seq[residue_number + 1:]
            permuted_seqs = make_xle_permutations(permuted_seq, residue_number + 1, permuted_seqs)
        permuted_seqs = make_xle_permutations(seq, residue_number + 1, permuted_seqs)
        return permuted_seqs

if __name__ == '__main__':
    main()