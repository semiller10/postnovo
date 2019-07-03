''' Analyze the top sequence candidates from de novo algorithms '''

import config
import utils

import gc
import multiprocessing
import pandas as pd

from functools import partial

def do_single_alg_procedure():
    '''
    Add additional Postnovo features to top de novo sequence candidates.

    Parameters
    ----------
    None

    Returns
    -------
    single_alg_prediction_df : pandas DataFrame
    '''

    single_alg_prediction_df = pd.DataFrame()
    for alg in config.globals['De Novo Algorithms']:
        utils.verbose_print(
            'Calculating Postnovo metrics for top-ranked de novo sequence candidates ' + 
            'from amino acid scores and potential isobaric/near-isobaric substitutions')
        for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
            #Load the DataFrame for the algorithm/fragment mass tolerance.
            input_df = utils.load_pkl_objects(
                config.globals['Output Directory'], 
                alg + '.' + config.globals['MGF Filename'] + '.' + frag_mass_tol + '.pkl')

            #The DataFrame multiindex is spectrum ID (equal to scan number) and candidate rank.
            input_df['Is ' + alg + ' Sequence'] = 1
            input_df[frag_mass_tol] = 1
            #Select top-ranked single-algorithm sequence predictions.
            input_df = input_df[input_df.index.get_level_values('Rank') == 0]
            #In test mode, all de novo predictions, including short ones, 
            #are retained for comparison to Postnovo results.
            #In predict and train modes, shorter predictions can be disregarded.
            input_df = input_df[
                input_df['Sequence Length'] >= config.globals['Minimum Postnovo Sequence Length']]

            if alg == 'Novor' or alg == 'DeepNovo':
                encoded_seqs = input_df['Encoded Sequence'].tolist()
                aa_scores_arrays = input_df[alg + ' Amino Acid Scores'].tolist()

                #Calculate metrics regarding relatively low-scoring subsequences.
                #Consider subsequences of length 2.
                partial_count_low_scoring_peptides = partial(
                    utils.count_low_scoring_peptides, pep_len=2)

                ##Single process
                #low_scoring_dipeptide_counts = []
                #for aa_scores in aa_scores_arrays:
                #    low_scoring_dipeptide_counts.append(
                #        partial_count_low_scoring_peptides(aa_scores))

                #Multiprocessing
                mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
                low_scoring_dipeptide_counts = mp_pool.map(
                    partial_count_low_scoring_peptides, aa_scores_arrays)
                mp_pool.close()
                mp_pool.join()

                input_df[alg + ' Low-Scoring Dipeptide Count'] = low_scoring_dipeptide_counts
                del(low_scoring_dipeptide_counts)
                
                #Consider subsequences of length 3.
                partial_count_low_scoring_peptides = partial(
                    utils.count_low_scoring_peptides, pep_len=3)

                ##Single process
                #low_scoring_tripeptide_counts = []
                #for aa_scores in aa_scores_arrays:
                #    low_scoring_tripeptide_counts.append(
                #        partial_count_low_scoring_peptides(aa_scores))

                #Multiprocessing
                mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
                low_scoring_tripeptide_counts = mp_pool.map(
                    partial_count_low_scoring_peptides, aa_scores_arrays)
                mp_pool.close()
                mp_pool.join()

                input_df[alg + ' Low-Scoring Tripeptide Count'] = low_scoring_tripeptide_counts
                del(low_scoring_tripeptide_counts)

                #Calculate metrics regarding potential (near-)isobaric substitutions.
                ##Single process
                #return_values = []
                #for encoded_seq, aa_scores in zip(encoded_seqs, aa_scores_arrays):
                #    return_values.append(
                #        utils.get_potential_substitution_info(encoded_seq, aa_scores, alg))

                #Multiprocessing
                partial_get_potential_substitution_info = partial(
                    utils.get_potential_substitution_info, alg=alg)
                zipped_args = zip(encoded_seqs, aa_scores_arrays)
                mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
                return_values = mp_pool.starmap(
                    partial_get_potential_substitution_info, zipped_args)
                mp_pool.close()
                mp_pool.join()

                isobaric_mono_di_sub_scores = []
                isobaric_di_sub_scores = []
                near_isobaric_mono_di_sub_scores = []
                near_isobaric_di_sub_scores = []
                isobaric_mono_di_sub_avg_positions = []
                isobaric_di_sub_avg_positions = []
                near_isobaric_mono_di_sub_avg_positions = []
                near_isobaric_di_sub_avg_positions = []
                for isobaric_subseqs_dict, near_isobaric_subseqs_dict in return_values:
                    isobaric_mono_di_sub_scores.append(isobaric_subseqs_dict[(1, 2)][1])
                    isobaric_di_sub_scores.append(isobaric_subseqs_dict[(2, 2)][1])
                    near_isobaric_mono_di_sub_scores.append(near_isobaric_subseqs_dict[(1, 2)][1])
                    near_isobaric_di_sub_scores.append(near_isobaric_subseqs_dict[(2, 2)][1])
                    isobaric_mono_di_sub_avg_positions.append(isobaric_subseqs_dict[(1, 2)][0])
                    isobaric_di_sub_avg_positions.append(isobaric_subseqs_dict[(2, 2)][0])
                    near_isobaric_mono_di_sub_avg_positions.append(
                        near_isobaric_subseqs_dict[(1, 2)][0])
                    near_isobaric_di_sub_avg_positions.append(
                        near_isobaric_subseqs_dict[(2, 2)][0])

                input_df[alg + ' Isobaric Mono-Dipeptide Substitution Score'] = \
                    isobaric_mono_di_sub_scores
                input_df[alg + ' Isobaric Dipeptide Substitution Score'] = isobaric_di_sub_scores
                input_df[alg + ' Near-Isobaric Mono-Dipeptide Substitution Score'] = \
                    near_isobaric_mono_di_sub_scores
                input_df[alg + ' Near-Isobaric Dipeptide Substitution Score'] = \
                    near_isobaric_di_sub_scores
                input_df[alg + ' Isobaric Mono-Dipeptide Substitution Average Position'] = \
                    isobaric_mono_di_sub_avg_positions
                input_df[alg + ' Isobaric Dipeptide Substitution Average Position'] = \
                    isobaric_di_sub_avg_positions
                input_df[alg + ' Near-Isobaric Mono-Dipeptide Substitution Average Position'] = \
                    near_isobaric_mono_di_sub_avg_positions
                input_df[alg + ' Near-Isobaric Dipeptide Substitution Average Position'] = \
                    near_isobaric_di_sub_avg_positions

            single_alg_prediction_df = pd.concat([single_alg_prediction_df, input_df])
            del(input_df)
            gc.collect()

    for is_alg_name in config.globals['De Novo Algorithm Origin Headers']:
        single_alg_prediction_df[is_alg_name].fillna(0, inplace=True)
    for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
        single_alg_prediction_df[frag_mass_tol].fillna(0, inplace=True)

    return single_alg_prediction_df