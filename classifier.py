''' Sequence accuracy classification model '''

import config
import input
import utils

from config import code_aa_dict, MIN_REF_MATCH_LEN, mod_code_standard_code_dict
from utils import encode_aas, find_subarray

import csv
import datetime
import gc
import math
import matplotlib
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os.path
import numpy as np
import pandas as pd
import pickle as pkl
import random
import sklearn as sk
import sklearn.metrics
import sys
import time
import warnings

from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from os.path import join, basename
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Set the backend to make image files on a server.
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

def classify(prediction_df):
    '''
    Classify de novo sequence predictions as accurate or inaccurate.

    Parameters
    ----------
    prediction_df : DataFrame object

    Returns
    -------
    None
    '''

    if config.globals['Mode'] == 'test' or config.globals['Mode'] == 'train':
        #Load "ground truth" database search results.
        utils.verbose_print(
            'Loading', basename(config.globals['Database Search Output Filepath']))
        #A default range index is used here.
        db_search_df = pd.read_csv(
            config.globals['Database Search Output Filepath'], sep='\t', header=0)

        if config.globals['FDR Cutoff'] < 1:
            db_search_df = utils.calculate_qvalues(db_search_df)
            #Filter to PSMs with a q-value meeting the cutoff.
            db_search_df = db_search_df[db_search_df['psm_qvalue'] <= config.globals['FDR Cutoff']]
            db_search_df.drop('psm_qvalue', axis=1, inplace=True)

        if config.globals['Reconcile Spectrum IDs']:
            #Update the scan number relating the spectrum in the MSGF+ results to the MGF file.
            #Postnovo format_mgf equates the spectrum ID and scan number in each spectrum header.
            #format_mgf also contains the old and new information in each header.
            old_id_new_id_dict = dict()
            with open(config.globals['MGF Filepath']) as in_f:
                for line in in_f:
                    if 'TITLE=' == line[:6]:
                        line_fragment = line.split(', Index: ')[1]
                        new_id, line_fragment = line_fragment.split(', Old index: ')
                        new_id = int(new_id)
                        old_id = int(line_fragment.split(', Old scan: ')[1].rstrip())
                        old_id_new_id_dict[old_id] = new_id
            #'ScanNum' is the stable spectrum identifier relating the results to the MGF file.
            db_search_df['ScanNum'] = db_search_df['ScanNum'].apply(
                lambda s: old_id_new_id_dict[s])
            del(old_id_new_id_dict)
            gc.collect()

        #Write the table of filtered PSMs to file.
        if config.globals['FDR Cutoff'] < 1:
            filtered_db_search_fp = \
                os.path.splitext(config.globals['Database Search Output Filepath'])[0] + '.' + \
                str(config.globals['FDR Cutoff']) + '.tsv'
            db_search_df.to_csv(
                filtered_db_search_fp, 
                sep='\t', 
                index=False, 
                quoting=csv.QUOTE_NONE, 
                float_format='%.8g')

        #Retain the relevent columns of the database search results for sequence comparison.
        db_search_df = db_search_df[['ScanNum', 'Peptide']]
        #Rename the spectrum identifier, "ScanNum", 
        #to the equivalent identifier for sequences in Postnovo, "Spectrum ID".
        db_search_df.columns=['Spectrum ID', 'Reference Sequence']
        db_search_df['Encoded Reference Sequence'] = db_search_df['Reference Sequence'].apply(
            lambda s: s.replace('I', 'L')).apply(utils.encode_aas)

        #######################################################################################
        #UNCOMMENT
        utils.verbose_print('Finding de novo sequence matches to database search PSMs')
        #Reset the Spectrum ID + Rank multiindex.
        prediction_df.reset_index(inplace=True)
        if 'index' in prediction_df.columns:
            prediction_df.drop('index', axis=1, inplace=True)
        prediction_df = prediction_df.merge(db_search_df, how='left', on='Spectrum ID')
        encoded_ref_seqs = []
        #For the practical purpose of a sequence comparison in each row, 
        #null (numpy nan) entries, which are typed as "float", are converted to empty numpy arrays.
        #pandas fillna cannot fill cells with a mutable object, so do the following loop.
        for encoded_ref_seq in prediction_df['Encoded Reference Sequence'].tolist():
            if type(encoded_ref_seq) == float:
                encoded_ref_seqs.append(np.array([]))
            else:
                encoded_ref_seqs.append(encoded_ref_seq)
        prediction_df['Encoded Reference Sequence'] = encoded_ref_seqs
        seq_matches = []
        for encoded_de_novo_seq, encoded_ref_seq in list(zip(
            prediction_df['Encoded Sequence'], prediction_df['Encoded Reference Sequence'])):
            ##REMOVE
            ##Distinguish between spectra possessing and lacking a database search PSM.
            #if encoded_ref_seq.size == 0:
            #    seq_matches.append(-1)
            #else:
            #    if find_subarray(encoded_de_novo_seq, encoded_ref_seq) == -1:
            #        seq_matches.append(0)
            #    else:
            #        seq_matches.append(1)
            if find_subarray(encoded_de_novo_seq, encoded_ref_seq) == -1:
                seq_matches.append(0)
            else:
                seq_matches.append(1)
        prediction_df['Sequence Matches Database Search PSM'] = seq_matches

        #Sometimes, de novo sequencing can identify spectra missed by database search.
        #This can be due to identification of part of the peptide by de novo sequencing 
        #while identification of more of the peptide would be required for a database search PSM.
        #It can also be due to the better performance 
        #of de novo sequencing performance than database search for certain spectra.
        #De novo sequences that match the reference fasta but not the database search PSMs 
        #should be further investigated as likely to be correct.
        utils.verbose_print('Loading', basename(config.globals['Reference Fasta Filepath']))
        ref_fasta_seqs = []
        with open(config.globals['Reference Fasta Filepath']) as in_f:
            for line in in_f:
                if line[0] == '>':
                    next_seq_in_next_line = True
                elif line != '\n':
                    if next_seq_in_next_line:
                        #Considering the sequence line after the header.
                        ref_fasta_seqs.append(line.strip().replace('I', 'L'))
                        next_seq_in_next_line = False
                    else:
                        #Considering an extension of the previous sequence line: 
                        #a limited number of amino acids are on each line.
                        ref_fasta_seqs[-1] += line.strip().replace('I', 'L')
        #The longer the match between the de novo sequence and the reference fasta sequence, 
        #the more likely the match is to be correct (not a false positive).
        #Filter de novo sequences by the empirically determined minimum sequence length 
        #needed for a statistically significant match to a sequence database of a given size.
        #Here is a back-of-the-envelope calculation that confirms the results of empirical trials 
        #in which sequences of varying lengths were randomly drawn from fasta files 
        #and searched back against the fasta 
        #to find the number of unique versus non-unique (chance) matches.
        #Assume the probability of any amino acid occurring is 0.05.
        #Assume a random distribution of amino acids in sequences.
        #Assume a proteome of 20,000 proteins of 500 amino acids in average length 
        #(about the size of the human proteome of canonical proteins).
        #The probability that a given sequence of 7 amino acids does not occur by chance is about: 
        #(1 - 0.05^7)^(20000 * 500) = (1 - 8*10^-10)^(10^7) 
        #= 1 - 8*10^-10 * 10^7 (by binomial approximation) = 0.992
        #The probability would be ~0.92 for a sequence of 6 amino acids 
        #and 0.99992 for a sequence of 9 amino acids.
        #Given 50,000 spectra with de novo sequence predictions, 
        #10,000 of which have matching de novo and database search sequences, 
        #then 320 of the remaining 40,000 de novo sequences (assuming all are of length 7) 
        #would be expected to match the reference fasta by chance -- 
        #about 3% the number of de novo sequences "verified" by database search 
        #would be false positives from reference fasta matching.
        #This level may be undesirable, especially for training the Postnovo random forest models.
        #A minimum sequence length of 9 is used by Postnovo, 
        #which in the previous example is expected to produce 3 false positive sequences (0.03%).
        utils.verbose_print((
            'Finding de novo sequence matches to the reference fasta '
            'for spectra without a database search PSM'))
        #For sequences meeting the minimum length cutoff, 
        #recode modified amino acids as their corresponding unmodified amino acids.
        #The sequences need to be hashed, so convert them back into strings.
        seqs_for_ref_fasta_search = []
        for encoded_seq, db_search_match in zip(
            prediction_df['Encoded Sequence'].tolist(), 
            prediction_df['Sequence Matches Database Search PSM'].tolist()):
            seq_for_ref_fasta_search = ''
            if db_search_match != 1:
                if encoded_seq.size >= MIN_REF_MATCH_LEN:
                    for code in encoded_seq:
                        if code in mod_code_standard_code_dict:
                            seq_for_ref_fasta_search += code_aa_dict[
                                mod_code_standard_code_dict[code]]
                        else:
                            seq_for_ref_fasta_search += code_aa_dict[code]
            seqs_for_ref_fasta_search.append(seq_for_ref_fasta_search)
        prediction_df['Sequence for Reference Fasta Search'] = seqs_for_ref_fasta_search

        #Search each unique qualifying sequence against the reference fasta.
        seqs_for_ref_fasta_search = list(set(seqs_for_ref_fasta_search))
        seqs_for_ref_fasta_search.remove('')

        ##Single-threaded
        #one_percent_number_seqs = len(seqs_for_ref_fasta_search) / 100
        #print_percent_progress_fn = partial(
        #    utils.print_percent_progress_singlethreaded, 
        #    procedure_str='Reference fasta search progress: ', 
        #    one_percent_total_count=one_percent_number_seqs)
        #partial_match_seq_to_ref_fasta = partial(
        #    match_seq_to_ref_fasta, 
        #    ref_fasta_seqs=ref_fasta_seqs, 
        #    print_percent_progress_fn=print_percent_progress_fn)
        #ref_fasta_matches = []
        #for seq_for_ref_fasta_search in seqs_for_ref_fasta_search:
        #    ref_fasta_matches.append(partial_match_seq_to_ref_fasta(seq_for_ref_fasta_search))

        #Multithreaded
        one_percent_number_seqs = \
            len(seqs_for_ref_fasta_search) / 100 / config.globals['CPU Count']
        mp_pool = multiprocessing.Pool(config.globals['CPU Count'])
        print_percent_progress_fn = partial(
            utils.print_percent_progress_multithreaded, 
            procedure_str='Reference fasta search progress: ', 
            one_percent_total_count=one_percent_number_seqs, 
            cores=config.globals['CPU Count'])
        partial_match_seq_to_ref_fasta = partial(
            match_seq_to_ref_fasta, 
            ref_fasta_seqs=ref_fasta_seqs, 
            print_percent_progress_fn=print_percent_progress_fn)
        ref_fasta_matches = mp_pool.map(
            partial_match_seq_to_ref_fasta, seqs_for_ref_fasta_search)
        mp_pool.close()
        mp_pool.join()
        del(ref_fasta_seqs)
        gc.collect()

        #Map each unique qualifying sequence to the result of the fasta reference search.
        seq_match_dict = dict()
        for seq_for_ref_fasta_search, ref_fasta_match in zip(
            seqs_for_ref_fasta_search, ref_fasta_matches):
            seq_match_dict[seq_for_ref_fasta_search] = ref_fasta_match
        del(seqs_for_ref_fasta_search, ref_fasta_matches)
        gc.collect()
        prediction_df['Exclusive Reference Fasta Match'] = [
            0 if seq == '' else seq_match_dict[seq] 
            for seq in prediction_df['Sequence for Reference Fasta Search'].tolist()]
        prediction_df.drop('Sequence for Reference Fasta Search', axis=1, inplace=True)

        #It is useful to include amino acid sequence strings in the prediction table 
        #for comparison to sequences predicted by individual de novo algorithms.
        seqs = []
        for encoded_seq in prediction_df['Encoded Sequence']:
            seq = ''
            for code in encoded_seq:
                seq += code_aa_dict[code]
            seqs.append(seq)
        prediction_df['Sequence'] = seqs

        ##REMOVE: debugging relic
        #prediction_df.to_pickle(os.path.splitext(config.globals['MGF Filepath'])[0] + '.comparison_df.pkl')
        ##REMOVE: debugging relic
        #prediction_df = pd.read_pickle(os.path.splitext(config.globals['MGF Filepath'])[0] + '.comparison_df.pkl')

        find_sequence_correctnesses(prediction_df)

    if config.globals['Mode'] == 'predict' or config.globals['Mode'] == 'test':
        #Predict sequence scores using Postnovo's random forest models.
        #The "test_plots" option in test mode also tabulates binary classification statistics 
        #and plots precision-recall and precision-yield curves in this function call.
        if config.globals['Mode'] == 'predict':
            prediction_df = predict_rf_scores(prediction_df)
        elif config.globals['Mode'] == 'test':
            prediction_df = predict_rf_scores(prediction_df, db_search_df=db_search_df)

        prediction_df = prediction_df.reset_index().set_index('Spectrum ID')
        #The user has the option in "predict" mode 
        #to select longer sequence predictions at the expense of accuracy.
        if config.globals['Maximum Postnovo Sequence Probability Sacrifice'] > 0:
            utils.verbose_print('Trading de novo sequence prediction accuracy for length')
            #Find the longest prediction for each spectrum meeting the score sacrifice conditions.
            #Remove predictions that do not meet the minimum estimated probability threshold.
            above_sacrifice_floor_df = prediction_df[
                prediction_df['Random Forest Score'] >= 
                config.globals['Minimum Postnovo Sequence Probability']]
            spectrum_dfs = [
                spectrum_df for _, spectrum_df 
                in above_sacrifice_floor_df.groupby(level='Spectrum ID')]
            del(above_sacrifice_floor_df)
            gc.collect()

            max_total_sacrifice = config.globals['Maximum Postnovo Sequence Probability Sacrifice']
            sacrifice_extension_ratio = config.globals[
                'Maximum Postnovo Sequence Probability Sacrifice Per Percent Length Extension'] \
                    * 100
            one_percent_spectrum_count = len(spectrum_dfs) / 100 / config.globals['CPU Count']

            ##Single-threaded
            #print_percent_progress_fn = partial(
            #    utils.print_percent_progress_singlethreaded, 
            #    procedure_str='Score-length tradeoff progress: ', 
            #    one_percent_total_count=one_percent_spectrum_count * config.globals['CPU Count'])
            #initialize_score_sacrifice_extension_workers(
            #    print_percent_progress_fn, 
            #    max_total_sacrifice, 
            #    sacrifice_extension_ratio)
            #selected_seq_dfs = []
            #for spectrum_df in spectrum_dfs:
            #    selected_seq_dfs.append(do_score_sacrifice_extension(spectrum_df))

            #Multiprocessing
            print_percent_progress_fn = partial(
                utils.print_percent_progress_multithreaded, 
                procedure_str='Score-length tradeoff progress: ', 
                one_percent_total_count=one_percent_spectrum_count, 
                cores=config.globals['CPU Count'])
            mp_pool = multiprocessing.Pool(
                config.globals['CPU Count'], 
                initializer=initialize_score_sacrifice_extension_workers, 
                initargs=(
                    print_percent_progress_fn, 
                    max_total_sacrifice, 
                    sacrifice_extension_ratio))
            selected_seq_dfs = mp_pool.map(do_score_sacrifice_extension, spectrum_dfs)
            mp_pool.close()
            mp_pool.join()

            del(spectrum_dfs)
            gc.collect()

            reported_prediction_df = pd.concat(selected_seq_dfs)
        else:
            best_prediction_df = prediction_df[
                prediction_df['Random Forest Score'] == 
                prediction_df.groupby(level='Spectrum ID')['Random Forest Score'].transform(max)]
            reported_prediction_df = best_prediction_df.groupby(level='Spectrum ID').first()

        #In "test" mode, make a plot of true versus predicted accuracy 
        #for the test dataset results from the full model.
        #The procedure is a simpler version of what occurs in the function, plot_leave_one_out.
        if config.globals['Make Test Plots']:
            binned_df = bin_data_by_score(prediction_df, config.globals['Dataset Name'], 'Final')

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            score_bin_midpoints = binned_df['Score Bin Midpoint'].values
            ax1.plot(
                score_bin_midpoints, 
                binned_df['Proportion Correct'].values, 
                linestyle='solid', 
                linewidth=0.5)
            ax2.plot(
                score_bin_midpoints, 
                binned_df['Number of Spectra'].values, 
                linestyle='dashed', 
                linewidth=0.5)

            #Plot one-to-one line, where predicted matches true proportion of correct predictions.
            ax1.plot(np.arange(0.005, 1, 0.01), np.arange(0.005, 1, 0.01), color='black')
            ax1.set_xlabel('Postnovo score (binned every 0.01)')
            ax1.set_ylabel('Proportion of spectra with accurate predictions')
            ax2.set_ylabel('Number of spectra (dashed)')

            fig.savefig(
                os.path.join(
                    config.globals['Output Directory'], 
                    'Final.true_vs_predicted_accuracy.pdf'), 
                bbox_inches='tight')

            del(binned_df)
            gc.collect()

        #In "predict" mode, the reported table needs a column of sequence strings.
        #It is only needed for the report, not for other operations, unlike in "test" and "train."
        if config.globals['Mode'] == 'predict':
            seqs = []
            for encoded_seq in reported_prediction_df['Encoded Sequence']:
                seq = ''
                for code in encoded_seq:
                    seq += code_aa_dict[code]
                seqs.append(seq)
            reported_prediction_df['Sequence'] = seqs
        reported_prediction_df['Sequence Length'] = reported_prediction_df[
            'Encoded Sequence'].apply(lambda encoded_seq: encoded_seq.size)
        #Given enough training data, random forest score estimates probability of correctness.
        reported_prediction_df.rename(
            columns={'Random Forest Score': 'Estimated Probability'}, inplace=True)
        reported_prediction_df = reported_prediction_df[
            reported_prediction_df['Estimated Probability'] >= config.min_prob]
        reported_prediction_df = reported_prediction_df.reset_index().set_index(
            ['Spectrum ID'] + 
            config.globals['De Novo Algorithm Origin Headers'] + 
            config.globals['Fragment Mass Tolerances'])
        #Select and reorder the reported columns.
        reported_prediction_df = reported_prediction_df[sorted(
            [col for col in reported_prediction_df.columns if col in config.reported_df_cols], 
            key=lambda col: config.reported_df_cols.index(col))]

        reported_prediction_df.to_csv(
            os.path.join(config.globals['Output Directory'], 'best_predictions.tsv'), 
            sep='\t', 
            float_format='%.5g')

    if config.globals['Mode'] == 'train':
        utils.verbose_print('Saving the new Postnovo training dataset')
        prediction_df = prediction_df.reset_index().set_index('Spectrum ID')
        if 'index' in prediction_df.columns:
            prediction_df.drop('index', axis=1, inplace=True)
        #The potential redundancy of the training dataset name was checked in the userargs module.
        prediction_df.drop([
            'Sequence', 
            'Encoded Sequence', 
            'Reference Sequence', 
            'Encoded Reference Sequence'], axis=1).to_csv(
                config.globals['Postnovo Training Dataset Filepath'], sep='\t')

        #This user option stops the training process just before creating the random forest models.
        if config.globals['Stop Before Training']:
            if os.path.exists(config.globals['Postnovo Training Record Filepath']):
                with open(config.globals['Postnovo Training Record Filepath'], 'a') as out_f:
                    out_f.write('\t'.join([
                        config.globals['Dataset Name'], 
                        str(datetime.datetime.now()).split('.')[0],
                       '\n']))
            else:
                with open(config.globals['Postnovo Training Record Filepath'], 'w') as out_f:
                    out_f.write('\t'.join(['Dataset', 'Timestamp', '\n']))
                    out_f.write('\t'.join([
                        config.globals['Dataset Name'], 
                        str(datetime.datetime.now()).split('.')[0],
                       '\n']))
            return

        train_models(prediction_df)

    return

def train_models(prediction_df=None):
    '''
    This part of the training workflow is placed in a separate function 
    to allow Postnovo models to be trained through the "retrain" option 
    without the (re-)creation of the associated data structures.

    Parameters
    ----------
    prediction_df : pandas DataFrame object
    
    Returns
    -------
    None
    '''

    utils.verbose_print('Loading all Postnovo training data')
    train_df = pd.DataFrame()
    if os.path.exists(config.globals['Postnovo Training Record Filepath']):
        train_dataset_names = pd.read_csv(
            config.globals['Postnovo Training Record Filepath'], sep='\t', header=0)[
                'Dataset'].tolist()
        for train_dataset_name in train_dataset_names:
            train_dataset_fp = os.path.join(
                config.globals['Postnovo Training Directory'], train_dataset_name + '.tsv')
            train_dataset_df = pd.read_csv(train_dataset_fp, sep='\t', header=0)
            train_dataset_df['Dataset'] = train_dataset_name
            train_df = pd.concat([train_df, train_dataset_df], ignore_index=True)
            del(train_dataset_df)
            gc.collect()
    else:
        #Create the Postnovo training data record file, writing the header line.
        with open(config.globals['Postnovo Training Record Filepath'], 'w') as out_f:
            out_f.write('\t'.join(['Dataset', 'Timestamp', '\n']))

    if not config.globals['Retrain']:
        #Write the record of the new training dataset.
        with open(config.globals['Postnovo Training Record Filepath'], 'a') as out_f:
            out_f.write('\t'.join([
                config.globals['Dataset Name'], str(datetime.datetime.now()).split('.')[0], '\n']))
        #Add the new training data to the full Postnovo training data.
        prediction_df['Dataset'] = config.globals['Dataset Name']
        prediction_df.reset_index(inplace=True)
        train_df = pd.concat([train_df, prediction_df], ignore_index=True)
        del(prediction_df)
        gc.collect()

    #Make random forests from the training data.
    make_rf_models(train_df, config.globals['Plot Feature Importance'])

    if config.globals['Leave One Out']:
        utils.verbose_print('Conducting leave-one-out analysis')

        #Perform leave-one-out analyses with each training dataset.
        train_df.reset_index(inplace=True)
        if 'level_0' in train_df.columns:
            train_df.drop('level_0', axis=1, inplace=True)
        train_dataset_names = pd.read_csv(
            config.globals['Postnovo Training Record Filepath'], sep='\t', header=0)[
                'Dataset'].tolist()
        for train_dataset_name in train_dataset_names:
            utils.verbose_print(
                '\nConducting leave-one-out analysis with dataset: ' + train_dataset_name)

            #Train the random forest models with the other datasets.
            subsampled_train_df = train_df[train_df['Dataset'] != train_dataset_name]
            make_rf_models(subsampled_train_df, leave_one_out_dataset_name=train_dataset_name)
            del(subsampled_train_df)
            gc.collect()

            #Predict sequence correctnesses for the spectra in the reserved dataset.
            subsampled_test_df = train_df[train_df['Dataset'] == train_dataset_name]
            find_sequence_correctnesses(subsampled_test_df)
            subsampled_test_df = predict_rf_scores(
                subsampled_test_df, leave_one_out_dataset_name=train_dataset_name)

            #Delete the saved random forest models.
            for is_alg_key in config.globals['De Novo Algorithm Origin Header Keys']:
                rf_model_key = tuple(
                    [alg for i, alg 
                        in enumerate(config.globals['De Novo Algorithms']) if is_alg_key[i]])
                rf_model_name = '-'.join([rf_model_alg for rf_model_alg in rf_model_key])
                try:
                    os.remove(
                        os.path.join(
                            config.globals['Postnovo Training Directory'], 
                            rf_model_name + 
                            '.leave_one_out.' + 
                            train_dataset_name + 
                            '.rf.pkl'))
                except FileNotFoundError:
                    continue

            #Find the best sequence prediction for each spectrum.
            subsampled_test_df = subsampled_test_df.reset_index().set_index('Spectrum ID')
            subsampled_test_df = subsampled_test_df[
                subsampled_test_df['Random Forest Score'] == 
                subsampled_test_df.groupby(level='Spectrum ID')[
                    'Random Forest Score'].transform(max)]
            subsampled_test_df = subsampled_test_df.groupby(level='Spectrum ID').first()

            binned_df = bin_data_by_score(subsampled_test_df, train_dataset_name, 'Final')
            #Write the new data to a file for all models, including this final combined model.
            #This is not only for the purpose of user inspection, 
            #but also the later production of plots in this procedure 
            #for each model containing data from each of the withheld datasets.
            if os.path.exists(config.globals['Leave-One-Out Data Filepath']):
                binned_df.to_csv(
                    config.globals['Leave-One-Out Data Filepath'], 
                    sep='\t', 
                    header=False, 
                    index=False, 
                    float_format='%.5g', 
                    mode='a')
            else:
                binned_df.to_csv(
                    config.globals['Leave-One-Out Data Filepath'], 
                    sep='\t', 
                    index=False, 
                    float_format='%.5g')
            del(subsampled_test_df, binned_df)
            gc.collect()

        #Plot the predicted versus true correctness of sequence predictions.
        plot_leave_one_out()
        os.remove(config.globals['Leave-One-Out Data Filepath'])

    return

def match_seq_to_ref_fasta(seq, ref_fasta_seqs, print_percent_progress_fn):
    '''
    Parameters
    ----------
    seq : str
    ref_fasta_seqs : list of sequences
    print_percent_progress_fn : function

    Returns
    -------
    int : 0 or 1
    '''

    print_percent_progress_fn()

    for fasta_seq in ref_fasta_seqs:
        if seq in fasta_seq:
            return 1

    return 0

def find_sequence_correctnesses(prediction_df):
    '''
    Determine the correctness of de novo sequence predictions.

    Parameters
    ----------
    prediction_df : DataFrame object

    Returns
    -------
    None
        The mutable object, prediction_df, is updated.
    '''

    #Use matches to FDR-controlled PSMs as well as long reference fasta matches.
    ref_matches = []
    for is_db_search_psm, is_fasta_match in zip(
        prediction_df['Sequence Matches Database Search PSM'].tolist(), 
        ##REMOVE
        ##The specification here of what counts of long reference fasta matches allows 
        ##the threshold length to be changed while tinkering with Postnovo.
        #((prediction_df['Encoded Sequence'].apply(
        #    lambda encoded_seq: (encoded_seq.size >= config.MIN_REF_MATCH_LEN))) & 
        #(prediction_df['Sequence Only Matches Reference Fasta'] == 1)).astype('int').tolist()):
        prediction_df['Exclusive Reference Fasta Match'].tolist()):
        if (is_db_search_psm == 1) or (is_fasta_match == 1):
            ref_matches.append(1)
        else:
            ref_matches.append(0)
    prediction_df['Reference Sequence Match'] = ref_matches

    return

def predict_rf_scores(
    prediction_df, 
    db_search_df=None, 
    leave_one_out_dataset_name=''):
    '''
    Score de novo sequences using Postnovo's random forest models.

    Parameters
    ----------
    prediction_df : DataFrame object
        A table of information on de novo sequence predictions.
    db_search_df : DataFrame object
        A table of information on database search PSMs.
    leave_one_out_dataset_name : str
        Used for determining the model filename in leave-one-out analyses.

    Returns
    -------
    prediction_df : DataFrame object
    '''

    if config.globals['Mode'] == 'test':
        prediction_df = prediction_df.reset_index().set_index(
            config.globals['De Novo Algorithm Origin Headers'])
        if config.globals['Make Test Plots']:
            #Precision-recall and precision-yield plots 
            #compare results from the constituent random forest and final Postnovo models 
            #to individual de novo sequencing algorithms and FDR-controlled PSMs.
            single_alg_classification_stats_dict = OrderedDict()
            for i, alg in enumerate(config.globals['De Novo Algorithms']):
                is_alg_key = [0] * len(config.globals['De Novo Algorithms'])
                is_alg_key[i] = 1
                is_alg_key = tuple(is_alg_key)
                model_score_name = config.alg_evaluation_score_name_dict[alg]
                model_prediction_df = prediction_df.xs(is_alg_key)
                #Select the "default" fragment mass tolerance parameterization.
                model_prediction_df = model_prediction_df[
                    model_prediction_df[config.default_frag_mass_tol_dict[
                        config.globals['Fragment Mass Resolution']]] == 1]
                model_prediction_df = model_prediction_df.reset_index().set_index('Spectrum ID')
                #Select the top-ranked de novo sequences.
                model_prediction_df = model_prediction_df[
                    model_prediction_df['Rank'] == model_prediction_df.groupby(
                        level='Spectrum ID')['Rank'].transform(min)]
                model_prediction_df = model_prediction_df[[
                    'Reference Sequence Match', 
                    'Exclusive Reference Fasta Match', 
                    'Encoded Sequence', 
                    model_score_name]]

                #Get precision, recall, and yield data 
                #for the results of INDIVIDUAL de novo algorithms.
                #Consider sequences of ALL lengths predicted by the algorithms, 
                #which have the potential to be shorter than the sequences used by Postnovo.
                classification_method_full_stats_dict, \
                    reported_precision_classification_method_stats_dict = \
                    get_classification_stats(model_prediction_df, db_search_df, model_score_name)
                single_alg_classification_stats_dict[alg] = classification_method_full_stats_dict
                save_selected_classification_stats(
                    reported_precision_classification_method_stats_dict, 
                    alg + ' (without Postnovo)', 
                    model_score_name)

                #The number of database search PSMs is plotted as a star on precision-yield plots.
                psm_yield = len(db_search_df)

    utils.verbose_print('Predicting random forest scores')
    #Initialize a column for the Postnovo score.
    prediction_df['Random Forest Score'] = None
    for is_alg_key in config.globals['De Novo Algorithm Origin Header Keys']:
        prediction_df = prediction_df.reset_index().set_index(
            config.globals['De Novo Algorithm Origin Headers'])
        #Example 3-algorithm DataFrame index keys (is_alg_key):
        #(1, 0, 0) for Novor top-ranked sequences
        #(0, 1, 1) for PepNovo+-DeepNovo consensus sequences
        #Corresponding keys to random forest models:
        #(1, 0, 0) -> ('Novor', )
        #(0, 1, 1) -> ('PepNovo', 'DeepNovo')
        rf_model_key = tuple(
            [alg for i, alg in enumerate(config.globals['De Novo Algorithms']) if is_alg_key[i]])
        rf_model_name = '-'.join([rf_model_alg for rf_model_alg in rf_model_key])

        try:
            model_prediction_df = prediction_df.xs(is_alg_key).reset_index()
            if 'level_0' in model_prediction_df.columns:
                model_prediction_df.drop('level_0', axis=1, inplace=True)
        except KeyError:
            utils.verbose_print(rf_model_name, 'de novo sequences were not found')
            continue
        
        #REMOVE
        print(rf_model_name)
        print('The number of de novo predictions < 9 amino acids out of ' + str(len(prediction_df)) + ' total : ')
        print(len(model_prediction_df[model_prediction_df['Encoded Sequence'].apply(
            lambda encoded_seq: encoded_seq.size) < 9]))

        #Filter out any short single algorithm sequences from the Postnovo prediction matrix 
        #that were retained for comparison of Postnovo predictions to individual algorithm results.
        if config.globals['Mode'] == 'test':
            #Consensus sequences do not have an entry in the 'Sequence Length' column, 
            #which is presently only a feature for single-algorithm models.
            model_prediction_df = model_prediction_df[
                model_prediction_df['Encoded Sequence'].apply(
                    lambda encoded_seq: encoded_seq.size) >= 
                config.globals['Minimum Postnovo Sequence Length']]

        #Get the features specific to the model.
        prediction_array = model_prediction_df.as_matrix(
            config.globals['Model Features Dict'][rf_model_key])

        #Load the model.
        if leave_one_out_dataset_name == '':
            rf_model = utils.load_pkl_objects(
                config.globals['Postnovo Training Directory'], 
                rf_model_name + '.rf.pkl')
        else:
            rf_model = utils.load_pkl_objects(
                config.globals['Postnovo Training Directory'], 
                rf_model_name + '.leave_one_out.' + leave_one_out_dataset_name + '.rf.pkl')

        rf_model.n_jobs = config.globals['CPU Count']
        #The predict_proba method returns an array of two columns, 
        #the first containing scores for incorrect sequences (0)  
        #and the second containing scores for correct sequences (1), 
        #with each row summing to 1 (scores are correctness probability estimates).
        utils.verbose_print('Predicting', rf_model_name, 'sequence scores')
        model_prediction_df['Random Forest Score'] = rf_model.predict_proba(prediction_array)[:, 1]
        #Update the master prediction table with the scores for this subset.
        #The multiindex should provide a unique identifier for each de novo prediction (row).
        #When consensus sequences are generated, 
        #the type of consensus sequence is needed for unique sequence identification.
        if config.globals['Feature Set ID'] <= 1:
            if (config.globals['Mode'] == 'predict') or (config.globals['Mode'] == 'test'):
                multiindex = config.globals['De Novo Algorithm Origin Headers'] + \
                    ['Spectrum ID'] + config.globals['Fragment Mass Tolerances']
            elif config.globals['Mode'] == 'train':
                multiindex = config.globals['De Novo Algorithm Origin Headers'] + \
                    ['Spectrum ID', 'Dataset'] + config.globals['Fragment Mass Tolerances']
        else:
            if (config.globals['Mode'] == 'predict') or (config.globals['Mode'] == 'test'):
                if len(rf_model_key) == 1:
                    multiindex = config.globals['De Novo Algorithm Origin Headers'] + \
                        ['Spectrum ID'] + config.globals['Fragment Mass Tolerances']
                elif len(rf_model_key) > 1:
                    multiindex = config.globals['De Novo Algorithm Origin Headers'] + \
                        ['Spectrum ID'] + config.globals['Fragment Mass Tolerances'] + \
                        ['Is Consensus Top-Ranked Sequence', 'Is Consensus Longest Sequence']
            elif config.globals['Mode'] == 'train':
                if len(rf_model_key) == 1:
                    multiindex = config.globals['De Novo Algorithm Origin Headers'] + \
                        ['Spectrum ID', 'Dataset'] + config.globals['Fragment Mass Tolerances']
                elif len(rf_model_key) > 1:
                    multiindex = config.globals['De Novo Algorithm Origin Headers'] + \
                        ['Spectrum ID', 'Dataset'] + config.globals['Fragment Mass Tolerances'] + \
                        ['Is Consensus Top-Ranked Sequence', 'Is Consensus Longest Sequence']
        model_prediction_df = model_prediction_df.reset_index().set_index(multiindex)
        if 'index' in model_prediction_df.columns:
            model_prediction_df.drop('index', axis=1, inplace=True)
        if 'level_0' in model_prediction_df.columns:
            model_prediction_df.drop('level_0', axis=1, inplace=True)
        prediction_df = prediction_df.reset_index().set_index(multiindex)
        if 'index' in prediction_df.columns:
            prediction_df.drop('index', axis=1, inplace=True)

        if config.globals['Leave One Out']:
            binned_df = bin_data_by_score(
                model_prediction_df, leave_one_out_dataset_name, rf_model_name)
            #Write the new data to a file for all models, including this constituent model.
            #This is not only for the purpose of user inspection, 
            #but also the later production in this program 
            #of plots for each model containing data from each of the withheld datasets.
            if os.path.exists(config.globals['Leave-One-Out Data Filepath']):
                binned_df.to_csv(
                    config.globals['Leave-One-Out Data Filepath'], 
                    sep='\t', 
                    header=False, 
                    index=False, 
                    float_format='%.5g', 
                    mode='a')
            else:
                binned_df.to_csv(
                    config.globals['Leave-One-Out Data Filepath'], 
                    sep='\t', 
                    index=False, 
                    float_format='%.5g')
        elif config.globals['Make Test Plots']:
            #Make a plot of true versus predicted accuracy for the test dataset and given model.
            #The procedure is a reduced version of what occurs in the function, plot_leave_one_out.
            binned_df = bin_data_by_score(
                model_prediction_df, config.globals['Dataset Name'], rf_model_name)

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            score_bin_midpoints = binned_df['Score Bin Midpoint'].values
            ax1.plot(
                score_bin_midpoints, 
                binned_df['Proportion Correct'].values, 
                linestyle='solid', 
                linewidth=0.5)
            ax2.plot(
                score_bin_midpoints, 
                binned_df['Number of Spectra'].values, 
                linestyle='dashed', 
                linewidth=0.5)

            #Plot one-to-one line, where predicted matches true proportion of correct predictions.
            ax1.plot(np.arange(0.005, 1, 0.01), np.arange(0.005, 1, 0.01), color='black')
            ax1.set_xlabel('Postnovo score (binned every 0.01)')
            ax1.set_ylabel('Proportion of spectra with accurate predictions')
            ax2.set_ylabel('Number of spectra (dashed)')

            fig.savefig(
                os.path.join(
                    config.globals['Output Directory'], 
                    rf_model_name + '.true_vs_predicted_accuracy.pdf'), 
                bbox_inches='tight')

            del(binned_df)
            gc.collect()

        prediction_df.update(model_prediction_df['Random Forest Score'])

        if config.globals['Make Test Plots']:
            utils.verbose_print(
                'Making precision-recall plots for', 
                rf_model_name, 
                'random forest predictions')
            #Plot binary classification statistics for the best spectrum sequence predictions 
            #from the random forest model under consideration (not the overall Postnovo model).
            model_prediction_df = model_prediction_df.reset_index().set_index('Spectrum ID')
            model_prediction_df = model_prediction_df[[
                'Reference Sequence Match', 
                'Exclusive Reference Fasta Match', 
                'Encoded Sequence', 
                'Random Forest Score']]
            model_prediction_df = model_prediction_df[
                model_prediction_df['Random Forest Score'] == 
                model_prediction_df.groupby(level='Spectrum ID')[
                    'Random Forest Score'].transform(max)]
            #Multiple sequences could have the highest score for a spectrum.
            model_prediction_df = model_prediction_df.groupby(level='Spectrum ID').first()

            classification_method_full_stats_dict, \
                reported_precision_classification_method_stats_dict = \
                get_classification_stats(model_prediction_df, db_search_df, 'Random Forest Score')
            save_selected_classification_stats(
                reported_precision_classification_method_stats_dict, 
                rf_model_name, 
                'Random Forest Score')
            #Plot classification statistics generated using Method 1.
            plot_precision_recall(
                classification_method_full_stats_dict, 
                single_alg_classification_stats_dict, 
                '1', 
                rf_model_name)
            plot_precision_yield(
                classification_method_full_stats_dict, 
                single_alg_classification_stats_dict, 
                psm_yield, 
                '1', 
                rf_model_name)
            #Plot classification statistics generated using Method 2.
            plot_precision_recall(
                classification_method_full_stats_dict, 
                single_alg_classification_stats_dict, 
                '2', 
                rf_model_name)
            #Plot classification statistics generated using Method 3.
            plot_precision_recall(
                classification_method_full_stats_dict, 
                single_alg_classification_stats_dict, 
                '3', 
                rf_model_name)
            plot_precision_yield(
                classification_method_full_stats_dict, 
                single_alg_classification_stats_dict, 
                psm_yield, 
                '3', 
                rf_model_name)

    #Filter out any short single algorithm sequences from the Postnovo results table 
    #that were retained for comparison of Postnovo predictions to individual algorithm results.
    if config.globals['Mode'] == 'test':
        prediction_df = prediction_df[
            prediction_df['Encoded Sequence'].apply(lambda encoded_seq: encoded_seq.size) >= 
            config.globals['Minimum Postnovo Sequence Length']]

    return prediction_df

def get_classification_stats(prediction_df, db_search_df, score_name):
    '''
    Parameters
    ----------
    prediction_df : DataFrame object
        Table of de novo sequence prediction information.
    db_search_df : DataFrame object
        Table of FDR-controlled (database search) PSMs.
    score_name : str
        Name of the score (column) used to rank de novo predictions.

    Returns
    -------
    classification_method_full_stats_dict : dict
        Binary classification statistics at every score threshold in the data.
    reported_precision_classification_method_stats_dict : OrderedDict object
        Binary classification statistics at select precision thresholds, meant for user output.
    '''

    #Three "precision-recall" methods are used.
    #Method 1 evaluates the prediction of database search PSMs, 
    #treating partial- and full-length de novo sequences the same.
    #Method 2 also evaluates the prediction of database search PSMs, 
    #but measures the proportion of the PSM peptide mass that is predicted.
    #Method 3 evaluates the prediction of both database search PSMs 
    #and PSMs found by strongly matching de novo sequences to the reference fasta.

    #Merge the de novo prediction and database search PSM tables on spectrum ID.
    prediction_df = prediction_df.reset_index().set_index('Spectrum ID')
    db_search_df = db_search_df.reset_index().set_index('Spectrum ID')
    if 'index' in db_search_df.columns:
        db_search_df.drop('index', axis=1, inplace=True)
    prediction_psm_df = prediction_df.merge(
        db_search_df, how='outer', left_index=True, right_index=True)

    #The score threshold shifts from high to low, 
    #labeling an increasing number of de novo predictions as correct 
    #(all are initially labeled correct; all are finally labeled incorrect).
    score_thresholds = []
    #Method 1
    #Precision = TP Count / (TP Count + FP Count)
    #True positives are spectra with a database search PSM and matching de novo prediction.
    #False positives are spectra with a de novo prediction and without a database search PSM, 
    #or a de novo prediction that does not match the spectrum's database search PSM.
    #"Recall" here is defined as the proportion of spectra with a database search PSM 
    #that have a matching de novo prediction.
    #"Recall" = TP Count / Database Search PSM Total Count
    #Yield is the number of true positives plus false positives.
    precisions_1 = []
    recalls_1 = []
    yields_1 = []
    true_positive_count_1 = 0
    false_positive_count_1 = 0
    yield_1 = 0

    #Method 2
    #"Precision" is weighted by the predicted proportion of corresponding of PSM peptide mass.
    #Since the measured mass of a partial-length de novo sequence cannot be reconstructed here, 
    #the theoretical masses of the de novo and PSM sequences are compared.
    #M = Mass of true positive de novo sequence / Mass of database search PSM
    #"Precision" = (M1 + M2 + ...) / (TP Count + FP Count)
    #"Recall" = (M1 + M2 + ...) / Database Search PSM Total Count
    precisions_2 = []
    recalls_2 = []

    true_positive_count_2 = 0
    standard_plus_mod_mass_dict = config.standard_plus_mod_mass_dict
    code_aa_dict = config.code_aa_dict
    de_novo_seq_masses = []
    for de_novo_seq in prediction_psm_df['Encoded Sequence']:
        if type(de_novo_seq) == np.ndarray:
            mass = 0
            for code in de_novo_seq:
                mass += standard_plus_mod_mass_dict[code_aa_dict[code]]
            de_novo_seq_masses.append(mass)
        else:
            de_novo_seq_masses.append(0)
    prediction_psm_df['De Novo Sequence Mass'] = de_novo_seq_masses
    db_search_psm_seq_masses = []
    for psm in prediction_psm_df['Encoded Reference Sequence']:
        if type(psm) == np.ndarray:
            mass = 0
            for code in psm:
                mass += standard_plus_mod_mass_dict[code_aa_dict[code]]
            db_search_psm_seq_masses.append(mass)
        else:
            db_search_psm_seq_masses.append(0)
    prediction_psm_df['Database Search PSM Mass'] = db_search_psm_seq_masses
    false_positive_count_2 = 0

    #Method 3
    #Precision = TP Count / (TP Count + FP Count)
    #Unlike Methods 1 and 2, true positives now include de novo sequences 
    #strongly matching the reference fasta.
    #"Recall" = TP Count / Database Search PSM Total Count
    #Yield = TP Count + FP Count
    precisions_3 = []
    recalls_3 = []
    yields_3 = []
    true_positive_count_3 = 0
    false_positive_count_3 = 0
    yield_3 = 0

    total_db_search_psms = sum(prediction_psm_df['Database Search PSM Mass'] > 0)

    prev_precision_1 = 1
    prev_precision_2 = 1
    prev_precision_3 = 1
    reported_precision_thresholds = config.reported_precision_thresholds
    reported_precision_classification_method_stats_dict = OrderedDict([
        (reported_precision_threshold, dict()) 
        for reported_precision_threshold in reported_precision_thresholds])
    #Since certain precision thresholds may not be crossed, initialize the recalls as nan.
    for precision_threshold, classification_method_stats_dict in \
        reported_precision_classification_method_stats_dict.items():
        classification_method_stats_dict['Recall 1'] = 0
        classification_method_stats_dict['Yield 1'] = 0
        classification_method_stats_dict['Recall 2'] = 0
        classification_method_stats_dict['Recall 3'] = 0
        classification_method_stats_dict['Yield 3'] = 0
    #Ignore spectra without a de novo prediction in the following loop.
    prediction_psm_df = prediction_psm_df[[
        score_name, 
        'Reference Sequence Match', 
        'Exclusive Reference Fasta Match', 
        'De Novo Sequence Mass', 
        'Database Search PSM Mass']][pd.notnull(prediction_psm_df[score_name])]
    for score, is_ref_match, is_fasta_match, de_novo_mass, psm_mass in sorted(zip(
        prediction_psm_df[score_name].values, 
        prediction_psm_df['Reference Sequence Match'].values, 
        prediction_psm_df['Exclusive Reference Fasta Match'].values, 
        prediction_psm_df['De Novo Sequence Mass'].values, 
        prediction_psm_df['Database Search PSM Mass'].values), key=lambda t: -t[0]):
        if psm_mass > 0:
            if de_novo_mass > 0:
                if is_ref_match:
                    true_positive_count_1 += 1
                    yield_1 += 1
                    m = de_novo_mass / psm_mass
                    true_positive_count_2 += m
                    true_positive_count_3 += 1
                    yield_3 += 1
                else:
                    false_positive_count_1 += 1
                    yield_1 += 1
                    false_positive_count_2 += 1
                    false_positive_count_3 += 1
                    yield_3 += 1
            else:
                pass
        else:
            if de_novo_mass > 0:
                if is_fasta_match:
                    true_positive_count_3 += 1
                    yield_3 += 1
                else:
                    pass
            else:
                raise RuntimeError(
                    'A spectrum without a database search PSM or de novo sequence was found.')

        #Record recalls and yields at predefined precision thresholds.
        #Recording occurs when the precision threshold is crossed from above.
        #This can occur multiple times, as precision can both rise and fall in this procedure, 
        #so update the recorded recall each time the threshold is crossed from above.
        score_thresholds.append(score)
        precisions_1.append(
            true_positive_count_1 / (true_positive_count_1 + false_positive_count_1))
        precisions_2.append(
            true_positive_count_2 / (true_positive_count_2 + false_positive_count_2))
        precisions_3.append(
            true_positive_count_3 / (true_positive_count_3 + false_positive_count_3))
        recalls_1.append(true_positive_count_1 / total_db_search_psms)
        recalls_2.append(true_positive_count_2 / total_db_search_psms)
        recalls_3.append(true_positive_count_3 / total_db_search_psms)
        yields_1.append(yield_1)
        yields_3.append(yield_3)

        current_precision_1 = precisions_1[-1]
        current_precision_2 = precisions_2[-1]
        current_precision_3 = precisions_3[-1]
        threshold_crossed_1 = False
        threshold_crossed_2 = False
        threshold_crossed_3 = False
        for reported_precision_threshold in reported_precision_thresholds:
            if prev_precision_1 > reported_precision_threshold >= current_precision_1:
                reported_precision_classification_method_stats_dict[reported_precision_threshold][
                    'Recall 1'] = recalls_1[-1]
                reported_precision_classification_method_stats_dict[reported_precision_threshold][
                    'Yield 1'] = yields_1[-1]
                threshold_crossed_1 = True
            if prev_precision_2 > reported_precision_threshold >= current_precision_2:
                reported_precision_classification_method_stats_dict[reported_precision_threshold][
                    'Recall 2'] = recalls_2[-1]
                threshold_crossed_2 = True
            if prev_precision_3 > reported_precision_threshold >= current_precision_3:
                reported_precision_classification_method_stats_dict[reported_precision_threshold][
                    'Recall 3'] = recalls_3[-1]
                reported_precision_classification_method_stats_dict[reported_precision_threshold][
                    'Yield 3'] = yields_3[-1]
                threshold_crossed_3 = True

        if threshold_crossed_1 or (current_precision_1 > prev_precision_1):
            prev_precision_1 = current_precision_1
        if threshold_crossed_2 or (current_precision_2 > prev_precision_2):
            prev_precision_2 = current_precision_2
        if threshold_crossed_3 or (current_precision_3 > prev_precision_3):
            prev_precision_3 = current_precision_3

    classification_method_full_stats_dict = dict()
    classification_method_full_stats_dict['Score Threshold'] = score_thresholds
    classification_method_full_stats_dict['Precision 1'] = precisions_1
    classification_method_full_stats_dict['Recall 1'] = recalls_1
    classification_method_full_stats_dict['Yield 1'] = yields_1
    classification_method_full_stats_dict['Precision 2'] = precisions_2
    classification_method_full_stats_dict['Recall 2'] = recalls_2
    classification_method_full_stats_dict['Precision 3'] = precisions_3
    classification_method_full_stats_dict['Recall 3'] = recalls_3
    classification_method_full_stats_dict['Yield 3'] = yields_3

    return classification_method_full_stats_dict, \
        reported_precision_classification_method_stats_dict

def save_selected_classification_stats(
    reported_precision_classification_method_stats_dict, model_name, model_score_name):
    '''
    Parameters
    ----------
    reported_precision_classification_method_stats_dict : OrderedDict object
    model_score_name : str

    Returns
    -------
    None
    '''

    if not os.path.exists(config.globals['Reported Binary Classification Statistics Filepath']):
        with open(config.globals['Reported Binary Classification Statistics Filepath'], 'w') as out_f:
            out_f.write('\t'.join([
                'Model', 
                'Score', 
                'Precision', 
                'Method 1 Recall', 
                'Method 1 Yield', 
                'Method 2 Recall', 
                'Method 3 Recall', 
                'Method 3 Yield', 
                '\n']))

    with open(config.globals['Reported Binary Classification Statistics Filepath'], 'a') as out_f:
        for precision_threshold, classification_method_stats_dict in \
            reported_precision_classification_method_stats_dict.items():
            out_f.write('\t'.join([
                model_name, 
                model_score_name, 
                str(round(precision_threshold, 3)), 
                str(round(classification_method_stats_dict['Recall 1'], 3)), 
                str(round(classification_method_stats_dict['Yield 1'], 3)), 
                str(round(classification_method_stats_dict['Recall 2'], 3)), 
                str(round(classification_method_stats_dict['Recall 3'], 3)), 
                str(round(classification_method_stats_dict['Yield 3'], 3)), 
                '\n']))

    return

def plot_precision_recall(
    classification_method_full_model_stats_dict, 
    single_alg_classification_stats_dict, 
    classification_method_id, 
    model_name, 
    x_label='Recall', 
    y_label='Precision', 
    colorbar_label='Moving threshold:\nRF score or\nde novo algorithm score percentile'):
    '''
    Plot precision-recall curves on the same plot, 
    with one curve being the Postnovo model under consideration, 
    and the others being the results of individual de novo sequencing algorithms.

    Parameters
    ----------
    classification_method_full_model_stats_dict : OrderedDict object
    single_alg_classification_stats_dict : OrderedDict object
    classification_method_id : str
    model_name : str
    x_label : str
    y_label : str
    colorbar_label : str

    Returns
    -------
    None
    '''

    model_precisions = classification_method_full_model_stats_dict[
        'Precision ' + classification_method_id]
    model_recalls = classification_method_full_model_stats_dict[
        'Recall ' + classification_method_id]
    model_score_thresholds = classification_method_full_model_stats_dict['Score Threshold']

    #Plot the precision-recall curve for the Postnovo model under consideration.
    fig, ax = plt.subplots()
    line_collection = colorline(
        model_recalls, model_precisions, np.array(model_score_thresholds) / (1 - 0))
    colorbar = plt.colorbar(line_collection, label=colorbar_label)

    #Plot the precision-recall curves for the individual algs.
    ##COMMENT
    #arrow_position = 8
    for alg, classification_stats_dict in single_alg_classification_stats_dict.items():
        single_alg_precisions = classification_stats_dict['Precision ' + classification_method_id]
        single_alg_recalls = classification_stats_dict['Recall ' + classification_method_id]
        single_alg_score_thresholds = classification_stats_dict['Score Threshold']

        ##COMMENT
        ##Plot arrows labeling the curves by alg.
        #annotation_x = single_alg_recalls[int(len(single_alg_recalls) / arrow_position)]
        #annotation_y = single_alg_precisions[int(len(single_alg_precisions) / arrow_position)]
        #plt.annotate(
        #    alg, 
        #    xy=(annotation_x, annotation_y), 
        #    xycoords='data', 
        #    xytext=(annotation_x - 25, annotation_y - 25), 
        #    textcoords='offset pixels', 
        #    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
        #    horizontalalignment='right', 
        #    verticalalignment='top')

        line_collection = colorline(
            single_alg_recalls, 
            single_alg_precisions, 
            np.array(single_alg_score_thresholds) / \
                (config.upper_score_bound_dict[alg] - config.lower_score_bound_dict[alg]))
        tick_locs = config.colorbar_tick_dict[alg]

    plt.plot(1, config.DEFAULT_PRECISION, color='r', marker='*', markersize=10)

    #Precision and recall range from 0 to 1.
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout(True)
    out_fp = os.path.join(
        config.globals['Output Directory'], 
        model_name + '.precision_recall.method_' + classification_method_id + '.pdf')
    plt.savefig(out_fp, bbox_inches='tight')
    plt.close()

    return

def plot_precision_yield(
    classification_method_full_model_stats_dict, 
    single_alg_classification_stats_dict, 
    psm_yield, 
    classification_method_id, 
    model_name, 
    x_label='Sequence Yield', 
    y_label='Sequence Precision', 
    colorbar_label='Moving threshold:\nRF score or\nde novo algorithm score percentile'):
    '''
    Plot precision-yield curves on the same plot, 
    with one curve being the Postnovo model under consideration, 
    and the others being the results of individual de novo sequencing algorithms.

    Parameters
    ----------
    classification_method_full_model_stats_dict : OrderedDict object
    single_alg_classification_stats_dict : OrderedDict object
    psm_yield : int
    classification_method_id : str
    model_name : str
    x_label : str
    y_label : str
    colorbar_label : str

    Returns
    -------
    None
    '''

    model_precisions = classification_method_full_model_stats_dict[
        'Precision ' + classification_method_id]
    model_yields = classification_method_full_model_stats_dict[
        'Yield ' + classification_method_id]
    model_score_thresholds = classification_method_full_model_stats_dict['Score Threshold']

    #Plot the precision-yield curve for the Postnovo model under consideration.
    fig, ax = plt.subplots()
    line_collection = colorline(
        model_yields, model_precisions, np.array(model_score_thresholds) / (1 - 0))
    colorbar = plt.colorbar(line_collection, label=colorbar_label)

    max_encountered_yield = 0
    #Adjust the x-axis maximum to the highest yield.
    #This yield can theoretically be higher than the yield of database search PSMs, 
    #if more de novo sequences are found than PSMs.
    if model_yields[-1] > max_encountered_yield:
        max_encountered_yield = model_yields[-1]

    #Plot the precision-yield curves for the individual algorithms.
    ##COMMENT
    #arrow_position = 8
    for alg, classification_stats_dict in single_alg_classification_stats_dict.items():
        single_alg_precisions = classification_stats_dict['Precision ' + classification_method_id]
        single_alg_yields = classification_stats_dict['Yield ' + classification_method_id]
        single_alg_score_thresholds = classification_stats_dict['Score Threshold']

        ##COMMENT
        ##Plot arrows labeling the curves by alg.
        #annotation_x = single_alg_yields[int(len(single_alg_yields) / arrow_position)]
        #annotation_y = single_alg_precisions[int(len(single_alg_precisions) / arrow_position)]
        #plt.annotate(
        #    alg, 
        #    xy=(annotation_x, annotation_y), 
        #    xycoords='data', 
        #    xytext=(annotation_x - 25, annotation_y - 25), 
        #    textcoords='offset pixels', 
        #    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
        #    horizontalalignment='right', 
        #    verticalalignment='top')

        line_collection = colorline(
            single_alg_yields, 
            single_alg_precisions, 
            np.array(single_alg_score_thresholds) / \
                (config.upper_score_bound_dict[alg] - config.lower_score_bound_dict[alg]))
        tick_locs = config.colorbar_tick_dict[alg]

        if single_alg_yields[-1] > max_encountered_yield:
            max_encountered_yield = single_alg_yields[-1]

    plt.plot(psm_yield, config.DEFAULT_PRECISION, color='r', marker='*', markersize=10)

    if psm_yield > max_encountered_yield:
        #Guarantee some room along the x-axis for the star.
        max_encountered_yield = psm_yield + 1000
    plt.xlim([1, max_encountered_yield])
    plt.ylim([0, 1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout(True)
    out_fp = os.path.join(
        config.globals['Output Directory'], 
        model_name + '.precision_yield.method_' + classification_method_id + '.pdf')
    plt.savefig(out_fp, bbox_inches='tight')
    plt.close()

    return

def colorline(x, y, z, cmap='jet', linewidth=3, alpha=1.0):
    '''
    Parameters
    ----------
    x : list
    y : list
    z : list
    cmap : matplotlib.cm.ScalarMappable object
    linewidth : float
    alpha : float

    Returns
    -------
    line_collection : matplotlib.collections.LineCollection object
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_collection = mcoll.LineCollection(
        segments, array=z, norm=plt.Normalize(0, 1), cmap=cmap, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(line_collection)

    return line_collection

def initialize_score_sacrifice_extension_workers(
    _print_percent_progress_fn, 
    _max_total_sacrifice, 
    _sacrifice_extension_ratio):
    '''
    Set global variables used in the function, do_score_sacrifice_extension.

    Parameters
    ----------
    _print_percent_progress_fn : function
    _max_total_sacrifice : float
    _sacrifice_extension_ratio : float

    Returns
    -------
    None
    '''

    global print_percent_progress_fn, max_total_sacrifice, sacrifice_extension_ratio

    print_percent_progress_fn = _print_percent_progress_fn
    max_total_sacrifice = _max_total_sacrifice
    sacrifice_extension_ratio = _sacrifice_extension_ratio

    return

def do_score_sacrifice_extension(spectrum_df):
    '''
    Select longer sequence predictions at the expense of accuracy.

    Parameters
    ----------
    spectrum_df : DataFrame object

    Returns
    -------
    selected_seq_df : DataFrame object
    '''

    #This function has a global scope to permit tracking progress across multiple processes.
    print_percent_progress_fn()

    min_prob = config.globals['Minimum Postnovo Sequence Probability']
    spectrum_df.sort_values('Random Forest Score', ascending=False, inplace=True)
    probs = spectrum_df['Random Forest Score'].tolist()
    highest_prob = probs[0]
    lower_prob_encoded_seqs = spectrum_df['Encoded Sequence'].tolist()[1:]
    current_longest_encoded_seq = spectrum_df.iloc[0]['Encoded Sequence']
    current_prob = highest_prob
    longest_row_index = 0
    for i, encoded_seq in enumerate(lower_prob_encoded_seqs):
        if encoded_seq.size > current_longest_encoded_seq.size:
            lower_prob = probs[i + 1]
            if lower_prob < min_prob:
                break
            #The longer sequence must contain the shorter, higher-probability sequence.
            if current_longest_encoded_seq in encoded_seq:
                #The longer sequence must be within max_total_sacrifice of the highest probability.
                if highest_prob - lower_prob <= max_total_sacrifice:
                    #The sacrifice per percent extension must meet the criterion.
                    length_weighted_max_sacrifice = sacrifice_extension_ratio * \
                        np.sum(np.reciprocal(np.arange(
                            len(current_longest_encoded_seq) + 1, 
                            len(encoded_seq) + 1, 
                            dtype=np.float16)))
                    if current_prob - lower_prob <= length_weighted_max_sacrifice:
                        longest_row_index = i + 1
                        current_longest_encoded_seq = encoded_seq
                        current_prob = lower_prob
    selected_seq_df = spectrum_df.iloc[[longest_row_index]]

    return selected_seq_df

def make_rf_models(train_df, plot_feature_importance=False, leave_one_out_dataset_name=''):
    '''
    Make, describe, and store Postnovo random forest models.

    Parameters
    ----------
    train_df : DataFrame object
    plot_feature_importance : bool
    leave_one_out_dataset_name : str
        Used for determining the model filename in leave-one-out analyses.
    
    Returns
    -------
    None
    '''

    if 'level_0' in train_df.columns:
        train_df.drop('level_0', axis=1, inplace=True)
    train_df.reset_index(inplace=True)
    train_df.set_index(config.globals['De Novo Algorithm Origin Headers'], inplace=True)
    #Map each random forest's array of training features to an array of validation targets 
    #(de novo sequence features map to de novo sequence accuracy).
    for is_alg_key in config.globals['De Novo Algorithm Origin Header Keys']:
        #Example 3-algorithm DataFrame index keys (is_alg_key):
        #(1, 0, 0) for Novor top-ranked sequences
        #(0, 1, 1) for PepNovo+-DeepNovo consensus sequences
        #Corresponding keys to random forest models:
        #(1, 0, 0) -> ('Novor', )
        #(0, 1, 1) -> ('PepNovo', 'DeepNovo')
        rf_model_key = tuple(
            [alg for i, alg in enumerate(config.globals['De Novo Algorithms']) if is_alg_key[i]])
        rf_model_name = '-'.join([rf_model_alg for rf_model_alg in rf_model_key])

        try:
            model_train_df = train_df.xs(is_alg_key)
        except KeyError:
            utils.verbose_print(
                'De novo sequences for the', rf_model_name, 
                'Postnovo random forest model were not found')
            continue

        rf_train_array = model_train_df.as_matrix(
            config.globals['Model Features Dict'][rf_model_key])
        rf_targets = model_train_df['Reference Sequence Match'].tolist()
        del(model_train_df)
        gc.collect()

        #Train the model.
        utils.verbose_print(
            'Training the Postnovo random forest model for', rf_model_name, 'sequences')
        rf_model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS, 
            max_depth=config.RF_MAX_DEPTH, 
            max_features=config.RF_MAX_FEATURES, 
            oob_score=True, 
            n_jobs=config.globals['CPU Count'])
        rf_model.fit(rf_train_array, rf_targets)
        del(rf_train_array, rf_targets)
        gc.collect()

        #Plot feature importances for the model, if the user opted to do so.
        if plot_feature_importance:
            utils.verbose_print(
                'Plotting feature importances for', rf_model_name, 'sequences')
            rf_features = config.globals['Model Features Dict'][rf_model_key]
            #First, plot one bar per feature.
            feature_importances = rf_model.feature_importances_
            #Error bars are 1 sigma, from feature importances in each random forest tree.
            feature_stdevs = np.std(
                [tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
            #Determine feature ranks.
            feature_sort_indices = np.argsort(feature_importances)[::-1]
            all_x_labels = [rf_features[i] for i in feature_sort_indices]
            all_y = [feature_importances[i] for i in feature_sort_indices]
            all_errors = [feature_stdevs[i] for i in feature_sort_indices]

            rf_feature_groups = []
            plot_handles = []
            #Give different colors to different feature groups.
            #This means that the bars from each feature group must be plotted separately.
            plt.figure(figsize=(6, 8))
            for feature_group, feature_group_features in config.feature_group_dict.items():
                if len(set(rf_features).intersection(feature_group_features)) > 0:
                    rf_feature_groups.append(feature_group)
                    feature_group_x = []
                    feature_group_y = []
                    feature_group_errors = []
                    for feature in feature_group_features:
                        #Determine the rank of the bar corresponding to the feature.
                        #For certain feature groups (e.g., 'Fraction Source Length'), 
                        #not every feature is necessarily present in the model.
                        try:
                            feature_index = all_x_labels.index(feature)
                            feature_group_x.append(feature_index)
                            feature_group_y.append(all_y[feature_index])
                            feature_group_errors.append(all_errors[feature_index])
                        except ValueError:
                            pass
                    #Add the bars for the feature group to the plot.
                    plot_handles.append(plt.bar(
                        feature_group_x, 
                        feature_group_y, 
                        yerr=feature_group_errors, 
                        color=config.feature_group_color_dict[feature_group]))

            #Determine the order in which feature groups appear in the plot.
            legend_labels = []
            for rf_feature in all_x_labels:
                for rf_feature_group in rf_feature_groups:
                    if (rf_feature in config.feature_group_dict[rf_feature_group]) and \
                        (rf_feature_group not in legend_labels):
                        legend_labels.append(rf_feature_group)

            plt.title('Postnovo Classification Model of ' + rf_model_name + ' Sequences')
            plt.xticks(
                range(len(all_x_labels)), 
                all_x_labels, 
                rotation=-45, 
                ha='left', 
                fontsize=8 - len(all_x_labels) // 15)
            plt.xlabel('Feature')
            plt.ylim(bottom=0)
            plt.ylabel('Importance')
            plt.legend(plot_handles, legend_labels, framealpha=0.5, fontsize=9)
            plt.tight_layout()
            #Save the plot to a PDF file.
            plt.savefig(
                os.path.join(
                    config.globals['Output Directory'], 
                    rf_model_name + '.feature_importances.pdf'), 
                bbox_inches='tight')
            plt.close()

            #Second, plot one bar per feature group.
            rf_feature_group_importances = []
            rf_feature_group_stdevs = []
            plt.figure(figsize=(6, 8))
            for rf_feature_group in rf_feature_groups:
                rf_feature_indices = []
                rf_feature_group_importance = 0
                for feature in config.feature_group_dict[rf_feature_group]:
                    if feature in rf_features:
                        rf_feature_index = rf_features.index(feature)
                        rf_feature_indices.append(rf_feature_index)
                        #Sum random forest feature importances across feature group.
                        rf_feature_group_importance += feature_importances[rf_feature_index]
                rf_feature_group_importances.append(rf_feature_group_importance)

                #Calculate the 1 sigma error in feature group importance.
                tree_feature_group_importances = []
                for tree in rf_model.estimators_:
                    #Sum tree feature importances across feature group.
                    tree_feature_group_importances.append(
                        sum([tree.feature_importances_[i] for i in rf_feature_indices]))
                rf_feature_group_stdevs.append(np.std(tree_feature_group_importances))

            #Determine feature group ranks and make plot.
            sorted_x_labels = []
            plot_handles = []
            for plot_index, rf_feature_group_index in enumerate(
                np.argsort(rf_feature_group_importances)[::-1]):
                rf_feature_group = rf_feature_groups[rf_feature_group_index]
                feature_group_y = [rf_feature_group_importances[rf_feature_group_index]]
                feature_group_error = [rf_feature_group_stdevs[rf_feature_group_index]]
                sorted_x_labels.append(rf_feature_group)
                #Add the bar for the feature group to the plot.
                plot_handles.append(plt.bar(
                    [plot_index], 
                    feature_group_y, 
                    yerr=feature_group_error, 
                    color=config.feature_group_color_dict[rf_feature_group]))

            plt.title('Postnovo Classification Model of ' + rf_model_name + ' Sequences')
            plt.xticks(
                range(len(sorted_x_labels)), 
                sorted_x_labels, 
                rotation=-45, 
                ha='left', 
                fontsize=9 - len(sorted_x_labels) // 15)
            plt.xlabel('Feature')
            plt.ylim(bottom=0)
            plt.ylabel('Importance')
            plt.legend(
                plot_handles, sorted_x_labels, loc='upper right', framealpha=0.5, fontsize=9)
            plt.tight_layout()
            #Save the plot to a PDF file.
            plt.savefig(
                os.path.join(
                    config.globals['Output Directory'], 
                    rf_model_name + '.feature_group_importances.pdf'), 
                bbox_inches='tight')
            plt.close()

        #Save each random forest model object.
        if leave_one_out_dataset_name == '':
            utils.save_pkl_objects(
                config.globals['Postnovo Training Directory'], 
                **{rf_model_name + '.rf.pkl': rf_model})
        else:
            utils.save_pkl_objects(
                config.globals['Postnovo Training Directory'], 
                **{
                    rf_model_name + '.leave_one_out.' + leave_one_out_dataset_name + '.rf.pkl': 
                    rf_model})

    train_df.reset_index(inplace=True)

    return

def bin_data_by_score(subsampled_test_df, dataset_name, model_type):
    '''
    Bin random forest scores and save tabulated binned scores and mean correctnesses.

    Parameters
    ----------
    subsampled_test_df : DataFrame object
        A table of de novo sequence predictions from a model 
        (subsampled from the full table for all models) 
        with a column of scores and a column of match values to FDR-controlled reference sequences.
    dataset_name : str
        The ID of the dataset under consideration.
    model_type : str
        The Postnovo model (e.g., constituent models such as "Novor-DeepNovo" -- or "Final")

    Returns
    -------
    None
    '''

    #Bin the sequences by score 
    #to compare the score to the predicted proportion of correct sequences in the bin.
    #Sort the sequences in order of score.
    rf_scores, ref_matches = zip(*sorted(zip(
        subsampled_test_df['Random Forest Score'].tolist(), 
        subsampled_test_df['Reference Sequence Match'].tolist())))
    score_bin_midpoints = []
    binned_ref_matches = []
    correct_proportions = []
    spectrum_counts = []
    current_score_bin_upperbound = config.SCORE_BIN_SIZE
    for rf_score, ref_match in zip(rf_scores, ref_matches):
        #Each score bin is inclusive of the upper bound.
        #Advance to the next bin when this upper bound is exceeded.
        binned_ref_matches.append(ref_match)
        if rf_score > current_score_bin_upperbound:
            score_bin_midpoints.append(
                current_score_bin_upperbound - config.SCORE_BIN_SIZE / 2)
            correct_proportions.append(np.mean(binned_ref_matches))
            spectrum_counts.append(len(binned_ref_matches))
            current_score_bin_upperbound += config.SCORE_BIN_SIZE
            binned_ref_matches = []

    binned_df = pd.DataFrame.from_dict({
        'Score Bin Midpoint': score_bin_midpoints, 
        'Proportion Correct': correct_proportions, 
        'Number of Spectra': spectrum_counts})
    binned_df['Dataset'] = dataset_name
    binned_df['Postnovo Model'] = model_type
    binned_df['Timestamp'] = str(datetime.datetime.now()).split('.')[0]
    #Reorder the columns.
    binned_df = binned_df[[
        'Dataset', 
        'Postnovo Model', 
        'Timestamp', 
        'Score Bin Midpoint', 
        'Proportion Correct', 
        'Number of Spectra']]

    return binned_df

def plot_leave_one_out():
    '''
    Conduct a leave-one-out analysis of model accuracy with each training dataset.

    Parameters
    ----------
    None
        Global variables are used.

    Output
    ------
    None
    '''

    leave_one_out_df = pd.read_csv(
        config.globals['Leave-One-Out Data Filepath'], sep='\t', header=0)

    plot_dict = dict()
    #Each model (individual random forests and the combined, final model) gets its own plot.
    model_names = list(set(leave_one_out_df['Postnovo Model']))
    dataset_names = list(set(leave_one_out_df['Dataset']))
    for model_name in model_names:
        plot_dict[model_name] = model_dict = dict()
        model_dict['Figure'], model_dict['Axis 1'] = plt.subplots()
        model_dict['Axis 2'] = model_dict['Axis 1'].twinx()

    #Plot results for each dataset
    for model_name, model_df in leave_one_out_df.groupby('Postnovo Model', sort=False):
        model_dict = plot_dict[model_name]
        for dataset_name, dataset_df in model_df.groupby('Dataset', sort=False):
            score_bin_midpoints = dataset_df['Score Bin Midpoint'].values
            model_dict['Axis 1'].plot(
                score_bin_midpoints, 
                dataset_df['Proportion Correct'].values, 
                linestyle='solid', 
                linewidth=0.5)
            model_dict['Axis 2'].plot(
                score_bin_midpoints, 
                dataset_df['Number of Spectra'].values, 
                linestyle='dashed', 
                linewidth=0.5)

    for model_name in model_names:
        model_dict = plot_dict[model_name]
        #Plot one-to-one lines, where predicted matches true proportions of correct predictions.
        model_dict['Axis 1'].plot(
            np.arange(0.005, 1, 0.01), np.arange(0.005, 1, 0.01), color='black')
        model_dict['Axis 1'].set_xlabel('Postnovo score (binned every 0.01)')
        model_dict['Axis 1'].set_ylabel('Proportion of spectra with accurate predictions')
        model_dict['Axis 2'].set_ylabel('Number of spectra (dashed)')
        #The legend is placed to the right outside of the plot area.
        model_dict['Axis 1'].legend(dataset_names, loc='center left', bbox_to_anchor=(1.16, 0.5))

        model_dict['Figure'].savefig(
            os.path.join(
                config.globals['Output Directory'], 
                model_name + '.true_vs_predicted_accuracy.pdf'), 
            bbox_inches='tight')

    return

##OLD CODE SNIPPETS FOR RANDOM FOREST HYPERPARAMETERIZATION OPTIMATION
##Plot out-of-bag error reduction with the incremental addition of trees to the rf.
#data_train_split, \
#    data_validation_split, \
#    target_train_split, \
#    target_validation_split = \
#    train_test_split(
#        train_target_arr_dict[alg_key]['train'], 
#        train_target_arr_dict[alg_key]['target'], 
#        stratify=train_target_arr_dict[alg_key]['target'])
#plot_errors(
#    data_train_split, 
#    data_validation_split, 
#    target_train_split, 
#    target_validation_split, 
#    alg_key)

##
#elif config.globals['mode'] == 'optimize':
#    utils.verbose_print('Optimizing random forest parameters')
#    optimized_params = optimize_model(train_target_arr_dict)
#    forest_dict = make_forest_dict(train_target_arr_dict, optimized_params)

##
#def optimize_model(train_target_arr_dict):

#    optimized_params = OrderedDict()
#    for alg_key in train_target_arr_dict:
#        optimized_params[alg_key] = OrderedDict()

#        data_train_split, data_validation_split, target_train_split, target_validation_split = \
#            train_test_split(
#                train_target_arr_dict[alg_key]['train'], 
#                train_target_arr_dict[alg_key]['target'], 
#                stratify = train_target_arr_dict[alg_key]['target'])
#        forest_grid = GridSearchCV(
#            RandomForestClassifier(n_estimators = config.RF_N_ESTIMATORS, oob_score = True), 
#            {'max_features': ['sqrt', None], 'max_depth': [depth for depth in range(11, 20)]}, 
#            n_jobs = config.globals['CPU Count'])
#        forest_grid.fit(data_train_split, target_train_split)
#        optimized_forest = forest_grid.best_estimator_
#        optimized_params[alg_key]['max_depth'] = optimized_forest.max_depth
#        utils.verbose_print(alg_key, 'optimized max depth:', optimized_forest.max_depth)
#        optimized_params[alg_key]['max_features'] = optimized_forest.max_features
#        utils.verbose_print(alg_key, 'optimized max features:', optimized_forest.max_features)

#        plot_feature_importances(
#            optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
#        plot_binned_feature_importances(
#            optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
#        plot_errors(
#            data_train_split, 
#            data_validation_split, 
#            target_train_split, 
#            target_validation_split, 
#            alg_key)

#    return optimized_params

##
#def plot_errors(
#    data_train_split, 
#    data_validation_split, 
#    target_train_split, 
#    target_validation_split, 
#    alg_key):
#    if len(alg_key) > 1:
#        utils.verbose_print(
#            'Plotting errors vs tree size for', '-'.join(alg_key), 'consensus sequences')
#    else:
#        utils.verbose_print('Plotting errors vs. tree size for', alg_key[0], 'sequences.')

#    ensemble_clfs = [(
#        #'max_features='sqrt', 
#        'max_features=None', 
#        RandomForestClassifier(
#            warm_start=True, 
#            max_features=None, 
#            oob_score=True, 
#            max_depth=15, 
#            n_jobs=config.globals['CPU Count'], 
#            random_state=1))]

#    oob_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
#    min_estimators = 10
#    max_estimators = 500

#    for label, clf in ensemble_clfs:
#        for tree_number in range(min_estimators, max_estimators + 1, 100):
#            clf.set_params(n_estimators=tree_number)
#            clf.fit(data_train_split, target_train_split)

#            oob_error = 1 - clf.oob_score_
#            oob_errors[label].append((tree_number, oob_error))

#    fig, ax1 = plt.subplots()
#    for label, oob_error in oob_errors.items():
#        xs, ys = zip(*oob_error)
#        ax1.plot(xs, ys, label='oob error: ' + label)

#    ax1.set_xlim(min_estimators, max_estimators)
#    ax1.set_xlabel('n_estimators')
#    ax1.set_ylabel('error rate')
#    ax1.legend(loc='upper right')
#    fig.set_tight_layout(True)

#    alg_key_str = '_'.join(alg_key)
#    save_path = join(config.globals['iodir'], alg_key_str + '_error.pdf')
#    fig.savefig(save_path, bbox_inches='tight')

#    return