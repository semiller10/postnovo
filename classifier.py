''' Sequence accuracy classification model '''

import csv
import datetime
import math
import matplotlib
# Set backend to make image files on server
matplotlib.use('Agg')
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os.path
import numpy as np
import pandas as pd
import pickle as pkl
import sklearn as sk
import sys
import warnings
warnings.filterwarnings('ignore')

from collections import Counter, OrderedDict
from functools import partial
from multiprocessing import Pool
from os.path import join, basename
from scipy.stats import norm
from sklearn.cluster import Birch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.dbsearch as dbsearch
    import postnovo.input as input
    import postnovo.utils as utils
else:
    import config
    import dbsearch
    import input as input
    import utils

ref_aa_sum = 0

def classify(prediction_df=None, input_df_dict=None):
    utils.verbose_print()
    if config.globals['mode'] in ['train', 'test', 'optimize']:
        #prediction_df, ref_correspondence_df, db_search_ref = find_target_accuracy(prediction_df)
        #with open(os.path.join(config.globals['iodir'], 'ref_correspondence_df.pkl'), 'wb') as f:
        #    pkl.dump(ref_correspondence_df, f, 2)
        #with open(os.path.join(config.globals['iodir'], 'db_search_ref.pkl'), 'wb') as f:
        #    pkl.dump(db_search_ref, f, 2)
        with open(os.path.join(config.globals['iodir'], 'ref_correspondence_df.pkl'), 'rb') as f:
            ref_correspondence_df = pkl.load(f)
        with open(os.path.join(config.globals['iodir'], 'db_search_ref.pkl'), 'rb') as f:
            db_search_ref = pkl.load(f)

        #Determine the total number of amino acids in the db search reference peptides.
        global ref_aa_sum
        for ref_seq in db_search_ref['ref seq'].tolist():
            ref_aa_sum += len(ref_seq)

    #utils.verbose_print('Formatting data for compatability with model')
    #prediction_df = standardize_prediction_df_cols(prediction_df)
    #utils.save_pkl_objects(config.globals['iodir'], **{'prediction_df': prediction_df})
    prediction_df = utils.load_pkl_objects(config.globals['iodir'], 'prediction_df')

    if config.globals['mode'] == 'predict':
        reported_prediction_df = make_predictions(prediction_df)
        reported_prediction_df.to_csv(
            os.path.join(config.globals['iodir'], 'best_predictions.csv')
        )
        #reported_prediction_df = pd.read_csv(
        #    os.path.join(config.globals['iodir'], 'best_predictions.csv'), header=0
        #)

        reported_prediction_df = reported_prediction_df[
            reported_prediction_df['probability'] >= config.globals['min_prob']
        ]

        df = reported_prediction_df.reset_index()
        if config.globals['db_search_fp']:
            df = dbsearch.merge_predictions(df)
        mass_grouped_df = dbsearch.group_predictions(df)
        retained_seq_dict = dbsearch.lengthen_seqs(mass_grouped_df)
        dbsearch.make_fasta(retained_seq_dict)

    elif config.globals['mode'] == 'test':

        reported_prediction_df = make_predictions(prediction_df, input_df_dict, db_search_ref)
        reported_prediction_df = reported_prediction_df.reset_index().merge(
            ref_correspondence_df.reset_index(), 
            how='left', 
            on=(
                config.globals['is_alg_names'] 
                + ['spec_id'] 
                + config.globals['frag_mass_tols'] 
                + ['is longest consensus', 'is top rank consensus']
            )
        )
        reported_prediction_df.set_index('spec_id', inplace=True)
        reported_cols_in_order = []
        for reported_df_col in config.reported_df_cols:
            if reported_df_col in reported_prediction_df.columns:
                reported_cols_in_order.append(reported_df_col)
        reported_prediction_df = reported_prediction_df.reindex_axis(
            reported_cols_in_order, axis=1
        )
        reported_prediction_df.to_csv(
            os.path.join(config.globals['iodir'], 'best_predictions.csv')
        )

    elif config.globals['mode'] in ['train', 'optimize']:
        
        utils.verbose_print('Updating training database')
        training_df = update_training_data(prediction_df)
        #training_df = pd.read_csv(
        #    os.path.join(config.data_dir, 'training_df.csv'),
        #    header=0,
        #    index_col=config.globals['is_alg_names']
        #)
        #REMOVE
        #training_df = utils.load_pkl_objects(config.data_dir, 'training_df')

        forest_dict = make_training_forests(training_df)
        utils.save_pkl_objects(config.data_dir, **{'forest_dict': forest_dict})
        #forest_dict = utils.load_pkl_objects(config.data_dir, 'forest_dict')

def find_seqs_in_paired_seqs(seq_list1, seq_list2):
    seq_pairs = list(zip(seq_list1, seq_list2))
    matches = []
    for seq_pair in seq_pairs:
        if seq_pair[0] in seq_pair[1]:
            matches.append(1)
        else:
            matches.append(0)
    return matches

def find_target_accuracy(prediction_df):

    utils.verbose_print('Loading', basename(config.globals['db_search_fp']))
    db_search_ref = load_db_search_ref_file(config.globals['db_search_fp'])
    utils.verbose_print('Loading', basename(config.globals['ref_fasta_fp']))
    fasta_ref = load_fasta_ref_file(config.globals['ref_fasta_fp'])

    utils.verbose_print('Finding sequence matches to database search reference')

    prediction_df.reset_index(inplace=True)
    comparison_df = prediction_df.merge(db_search_ref, how='left', on='spec_id')

    #Null entries exist for spectra with a de novo but not a db search sequence.
    prediction_df['spec has db search PSM'] = comparison_df['ref seq'].notnull().astype(int)
    comparison_df['ref seq'][comparison_df['ref seq'].isnull()] = ''
    #Add a column recording whether de novo seq is in db search reference (1 if true, 0 if false).
    prediction_df['de novo seq matches db search seq'] = find_seqs_in_paired_seqs(
        comparison_df['seq'].tolist(), 
        comparison_df['ref seq'].tolist()
    )
    #Add column for the length of the de novo seq.
    prediction_df['predict len'] = comparison_df['seq'].apply(len)

    utils.verbose_print(
        'Finding de novo sequence matches to fasta reference '
        'for spectra lacking database search PSMs'
    )

    #Determine the number of de novo seqs lacking ref matches that meet the length criterion.
    no_db_search_psm_df = prediction_df[prediction_df['spec has db search PSM'] == 0]
    no_db_search_psm_df = no_db_search_psm_df[
        no_db_search_psm_df['seq'].apply(len) >= config.min_ref_match_len
    ]
    unique_long_denovo_seqs = list(set(no_db_search_psm_df['seq']))

    #What is the purpose of this communication?
    #utils.verbose_print(
    #    'Finding minimum de novo sequence length to uniquely match fasta reference'
    #)
    #config.min_ref_match_len = find_min_seq_len(fasta_ref=fasta_ref, cores=config.globals['cpus'])
    one_percent_number_denovo_seqs = len(unique_long_denovo_seqs) / 100 / config.globals['cpus']

    #Match de novo to the reference fasta file.
    #Multiprocess
    multiprocessing_pool = Pool(config.globals['cpus'])
    print_percent_progress_fn = partial(
        utils.print_percent_progress_multithreaded, 
        procedure_str='Reference sequence matching progress: ', 
        one_percent_total_count=one_percent_number_denovo_seqs, 
        cores=config.globals['cpus']
    )
    single_var_match_seq_to_fasta_ref = partial(
        match_seq_to_fasta_ref, 
        fasta_ref=fasta_ref, 
        print_percent_progress_fn=print_percent_progress_fn
    )
    #Return a 1 if there is a match, 0 if there isn't.
    fasta_matches = multiprocessing_pool.map(
        single_var_match_seq_to_fasta_ref, unique_long_denovo_seqs
    )
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    fasta_match_dict = dict(zip(unique_long_denovo_seqs, fasta_matches))
    single_var_get_match_from_dict = partial(get_match_from_dict, match_dict=fasta_match_dict)
    no_db_search_psm_df['correct de novo seq not found in db search'] = \
        no_db_search_psm_df['seq'].apply(single_var_get_match_from_dict)
    prediction_df = prediction_df.merge(
        no_db_search_psm_df['correct de novo seq not found in db search'].to_frame(), 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    prediction_df['correct de novo seq not found in db search'].fillna(0, inplace = True)

    #Count as correct all de novo seqs that matched either the db search results or the ref fasta.
    prediction_df['ref match'] = prediction_df['de novo seq matches db search seq'] + \
        prediction_df['correct de novo seq not found in db search']
    prediction_df.set_index(
        config.globals['is_alg_names'] + 
        ['spec_id'] + 
        config.globals['frag_mass_tols'] + 
        ['is longest consensus', 'is top rank consensus'], 
        inplace=True
    )
    #Store information on de novo seq accuracy in a separate structure.
    ref_correspondence_df = pd.concat(
        [
            prediction_df['spec has db search PSM'], 
            prediction_df['de novo seq matches db search seq'], 
            prediction_df['correct de novo seq not found in db search']
        ], 
        axis=1
    )
    prediction_df.drop(
        [
            'spec has db search PSM', 
            'de novo seq matches db search seq', 
            'correct de novo seq not found in db search'
        ], 
        axis=1, 
        inplace=True
    )
    prediction_df = prediction_df.reset_index().set_index(
        config.globals['is_alg_names'] + ['spec_id']
    )

    return prediction_df, ref_correspondence_df, db_search_ref

def get_match_from_dict(seq, match_dict):
    return match_dict[seq]

def load_db_search_ref_file(db_search_fp):

    db_search_df = pd.read_csv(db_search_fp, sep='\t', header=0)
    
    db_search_df['is_decoy'] = db_search_df['Protein'].apply(
        lambda s: 1 if 'XXX_' in s else 0
    )
    db_search_df.sort_values('SpecEValue', inplace=True)
    decoy_count = 0
    target_count = 0
    target_count_denom = target_count
    headers = db_search_df.columns.tolist()
    spec_evalue_col = headers.index('SpecEValue')
    qvalues = []
    for row in db_search_df.itertuples(index=False):
        #If the spectrum matched a decoy
        if row[-1]:
            decoy_count += 1
            target_count_denom = target_count
        else:
            target_count += 1

        if decoy_count == 0:
            qvalues.append(0)
        else:
            try:
                qvalues.append(decoy_count / target_count_denom)
            except ZeroDivisionError:
                qvalues.append(1)
    db_search_df['psm_qvalue'] = qvalues
    db_search_df.sort_values(['SpecID', 'psm_qvalue'], inplace=True)
    db_search_df.drop_duplicates(subset='SpecID', inplace=True)
    db_search_df = db_search_df[db_search_df['psm_qvalue'] <= config.max_fdr]
    db_search_df.drop('psm_qvalue', axis=1, inplace=True)

    out_fp = os.path.splitext(db_search_fp)[0] + '.' + str(config.max_fdr) + '.tsv'
    db_search_df.to_csv(out_fp, sep='\t', index=False, quoting=csv.QUOTE_NONE)

    #Recover the spectrum ID assigned in the mgf file.
    db_search_df['SpecID'] = db_search_df['Title'].apply(
        lambda s: int(s.split('; SpectrumID: "')[1].split('"; scans: "')[0])
    )

    db_search_df = db_search_df[['SpecID', 'Peptide']]
    db_search_df.columns=['spec_id', 'ref seq']

    #REMOVE
    trans_dict = {ord(c): '' for c in config.mod_chars}
    db_search_df['ref seq'] = db_search_df['ref seq'].apply(lambda s: s.translate(trans_dict))

    db_search_df['ref seq'] = db_search_df['ref seq'].apply(lambda s: s.replace('I', 'L'))

    return db_search_df

def load_fasta_ref_file(fasta_ref_file_path):

    with open(fasta_ref_file_path) as f:
        lines = f.readlines()

    fasta_ref = []
    for line in lines:
        if line[0] == '>':
            next_seq_in_next_line = True
        elif line != '\n':
            if next_seq_in_next_line:
                fasta_ref.append(line.strip().replace('I', 'L'))
                next_seq_in_next_line = False
            else:
                fasta_ref[-1] += line.strip().replace('I', 'L')

    return fasta_ref

def match_seq_to_fasta_ref(denovo_seq, fasta_ref, print_percent_progress_fn):

    print_percent_progress_fn()

    for fasta_seq in fasta_ref:
        if denovo_seq in fasta_seq:
            return 1
    return 0

def standardize_prediction_df_cols(prediction_df):

    prediction_df.drop('is top rank single alg', inplace=True)
    min_retention_time = prediction_df['retention time'].min()
    max_retention_time = prediction_df['retention time'].max()
    #This if statement is for the purpose of debugging with a single spectrum.
    if max_retention_time - min_retention_time == 0:
        prediction_df['retention time'] = 0.5
    else:
        prediction_df['retention time'] = \
            (prediction_df['retention time'] - min_retention_time) / \
            (max_retention_time - min_retention_time)
    prediction_df.sort_index(1, inplace=True)

    return prediction_df

def update_training_data(prediction_df):

    prediction_df['timestamp'] = str(datetime.datetime.now()).split('.')[0]
    prediction_df.reset_index(inplace = True)
    try:
        training_df = pd.read_csv(join(config.data_dir, 'training_df.csv'))
        training_df = pd.concat([training_df, prediction_df])
    except (FileNotFoundError, OSError) as e:
        training_df = prediction_df
    training_df.set_index(['timestamp', 'spec_id'], inplace=True)
    training_df.to_csv(join(config.data_dir, 'training_df.csv'))
    training_df.reset_index(inplace=True)
    training_df.set_index(config.globals['is_alg_names'], inplace=True)

    return training_df

def make_predictions(prediction_df, input_df_dict=None, db_search_ref=None):

    def retrieve_single_alg_score_accuracy_df(alg):
        '''
        Calculate the accuracy of individual algorithms' de novo sequences.
        '''

        #Consider the de novo sequences predicted at standard fragment mass tolerances.
        if config.globals['frag_resolution'] == 'low':
            frag_mass_tol = '0.5'
        elif config.globals['frag_resolution'] == 'high':
            frag_mass_tol = '0.05'

        #Load the dataset.
        if alg == 'novor':
            score_col_name = 'avg aa score'
            for novor_fp in config.globals['novor_fps']:
                if frag_mass_tol + '.novor.csv' in novor_fp:
                    single_alg_df = input.load_novor_file(novor_fp)
                    break
        elif alg == 'pn':
            score_col_name = 'rank score'
            for pn_fp in config.globals['pn_fps']:
                if frag_mass_tol + '.mgf.out' in pn_fp:
                    single_alg_df = input.load_pn_file(pn_fp)
                    break
        elif alg == 'deepnovo':
            score_col_name = 'avg aa score'
            for deepnovo_fp in config.globals['deepnovo_fps']:
                if frag_mass_tol + '.deepnovo.tab' in deepnovo_fp:
                    single_alg_df = input.load_deepnovo_file(deepnovo_fp)
                    break
        #REMOVE
        #single_alg_df = input_df_dict[alg][frag_mass_tol]

        single_alg_df = single_alg_df.groupby(
            single_alg_df.index.get_level_values('spec_id')
        ).first()
        single_alg_df = single_alg_df[['seq', score_col_name]]
        single_alg_df.reset_index(inplace=True)
        #Record the lengths of predictions and references to calculate amino acid level statistics.
        single_alg_df['predict len'] = single_alg_df['seq'].apply(lambda x: len(x))
        single_alg_df = single_alg_df[single_alg_df['predict len'] >= config.globals['min_len']]

        single_alg_df = single_alg_df.merge(db_search_ref, how='left', on='spec_id')
        single_alg_df['ref seq'][single_alg_df['ref seq'].isnull()] = ''
        single_alg_df['ref match'] = find_seqs_in_paired_seqs(
            single_alg_df['seq'].tolist(), 
            single_alg_df['ref seq'].tolist()
        )

        single_alg_df = single_alg_df.set_index('spec_id')[
            [score_col_name, 'ref match', 'predict len']
        ]
        single_alg_df.columns = ['score', 'ref match', 'predict len']

        return single_alg_df

    #Full comparison plots
    #Make a reported prediction df that goes down to score = 0.
    #Retrieve the three top single-alg predictions.
    #Make a new df with probability cols and paired ref match cols.
    #Use the df in tailored precision-recall and precision-yield fns.

    #REMOVE
    print(config.data_dir, flush=True)
    forest_dict = utils.load_pkl_objects(config.data_dir, 'forest_dict')

    #Run the Postnovo model for each combination of algorithms (single alg seqs, consensus seqs).
    prediction_df['probability'] = np.nan
    for multiindex_key in config.globals['is_alg_keys']:
        #REMOVE
        print(str(multiindex_key), flush=True)
        alg_group = tuple(
            [alg for i, alg in enumerate(config.globals['algs']) if multiindex_key[i]]
        )
        #REMOVE
        print(alg_group, flush=True)

        alg_group_data = prediction_df.xs(multiindex_key)
        if config.globals['mode'] == 'predict':
            #Remove the cols that are not features in ANY random forest.
            alg_group_data.reset_index(level='spec_id', inplace=True)
            alg_group_data.drop(
                ['spec_id', 'seq', 'scan', 'probability', 'measured mass', 'mass error'], 
                axis=1, 
                inplace=True
            )
        elif config.globals['mode'] == 'test':                
            #Remove the cols that are not features in ANY random forest.
            alg_group_data.reset_index(level='spec_id', inplace=True)
            alg_group_data.drop(
                [
                    'spec_id', 
                    'seq', 
                    'scan', 
                    'ref match', 
                    'probability', 
                    'measured mass', 
                    'mass error'
                ], 
                axis=1, 
                inplace=True
            )
        #Remove empty cols that are not features in the appropriate rf.
        alg_group_data.dropna(1, inplace=True)
        forest_dict[alg_group].n_jobs = config.globals['cpus']
        probabilities = forest_dict[alg_group].predict_proba(alg_group_data.as_matrix())[:, 1]

        #Add the predicted probabilities for the alg combination to the full table of results.
        prediction_df.loc[multiindex_key, 'probability'] = probabilities

        #UNCOMMENT (REMOVE?)
        #if config.globals['mode'] == 'test':
        #    utils.verbose_print('Making', '_'.join(alg_group), 'test plots')
        #    #plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data)
        #    postnovo_alg_combo_df = prediction_df.xs(multiindex_key)
        #    postnovo_alg_combo_df = postnovo_alg_combo_df.reset_index().set_index('scan')[['probability', 'ref match']]
        #    max_probabilities = postnovo_alg_combo_df.groupby(level='scan')['probability'].transform(max)
        #    postnovo_alg_combo_df = postnovo_alg_combo_df[postnovo_alg_combo_df['probability'] == max_probabilities][['probability', 'ref match']]
        #    postnovo_alg_combo_df = postnovo_alg_combo_df.groupby(level='scan').first()

        #    plot_precision_recall_curve(postnovo_alg_combo_df, alg_score_accuracy_df_dict, len(db_search_ref))
        #    plot_precision_yield_curve(postnovo_alg_combo_df, alg_score_accuracy_df_dict, len(db_search_ref))

    prediction_df = prediction_df.reset_index().set_index('spec_id')

    if config.globals['max_total_sacrifice'] > 0:
        # Recover the longest seq prediction that fulfills the score sacrifice conditions
        # First, filter to those predictions with prob score above <sacrifice_floor>
        above_sac_floor_df = prediction_df[
            prediction_df['probability'] >= config.globals['sacrifice_floor']
        ]
        # Loop through the predictions for each spectrum
        above_sac_floor_spec_dfs = [df for _, df in above_sac_floor_df.groupby(level='spec_id')]

        one_percent_number_specs_above_sac_floor = \
            len(above_sac_floor_spec_dfs) / 100 / config.globals['cpus']
        max_total_sacrifice = config.globals['max_total_sacrifice']
        sacrifice_extension_ratio = config.globals['max_sacrifice_per_percent_extension'] * 100

        ##Single-threaded
        #for spec_df in above_sac_floor_spec_dfs:
        #    one_percent_number_specs_above_sac_floor * config.globals['cpus']
        #    print_percent_progress_fn = partial(
        #        utils.print_percent_progress_multithreaded,
        #        procedure_str='Score-length tradeoff progress: ',
        #        one_percent_total_count=one_percent_number_specs_above_sac_floor
        #        )
        #initialize_workers(
        #    print_percent_progress_fn, 
        #    max_total_sacrifice, 
        #    sacrifice_extension_ratio
        #    )
        #    reported_above_sac_floor_spec_dfs = []
        #    for spec_df in above_sac_floor_spec_dfs:
        #        reported_above_sac_floor_spec_dfs.append(do_score_sacrifice_extension(spec_df))

        #Multiprocessing
        print_percent_progress_fn = partial(
            utils.print_percent_progress_multithreaded,
            procedure_str='Score-length tradeoff progress: ',
            one_percent_total_count=one_percent_number_specs_above_sac_floor,
            cores=config.globals['cpus']
        )
        mp_pool = multiprocessing.Pool(
            config.globals['cpus'], 
            initializer=initialize_workers, 
            initargs=(
                print_percent_progress_fn, 
                max_total_sacrifice, 
                sacrifice_extension_ratio
            )
        )
        reported_above_sac_floor_spec_dfs = mp_pool.map(
            do_score_sacrifice_extension, above_sac_floor_spec_dfs
        )
        mp_pool.close()
        mp_pool.join()
        reported_prediction_df = reported_above_sac_floor_df = \
            pd.concat(reported_above_sac_floor_spec_dfs)

        # Find the best predictions from below the sacrifice floor
        #Try statement for dealing with a single spectrum in debugging
        try:
            predictions_below_floor_df = pd.concat([
                spec_df for _, spec_df in prediction_df.groupby(level='spec_id') 
                if (spec_df['probability'] <= config.globals['sacrifice_floor']).all()
            ])
            max_probabilities = \
                predictions_below_floor_df.groupby(level='spec_id')['probability'].transform(max)
            best_predictions_below_floor_df = predictions_below_floor_df[
                predictions_below_floor_df['probability'] == max_probabilities
            ]
            reported_below_sac_floor_df = \
                best_predictions_below_floor_df.groupby(level='spec_id').first()

            # Concatenate the data from above and below the floor
            reported_prediction_df = pd.concat([
                reported_above_sac_floor_df, 
                reported_below_sac_floor_df
            ]).sort_index()
        except ValueError:
            reported_prediction_df.sort_index(inplace=True)
    else:
        max_probabilities = prediction_df.groupby(level='spec_id')['probability'].transform(max)
        best_prediction_df = prediction_df[prediction_df['probability'] == max_probabilities]
        best_prediction_df = best_prediction_df.groupby(level='spec_id').first()
        reported_prediction_df = best_prediction_df

    #Make plots from test data.
    if config.globals['mode'] == 'test':
        #Prepare single-alg "raw" de novo data for precision-recall and precision-yield plots.
        alg_score_accuracy_df_dict = OrderedDict().fromkeys(alg_group)
        for alg in alg_group:
            alg_score_accuracy_df_dict[alg] = retrieve_single_alg_score_accuracy_df(alg)

        #COMMENT
        #Load Peaks output for comparison.
        peaks_fp = os.path.join(
            config.globals['iodir'], 
            config.globals['filename'] + '.0.05.peaks.csv'
        )
        peaks_df = pd.read_csv(peaks_fp, header=0)
        peaks_df = peaks_df.groupby('Scan', as_index=False).first()
        peaks_df = peaks_df[['Scan', 'Peptide', 'ALC (%)', 'm/z']]
        peaks_df.columns = ['scan', 'seq', 'score', 'mz']
        peaks_df['scan'] = peaks_df['scan'].apply(lambda s: s.split(':')[1])

        index_dict = OrderedDict()
        with open(config.globals['mgf_fp']) as handle:
            for line in handle.readlines():
                if line[:6] == 'TITLE=':
                    spec_id = line.split('; SpectrumID: "')[1].split('"; scans: "')[0]
                elif line[:8] == 'PEPMASS=':
                    pepmass = line.split('PEPMASS=')[1].rstrip()
                elif line[:6] == 'SCANS=':
                    scan = line.split('SCANS=')[1].rstrip()
                elif line == 'END IONS\n':
                    if scan in index_dict:
                        index_dict[scan].append((pepmass, spec_id))
                    else:
                        index_dict[scan] = [(pepmass, spec_id)]

        spec_ids = []
        for scan, peaks_mz in zip(peaks_df['scan'].tolist(), peaks_df['mz'].tolist()):
            mgf_entries = index_dict[scan]
            if len(mgf_entries) == 1:
                spec_ids.append(mgf_entries[0][1])
            elif len(mgf_entries) > 1:
                for t in mgf_entries:
                    mgf_mz = float(t[0])
                    if (mgf_mz - 0.01) <= float(peaks_mz) <= (mgf_mz + 0.01):
                        spec_ids.append(int(t[1]))
                        break
                else:
                    raise AssertionError('Peaks and mgf data do not match up')
                    print('Peaks: scan = ' + scan + ', m/z = ' + peaks_mz)
                    print('mgf entries: ')
                    print(mgf_entries)
            else:
                raise AssertionError(scan + ' was not found in the mgf file')
        peaks_df['spec_id'] = spec_ids

        peaks_df['seq'] = peaks_df['seq'].apply(lambda s: utils.remove_mod_chars(seq=s))
        peaks_df['predict len'] = peaks_df['seq'].apply(len)
        peaks_df = peaks_df.merge(db_search_ref, how='left', on='spec_id')
        peaks_df['ref seq'][peaks_df['ref seq'].isnull()] = ''
        peaks_df['ref match'] = find_seqs_in_paired_seqs(
            peaks_df['seq'].tolist(), peaks_df['ref seq'].tolist()
        )
        peaks_df = peaks_df.set_index('spec_id')[['seq', 'scan', 'score', 'ref match', 'predict len']]
        peaks_df = peaks_df.sort_values('score', ascending=False)
        alg_score_accuracy_df_dict['peaks'] = peaks_df

        plot_precision_recall_curve(
            reported_prediction_df[['probability', 'ref match']], 
            alg_score_accuracy_df_dict, 
            len(db_search_ref), 
            all_postnovo_predictions=True
        )
        plot_precision_yield_curve(
            reported_prediction_df[['probability', 'ref match']], 
            alg_score_accuracy_df_dict, 
            len(db_search_ref), 
            all_postnovo_predictions=True
        )
        plot_aa_fraction_recall_curve(
            reported_prediction_df[['probability', 'ref match', 'predict len']], 
            alg_score_accuracy_df_dict, 
            all_postnovo_predictions=True
        )

    reported_cols_in_order = []
    for reported_df_col in config.reported_df_cols:
        if reported_df_col in reported_prediction_df.columns:
            reported_cols_in_order.append(reported_df_col)
    reported_prediction_df = reported_prediction_df.reindex_axis(reported_cols_in_order, axis = 1)

    return reported_prediction_df

def initialize_workers(
    _print_percent_progress_fn, 
    _max_total_sacrifice, 
    _sacrifice_extension_ratio
    ):

    global print_percent_progress_fn, max_total_sacrifice, sacrifice_extension_ratio

    print_percent_progress_fn = _print_percent_progress_fn
    max_total_sacrifice = _max_total_sacrifice
    sacrifice_extension_ratio = _sacrifice_extension_ratio

    return

def do_score_sacrifice_extension(spec_df):

    spec_df.sort_values('probability', ascending=False, inplace=True)
    probs = spec_df['probability'].tolist()
    highest_prob = probs[0]
    lower_prob_seqs = spec_df['seq'].tolist()[1:]
    current_longest_seq = spec_df.iloc[0]['seq']
    current_prob = highest_prob
    longest_row_index = 0
    for i, seq in enumerate(lower_prob_seqs):
        if len(seq) > len(current_longest_seq):
            # The longer seq must contain the shorter, higher-prob seq
            if current_longest_seq in seq:
                lower_prob = probs[i + 1]
                # The longer seq must be within <max_total_sacrifice> score of highest score
                if highest_prob - lower_prob <= max_total_sacrifice:
                    # Finally, the loss in score per percent extension must meet the criterion
                    length_weighted_max_sacrifice = sacrifice_extension_ratio * \
                        np.sum(np.reciprocal(np.arange(
                            len(current_longest_seq) + 1, len(seq) + 1, dtype=np.float16
                        )))
                    if current_prob - lower_prob <= length_weighted_max_sacrifice:
                        longest_row_index = i + 1
                        current_longest_seq = seq
                        current_prob = lower_prob

    return spec_df.iloc[[longest_row_index]]

def make_training_forests(training_df):

    train_target_arr_dict = make_train_target_arr_dict(training_df)
    
    if config.globals['mode'] == 'train':
        forest_dict = make_forest_dict(train_target_arr_dict, config.rf_default_params)

        ##REMOVE
        for alg_key in forest_dict:
            data_train_split, \
                data_validation_split, \
                target_train_split, \
                target_validation_split = \
                train_test_split(
                    train_target_arr_dict[alg_key]['train'], 
                    train_target_arr_dict[alg_key]['target'], 
                    stratify=train_target_arr_dict[alg_key]['target']
                )
            #plot_feature_importances(
            #    forest_dict[alg_key], 
            #    alg_key, 
            #    train_target_arr_dict[alg_key]['feature_names']
            #)
            plot_binned_feature_importances(
                forest_dict[alg_key], 
                alg_key, 
                train_target_arr_dict[alg_key]['feature_names']
            )
            #plot_errors(
            #    data_train_split, 
            #    data_validation_split, 
            #    target_train_split, 
            #    target_validation_split, 
            #    alg_key
            #)

    elif config.globals['mode'] == 'optimize':
        utils.verbose_print('Optimizing random forest parameters')
        optimized_params = optimize_model(train_target_arr_dict)
        forest_dict = make_forest_dict(train_target_arr_dict, optimized_params)

    return forest_dict

def make_train_target_arr_dict(training_df):

    training_df.sort_index(inplace=True)
    model_keys_used = []
    train_target_arr_dict = OrderedDict()
    for multiindex_key in config.globals['is_alg_keys']:
        print(str(multiindex_key), flush=True)
        model_key = tuple(
            [alg for i, alg in enumerate(config.globals['algs']) if multiindex_key[i]]
        )
        print(model_key, flush=True)
        model_keys_used.append(model_key)
        train_target_arr_dict[model_keys_used[-1]] = OrderedDict().fromkeys(['train', 'target'])
        try:
            alg_group_df = training_df.xs(multiindex_key).reset_index().set_index(
                ['spec_id', 'seq']
            )
            alg_group_df.dropna(1, inplace=True)
            train_columns = alg_group_df.columns.tolist()
            for c in config.globals['is_alg_names'] + [
                'ref match', 'mass error', 'measured mass', 'timestamp'
            ]:
                train_columns.remove(c)
            print(train_columns)
            train_target_arr_dict[model_key]['train'] = alg_group_df.as_matrix(train_columns)
            train_target_arr_dict[model_key]['target'] = alg_group_df['ref match'].tolist()
            train_target_arr_dict[model_key]['feature_names'] = train_columns
        except KeyError:
            print(str(model_keys_used[-1]) + ' predictions were not found')

    return train_target_arr_dict

def make_forest_dict(train_target_arr_dict, rf_params):

    forest_dict = OrderedDict().fromkeys(train_target_arr_dict)
    for alg_key in forest_dict:
        if len(alg_key) > 1:
            utils.verbose_print(
                'Making random forest for', '-'.join(alg_key), 'consensus sequences'
            )
        else:
            utils.verbose_print('Making random forest for', alg_key[0], 'sequences')

        train_data = train_target_arr_dict[alg_key]['train']
        target_data = train_target_arr_dict[alg_key]['target']
        forest = RandomForestClassifier(
            n_estimators=config.rf_n_estimators, 
            max_depth=rf_params[alg_key]['max_depth'], 
            max_features=rf_params[alg_key]['max_features'], 
            oob_score=True, 
            n_jobs=config.globals['cpus']
        )
        forest.fit(train_data, target_data)
        forest_dict[alg_key] = forest

    return forest_dict

def optimize_model(train_target_arr_dict):

    optimized_params = OrderedDict()
    for alg_key in train_target_arr_dict:
        optimized_params[alg_key] = OrderedDict()

        data_train_split, data_validation_split, target_train_split, target_validation_split = \
            train_test_split(
                train_target_arr_dict[alg_key]['train'], 
                train_target_arr_dict[alg_key]['target'], 
                stratify = train_target_arr_dict[alg_key]['target']
            )
        forest_grid = GridSearchCV(
            RandomForestClassifier(n_estimators = config.rf_n_estimators, oob_score = True), 
            {'max_features': ['sqrt', None], 'max_depth': [depth for depth in range(11, 20)]}, 
            n_jobs = config.globals['cpus']
        )
        forest_grid.fit(data_train_split, target_train_split)
        optimized_forest = forest_grid.best_estimator_
        optimized_params[alg_key]['max_depth'] = optimized_forest.max_depth
        utils.verbose_print(alg_key, 'optimized max depth:', optimized_forest.max_depth)
        optimized_params[alg_key]['max_features'] = optimized_forest.max_features
        utils.verbose_print(alg_key, 'optimized max features:', optimized_forest.max_features)

        plot_feature_importances(
            optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names']
        )
        plot_binned_feature_importances(
            optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names']
        )
        plot_errors(
            data_train_split, 
            data_validation_split, 
            target_train_split, 
            target_validation_split, 
            alg_key
        )

    return optimized_params

def plot_feature_importances(forest, alg_key, feature_names):
    if len(alg_key) > 1:
        utils.verbose_print(
            'Plotting feature importances for', '-'.join(alg_key), 'consensus sequences'
        )
    else:
        utils.verbose_print('Plotting feature importances for', alg_key[0], 'sequences')

    importances = forest.feature_importances_
    feature_std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots()
    ax.set_title('Feature importances')
    x = np.arange(len(importances))
    ax.bar(
        left=x, 
        height=importances[indices], 
        color='r', 
        yerr=feature_std[indices], 
        width=0.9, 
        align='center'
    )
    ax.set_xticks(x)
    labels = np.array(feature_names)[indices]
    ax.set_xticklabels(labels, rotation=-45, ha='left')
    ax.set_xlim([-1, len(importances)])
    ax.set_ylim(ymin = 0)
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(config.globals['iodir'], alg_key_str + '_feature_importances.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_binned_feature_importances(forest, alg_key, feature_names):
    if len(alg_key) > 1:
        utils.verbose_print(
            'Plotting feature importances for', '-'.join(alg_key), 'consensus sequences'
        )
    else:
        utils.verbose_print('Plotting feature importances for', alg_key[0], 'sequences')

    feature_importances = forest.feature_importances_
    feature_group_importances = []
    feature_group_stds = []
    for feature_group, features in config.feature_groups.items():
        feature_group_importance = 0.0
        feature_group_var = 0.0
        for i, feature_name in enumerate(feature_names):
            if feature_name in features:
                feature_group_importance += feature_importances[i]
                feature_group_var += np.var(
                    [tree.feature_importances_[i] for tree in forest.estimators_], axis = 0)
        feature_group_importances.append(feature_group_importance)
        feature_group_stds.append(np.sqrt(feature_group_var))

    feature_group_importances = np.array(feature_group_importances)
    feature_group_stds = np.array(feature_group_stds)
    indices = np.argsort(feature_group_importances)[::-1]

    fig, ax = plt.subplots()
    ax.set_title('Binned feature importances')
    x = np.arange(len(feature_group_importances))
    ax.bar(
        left=x, 
        height=feature_group_importances[indices], 
        color='r', 
        yerr=feature_group_stds[indices], 
        width=0.9, 
        align='center'
    )
    ax.set_xticks(x)
    labels = np.array(list(config.feature_groups))[indices]
    ax.set_xticklabels(labels, rotation = -45, ha = 'left')
    ax.set_xlim([-1, len(feature_group_importances)])
    ax.set_ylim(ymin = 0)
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(config.globals['iodir'], alg_key_str + '_binned_feature_importances.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_errors(
    data_train_split, 
    data_validation_split, 
    target_train_split, 
    target_validation_split, 
    alg_key
):
    if len(alg_key) > 1:
        utils.verbose_print(
            'Plotting errors vs tree size for', '-'.join(alg_key), 'consensus sequences'
        )
    else:
        utils.verbose_print('Plotting errors vs tree size for', alg_key[0], 'sequences')

    ensemble_clfs = [
        #('max_features=\'sqrt\'',
        # RandomForestClassifier(warm_start = True, max_features = 'sqrt', oob_score = True, max_depth = 15, n_jobs = config.globals['cpus'], random_state = 1)),
        (
            'max_features=None', 
            RandomForestClassifier(
                warm_start=True, 
                max_features=None, 
                oob_score=True, 
                max_depth=15, 
                n_jobs=config.globals['cpus'], 
                random_state=1
            )
         )
    ]

    oob_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
    #validation_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
    min_estimators = 10
    max_estimators = 500

    for label, clf in ensemble_clfs:
        for tree_number in range(min_estimators, max_estimators + 1, 100):
            clf.set_params(n_estimators = tree_number)
            clf.fit(data_train_split, target_train_split)

            oob_error = 1 - clf.oob_score_
            oob_errors[label].append((tree_number, oob_error))

            #validation_error = 1 - clf.score(data_validation_split, target_validation_split)
            #validation_errors[label].append((tree_number, validation_error))

    fig, ax1 = plt.subplots()
    for label, oob_error in oob_errors.items():
        xs, ys = zip(*oob_error)
        ax1.plot(xs, ys, label = 'oob error: ' + label)
    #for label, validation_error in validation_errors.items():
    #    xs, ys = zip(*validation_error)
    #    ax1.plot(xs, ys, label = 'validation error: ' + label)

    ax1.set_xlim(min_estimators, max_estimators)
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('error rate')
    ax1.legend(loc='upper right')
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(config.globals['iodir'], alg_key_str + '_error.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        accuracy_labels, probabilities, pos_label=1
    )
    model_auc = roc_auc_score(accuracy_labels, probabilities)

    alg_scores_dict = OrderedDict()
    for alg in alg_group:
        if alg == 'novor':
            alg_scores_dict[alg] = alg_group_data['avg novor aa score']
        elif alg == 'pn':
            alg_scores_dict[alg] = alg_group_data['rank score']

    alg_roc_dict = OrderedDict()
    alg_auc_dict = OrderedDict()
    for alg in alg_scores_dict:
        alg_roc_dict[alg] = roc_curve(accuracy_labels, alg_scores_dict[alg], pos_label=1)
        alg_auc_dict[alg] = roc_auc_score(accuracy_labels, alg_scores_dict[alg])

    fig, ax = plt.subplots()

    model_line_collection = colorline(false_positive_rate, true_positive_rate, thresholds)
    plt.colorbar(
        model_line_collection, 
        label='Moving threshold:\nrandom forest probability or\nde novo algorithm score percentile'
    )
    annotation_x = false_positive_rate[len(false_positive_rate) // 2]
    annotation_y = true_positive_rate[len(true_positive_rate) // 2]
    plt.annotate(
        'random forest\nauc = ' + str(round(model_auc, 2)), 
        xy=(annotation_x, annotation_y), 
        xycoords='data', 
        xytext=(annotation_x + 50, annotation_y - 50), 
        textcoords='offset pixels', 
        arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
        horizontalalignment='left', 
        verticalalignment='top'
    )

    arrow_position = 3
    for alg in alg_roc_dict:
        alg_fpr = alg_roc_dict[alg][0]
        alg_tpr = alg_roc_dict[alg][1]
        alg_thresh = alg_roc_dict[alg][2].argsort() / alg_roc_dict[alg][2].size
        annotation_x = alg_fpr[len(alg_fpr) // arrow_position]
        annotation_y = alg_tpr[len(alg_tpr) // arrow_position]
        arrow_position -= 1
        colorline(alg_fpr, alg_tpr, alg_thresh)
        plt.annotate(
            alg + '\nauc = ' + str(round(alg_auc_dict[alg], 2)), 
            xy=(annotation_x, annotation_y), 
            xycoords='data', 
            xytext=(annotation_x + 50, annotation_y - 50), 
            textcoords='offset pixels', 
            arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
            horizontalalignment='left', 
            verticalalignment='top'
        )

    plt.plot([0, 1], [0, 1], linestyle='--', c='black')

    if len(alg_group) == 1:
        plt.title('roc curve: ' + alg_group[0] + ' sequences')
    else:
        plt.title('roc curve: ' + '-'.join(alg_group) + ' consensus sequences')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('false positive rate = ' + r'$\frac{F_p}{F_p + T_n}$')
    plt.ylabel('true positive rate = ' + r'$\frac{T_p}{T_p + F_n}$')
    plt.tight_layout(True)

    save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_roc.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_precision_recall_curve(
    postnovo_alg_combo_df, 
    alg_score_accuracy_df_dict, 
    db_search_yield, 
    all_postnovo_predictions=False
):

    postnovo_seq_recall = []
    postnovo_seq_precision = []
    postnovo_seq_thresholds = []
    #Record the cumulative number of correct Postnovo seqs.
    cumulative_true_positives = 0
    cumulative_labeled_positives = 0
    #Loop through sequences in descending order of score.
    for score, ref_match in sorted(
        zip(
            postnovo_alg_combo_df['probability'].tolist(), 
            postnovo_alg_combo_df['ref match'].tolist(), 
        ), 
        key=lambda t: -t[0]
    ):
        if ref_match:
            cumulative_true_positives += 1
        cumulative_labeled_positives += 1
        postnovo_seq_recall.append(cumulative_labeled_positives / db_search_yield)
        postnovo_seq_precision.append(cumulative_true_positives / cumulative_labeled_positives)
        postnovo_seq_thresholds.append(score)

    #Record the cumulative number of correct de novo alg seqs.
    alg_seq_recall_dict = OrderedDict()
    alg_seq_precision_dict = OrderedDict()
    alg_seq_threshold_dict = OrderedDict()
    for alg, alg_score_accuracy_df in alg_score_accuracy_df_dict.items():
        alg_seq_recall = []
        alg_seq_precision = []
        alg_seq_thresholds = []
        cumulative_true_positives = 0
        cumulative_labeled_positives = 0
        for score, ref_match in sorted(
            zip(
                alg_score_accuracy_df['score'].tolist(), 
                alg_score_accuracy_df['ref match'].tolist()
            ), 
            key=lambda t: -t[0]
        ):
            if ref_match:
                cumulative_true_positives += 1
            cumulative_labeled_positives += 1
            alg_seq_recall.append(cumulative_labeled_positives / db_search_yield)
            alg_seq_precision.append(cumulative_true_positives / cumulative_labeled_positives)
            alg_seq_thresholds.append(score)
        alg_seq_recall_dict[alg] = alg_seq_recall
        alg_seq_precision_dict[alg] = alg_seq_precision
        alg_seq_threshold_dict[alg] = alg_seq_thresholds

    #Plot the precision-recall curve for Postnovo.
    fig, ax = plt.subplots()

    line_collection = colorline(
        postnovo_seq_recall, postnovo_seq_precision, postnovo_seq_thresholds
    )
    cb = plt.colorbar(
        line_collection, 
        label=(
            'Moving threshold:\nPostnovo score or\n'
            'de novo algorithm score percentile'
        )
    )

    #Plot the precision-recall curves for the individual algs.
    #COMMENT
    arrow_position = 8
    for alg, alg_seq_recall in alg_seq_recall_dict.items():
        alg_seq_precision = alg_seq_precision_dict[alg]
        alg_seq_thresholds = alg_seq_threshold_dict[alg]

        #COMMENT
        #Plot arrows labeling the curves by alg.
        annotation_x = alg_seq_recall[int(len(alg_seq_recall) / arrow_position)]
        annotation_y = alg_seq_precision[int(len(alg_seq_precision) / arrow_position)]
        plt.annotate(
            alg, 
            xy=(annotation_x, annotation_y), 
            xycoords='data', 
            xytext = (annotation_x - 25, annotation_y - 25), 
            textcoords = 'offset pixels', 
            arrowprops = dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
            horizontalalignment='right', 
            verticalalignment='top'
        )

        if alg == 'novor':
            line_collection = colorline(
                alg_seq_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(0, 100)
            )
            tick_locs = [0, 20, 40, 60, 80, 100]
        elif alg == 'pn':
            line_collection = colorline(
                alg_seq_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(-10, 15)
            )
            tick_locs = [-10, -5, 0, 5, 10, 15]
        elif alg == 'deepnovo':
            line_collection = colorline(
                alg_seq_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(0, 1)
            )
            tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
        elif alg == 'peaks':
            line_collection = colorline(
                alg_seq_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(0, 100)
            )
            tick_locs = [0, 20, 40, 60, 80, 100]

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout(True)

    if all_postnovo_predictions:
        save_path = join(config.globals['iodir'], 'full_precision_recall.pdf')
    else:
        save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_precision_recall.pdf')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()

    return

def plot_precision_yield_curve(
    postnovo_alg_combo_df, 
    alg_score_accuracy_df_dict, 
    db_search_yield, 
    all_postnovo_predictions=False
):

    postnovo_seq_yield = []
    postnovo_seq_precision = []
    postnovo_seq_thresholds = []
    #Record the cumulative number of correct Postnovo seqs.
    cumulative_true_positives = 0
    cumulative_labeled_positives = 0
    #Loop through sequences in descending order of score.
    for score, ref_match in sorted(
        zip(
            postnovo_alg_combo_df['probability'].tolist(), 
            postnovo_alg_combo_df['ref match'].tolist(), 
        ), 
        key=lambda t: -t[0]
    ):
        if ref_match:
            cumulative_true_positives += 1
        cumulative_labeled_positives += 1
        postnovo_seq_yield.append(cumulative_labeled_positives)
        postnovo_seq_precision.append(cumulative_true_positives / cumulative_labeled_positives)
        postnovo_seq_thresholds.append(score)

    #Record the cumulative number of correct de novo alg seqs.
    alg_seq_yield_dict = OrderedDict()
    alg_seq_precision_dict = OrderedDict()
    alg_seq_threshold_dict = OrderedDict()
    for alg, alg_score_accuracy_df in alg_score_accuracy_df_dict.items():
        alg_seq_yield = []
        alg_seq_precision = []
        alg_seq_thresholds = []
        cumulative_true_positives = 0
        cumulative_labeled_positives = 0
        for score, ref_match in sorted(
            zip(
                alg_score_accuracy_df['score'].tolist(), 
                alg_score_accuracy_df['ref match'].tolist()
            ), 
            key=lambda t: -t[0]
        ):
            if ref_match:
                cumulative_true_positives += 1
            cumulative_labeled_positives += 1
            alg_seq_yield.append(cumulative_labeled_positives)
            alg_seq_precision.append(cumulative_true_positives / cumulative_labeled_positives)
            alg_seq_thresholds.append(score)
        alg_seq_yield_dict[alg] = alg_seq_yield
        alg_seq_precision_dict[alg] = alg_seq_precision
        alg_seq_threshold_dict[alg] = alg_seq_thresholds

    #Plot the precision-yield curve for Postnovo.
    fig, ax = plt.subplots()

    max_encountered_yield = 0
    db_search_precision = 1 - config.max_fdr

    #postnovo_seq_yield_pts = postnovo_seq_yield[::10]
    #postnovo_seq_precision_pts = postnovo_seq_precision[::10]
    #postnovo_seq_threshold_pts = postnovo_seq_thresholds[::10]
    postnovo_seq_yield_pts = postnovo_seq_yield
    postnovo_seq_precision_pts = postnovo_seq_precision
    postnovo_seq_threshold_pts = postnovo_seq_thresholds


    line_collection = colorline(
        postnovo_seq_yield_pts, postnovo_seq_precision_pts, postnovo_seq_threshold_pts
    )
    cb = plt.colorbar(
        line_collection, 
        label=(
            'Moving threshold:\nPostnovo score or\n'
            'de novo algorithm score percentile'
        )
    )

    #Push the x-axis maximum to the highest yield.
    #Yield can theoretically be higher than db search yield, 
    #if more de novo seqs are found than db search PSMs.
    if postnovo_seq_yield_pts[-1] > max_encountered_yield:
        max_encountered_yield = postnovo_seq_yield_pts[-1]    

    #Plot precision-yield curves for the individual algs.
    #COMMENT
    arrow_position = 5
    for alg, alg_seq_yield in alg_seq_yield_dict.items():
        alg_seq_precision = alg_seq_precision_dict[alg]
        alg_seq_thresholds = alg_seq_threshold_dict[alg]

        #alg_seq_yield_pts = alg_seq_yield[::10]
        #alg_seq_precision_pts = alg_seq_precision[::10]
        #alg_seq_threshold_pts = alg_seq_thresholds[::10]
        alg_seq_yield_pts = alg_seq_yield
        alg_seq_precision_pts = alg_seq_precision
        alg_seq_threshold_pts = alg_seq_thresholds

        if alg == 'novor':
            line_collection = colorline(
                alg_seq_yield_pts, 
                alg_seq_precision_pts, 
                alg_seq_threshold_pts, 
                norm=plt.Normalize(0, 100)
            )
            tick_locs = [0, 20, 40, 60, 80, 100]
        elif alg == 'pn':
            line_collection = colorline(
                alg_seq_yield_pts, 
                alg_seq_precision_pts, 
                alg_seq_threshold_pts, 
                norm=plt.Normalize(-10, 15)
            )
            tick_locs = [-10, -5, 0, 5, 10, 15]
        elif alg == 'deepnovo':
            line_collection = colorline(
                alg_seq_yield_pts, 
                alg_seq_precision_pts, 
                alg_seq_threshold_pts, 
                norm=plt.Normalize(0, 1)
            )
            tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if alg == 'peaks':
            line_collection = colorline(
                alg_seq_yield_pts, 
                alg_seq_precision_pts, 
                alg_seq_threshold_pts, 
                norm=plt.Normalize(0, 100)
            )
            tick_locs = [0, 20, 40, 60, 80, 100]

        annotation_x = alg_seq_yield_pts[int(len(alg_seq_yield_pts) / arrow_position)]
        annotation_y = alg_seq_precision_pts[int(len(alg_seq_precision_pts) / arrow_position)]
        arrow_position -= 0.5
        plt.annotate(
            alg,
            xy=(annotation_x, annotation_y),
            xycoords='data',
            xytext = (-25, -25),
            textcoords='offset pixels',
            arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=6),
            horizontalalignment='right', 
            verticalalignment='top'
            )

        if alg_seq_yield_pts[-1] > max_encountered_yield:
            max_encountered_yield = alg_seq_yield_pts[-1]

    ax.plot(db_search_yield, db_search_precision, color='r', marker='*', markersize=10)
    if db_search_yield > max_encountered_yield:
        max_encountered_yield = db_search_yield + 1000
    plt.xlim([1, max_encountered_yield])
    plt.ylim([0, 1])

    plt.xlabel('Yield')
    plt.ylabel('Precision')
    plt.tight_layout(True)

    plt.tight_layout(True)
    if all_postnovo_predictions:
        save_path = join(config.globals['iodir'], 'full_precision_yield.pdf')
    else:
        save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_precision_yield.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')
    plt.close()

    return

def plot_aa_fraction_recall_curve(
    postnovo_alg_combo_df, 
    alg_score_accuracy_df_dict, 
    all_postnovo_predictions=False
):

    postnovo_aa_recall = []
    postnovo_seq_precision = []
    postnovo_seq_thresholds = []
    #Record the cumulative length of amino acids in correct Postnovo seqs.
    cumulative_len = 0
    cumulative_true_positives = 0
    cumulative_labeled_positives = 0
    #Loop through sequences in descending order of score.
    for score, ref_match, predict_len in sorted(
        zip(
            postnovo_alg_combo_df['probability'].tolist(), 
            postnovo_alg_combo_df['ref match'].tolist(), 
            postnovo_alg_combo_df['predict len'].tolist()
        ), 
        key=lambda t: -t[0]
    ):
        #If the de novo seq is found within a db search PSM, 
        #add the number of amino acids in the de novo seq.
        if ref_match:
            cumulative_len += predict_len
            cumulative_true_positives += 1
        cumulative_labeled_positives += 1
        postnovo_aa_recall.append(cumulative_len / ref_aa_sum)
        postnovo_seq_precision.append(cumulative_true_positives / cumulative_labeled_positives)
        postnovo_seq_thresholds.append(score)

    #Record the cumulative length of amino acids in correct individual de novo alg seqs.
    alg_aa_recall_dict = OrderedDict()
    alg_seq_precision_dict = OrderedDict()
    alg_seq_threshold_dict = OrderedDict()
    for alg, alg_score_accuracy_df in alg_score_accuracy_df_dict.items():
        alg_aa_recall = []
        alg_seq_precision = []
        alg_seq_thresholds = []
        cumulative_len = 0
        cumulative_true_positives = 0
        cumulative_labeled_positives = 0
        for score, ref_match, predict_len in sorted(
            zip(
                alg_score_accuracy_df['score'].tolist(), 
                alg_score_accuracy_df['ref match'].tolist(), 
                alg_score_accuracy_df['predict len'].tolist()
            ), 
            key=lambda t: -t[0]
        ):
            if ref_match:
                cumulative_len += predict_len
                cumulative_true_positives += 1
            cumulative_labeled_positives += 1
            alg_aa_recall.append(cumulative_len / ref_aa_sum)
            alg_seq_precision.append(cumulative_true_positives / cumulative_labeled_positives)
            alg_seq_thresholds.append(score)
        alg_aa_recall_dict[alg] = alg_aa_recall
        alg_seq_precision_dict[alg] = alg_seq_precision
        alg_seq_threshold_dict[alg] = alg_seq_thresholds

    fig, ax = plt.subplots()

    #Plot the precision-aa recall curve for Postnovo.
    line_collection = colorline(
        postnovo_aa_recall, postnovo_seq_precision, postnovo_seq_thresholds
    )
    cb = plt.colorbar(
        line_collection, 
        label=(
            'Moving threshold:\nPostnovo score or\n'
            'de novo algorithm score percentile'
        )
    )

    #Plot the precision-aa recall curves for the individual algs.
    #COMMENT
    arrow_position = 3
    for alg, alg_aa_recall in alg_aa_recall_dict.items():
        alg_seq_precision = alg_seq_precision_dict[alg]
        alg_seq_thresholds = alg_seq_threshold_dict[alg]

        #COMMENT
        #Plot arrows labeling the curves by alg.
        annotation_x = alg_aa_recall[int(len(alg_aa_recall) / arrow_position)]
        annotation_y = alg_seq_precision[int(len(alg_seq_precision) / arrow_position)]
        plt.annotate(
            alg, 
            xy=(annotation_x, annotation_y), 
            xycoords='data', 
            xytext = (annotation_x - 25, annotation_y - 25), 
            textcoords = 'offset pixels', 
            arrowprops = dict(facecolor='black', shrink=0.01, width=1, headwidth=6), 
            horizontalalignment='right', 
            verticalalignment='top'
        )

        if alg == 'novor':
            line_collection = colorline(
                alg_aa_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(0, 100)
            )
            tick_locs = [0, 20, 40, 60, 80, 100]
        elif alg == 'pn':
            line_collection = colorline(
                alg_aa_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(-10, 15)
            )
            tick_locs = [-10, -5, 0, 5, 10, 15]
        elif alg == 'deepnovo':
            line_collection = colorline(
                alg_aa_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(0, 1)
            )
            tick_locs = [0, 0.2, 0.4, 0.6, 0.8, 1]
        elif alg == 'peaks':
            line_collection = colorline(
                alg_aa_recall, alg_seq_precision, alg_seq_thresholds, norm=plt.Normalize(0, 100)
            )
            tick_locs = [0, 20, 40, 60, 80, 100]

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Amino Acid Recall')
    plt.ylabel('Sequence Precision')
    plt.tight_layout(True)

    if all_postnovo_predictions:
        save_path = join(config.globals['iodir'], 'full_aa_recall.pdf')
    else:
        save_path = join(config.globals['iodir'], '_'.join(alg_group) + '_aa_recall.pdf')
    fig.savefig(save_path, bbox_inches='tight')

    return

def colorline(x, y, z, cmap='jet', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):

    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_collection = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax = plt.gca()
    ax.add_collection(line_collection)
    return line_collection

def make_precision_list(df, sort_col):
    # Make a list of cumulative precision values as successively lower-scoring seqs are considered

    df.sort_values(sort_col, ascending=False, inplace=True)
    ref_matches = df['ref match'].tolist()
    cumulative_ref_matches = [ref_matches[0]]
    precision_list = [cumulative_ref_matches[0] / 1]
    for i, ref_match in enumerate(ref_matches[1:]):
        cumulative_ref_match_sum = cumulative_ref_matches[-1] + ref_match
        cumulative_ref_matches.append(cumulative_ref_match_sum)
        precision_list.append(cumulative_ref_match_sum / (i + 2))

    return precision_list