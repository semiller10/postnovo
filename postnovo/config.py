''' Variables used across project '''

import numpy as np

from os.path import join, dirname, realpath
from itertools import product
from collections import OrderedDict


# standard constants
_proton_mass = 1.007276
_seconds_in_min = 60
# ms_aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
_unicode_decimal_A = 65

# user args
# run level settings: predict (default), train, test, optimize
_run_type = ['predict']
_verbose = [True]
_novor_files = []
_novor_tols = []
_peaks_files = []
_peaks_tols = []
_pn_files = []
_pn_tols = []
_min_prob = [0.5]
_min_len = [6]
_ref_file = []
_cores = [1]

# directories
_postnovo_parent_dir = dirname(dirname(realpath(__file__)))
_test_dir = join(_postnovo_parent_dir, 'test')
_training_dir = join(_postnovo_parent_dir, 'training')
_userfiles_dir = join(_postnovo_parent_dir, 'userfiles')
_output_dir = join(_postnovo_parent_dir, 'output')

# program constraints
_accepted_algs = ['novor', 'peaks', 'pn']
_possible_alg_combos = []
for numerical_alg_combo in list(product((0, 1), repeat = len(_accepted_algs)))[1:]:
    _possible_alg_combos.append(tuple([alg for i, alg in enumerate(_accepted_algs) if numerical_alg_combo[i]]))
_seqs_reported_per_alg_dict = {'novor': 1, 'peaks': 20, 'pn': 20}
_accepted_mass_tols = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']

# global info from user input
_alg_list = []
_tol_list = []
_alg_tols_dict = OrderedDict()
## example
## _alg_tols_dict = odict('novor': odict('0.4': 'proteome-0.4.novor.csv', '0.5': 'proteome-0.5.novor.csv'),
##                     'pn': odict('0.4': 'proteome-0.4.mgf.out', '0.5': 'proteome-0.5.mgf.out'))
_tol_alg_dict = OrderedDict()
## example
## tol_alg_dict = odict('0.4': ['novor', 'pn'], '0.5': ['novor', 'pn'])
_tol_basenames_dict = OrderedDict()
## example
## _tol_basenames_dict = odict('0.4': ['proteome-0.4.novor.csv', 'proteome-0.4.mgf.out'],
##                              '0.5': ['proteome-0.5.novor.csv', 'proteome-0.5.mgf.out'])

# de novo output characteristics
_novor_dropped_chars = {ord(char): None for char in
                       ''.join([str(i) for i in range(10)] + ['(', ')'])}
_pn_dropped_chars = {ord(char): None for char in
                       ''.join([str(i) for i in range(10)] + ['+', '-', '.'])}

# training parameters
_train_consensus_len = 6
_rf_n_estimators = 150
_rf_default_optimized_params = {alg_combo: {'max_depth': 15, 'max_features': 'sqrt'} for alg_combo in _possible_alg_combos}

# feature selection from input data
_prediction_dict_source_cols = {'novor': ['retention time', 'seq', 'aa score', 'avg aa score', 'encoded seq'],
                               'peaks': [],
                               'pn': ['seq', 'rank score', 'pn score', 'sqs', 'encoded seq']}
_single_alg_prediction_dict_cols = {'general': ['scan', 'is top rank single alg', 'seq', 'len', 'avg rank'],
                                   'novor': ['retention time', 'is novor seq', 'avg novor aa score'],
                                   'peaks': [],
                                   'pn': ['is pn seq', 'rank score', 'pn score', 'pn rank', 'sqs']}
_consensus_prediction_dict_cols = {'general': ['scan', 'seq', 'len', 'avg rank', 'is longest consensus', 'is top rank consensus'],
                                  'novor': ['retention time', 'is novor seq', 'fraction novor parent len', 'avg novor aa score'],
                                  'peaks': [],
                                  'pn': ['is pn seq', 'fraction pn parent len', 'rank score', 'pn score', 'pn rank', 'sqs']}

# parameters for subsampling training data
_subsample_size = 20000
_subsample_accuracy_divisor = 10
_subsample_accuracy_distribution_lower_bound = 0
_subsample_accuracy_distribution_upper_bound = 3
_subsample_accuracy_distribution_sigma = 0.9
_subsample_accuracy_distribution_mu_location = 0.5
_clustering_min_retained_features = 2
_clustering_feature_retention_factor_dict = {1: 1400000, 2: 900000, 3: 900000}
_clustering_birch_threshold = 1

# feature ranking for subsampling
_features_ordered_by_importance = ['rank score', 'pn score', 'avg novor aa score', 'avg rank',
                                  'retention time', 'pn rank', 'sqs', 'fraction novor parent len',
                                  'fraction pn parent len', 'len', 'is longest consensus', 'is top rank consensus',
                                  '0.5', '0.4', '0.6', '0.3', '0.7', '0.2']

# report
_reported_df_cols = ['seq', 'probability', 'ref match',
                    'is novor seq', 'is peaks seq', 'is pn seq',
                    '0.2', '0.3', '0.4', '0.5', '0.6', '0.7',
                    'avg novor aa score', 'rank score', 'pn score',
                    'avg rank', 'peaks rank', 'pn rank',
                    'len', 'fraction novor parent len', 'fraction peaks parent len', 'fraction pn parent len',
                    'is longest consensus', 'is top rank consensus', 'is top rank single alg',
                    'sqs', 'retention time']