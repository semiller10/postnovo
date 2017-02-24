''' Variables used across project '''

import numpy as np

from os.path import join, dirname, realpath

postnovo_par_dir = dirname(dirname(realpath(__file__)))
test_dir = join(postnovo_par_dir, 'test')
training_dir = join(postnovo_par_dir, 'training')
user_files_dir = join(postnovo_par_dir, 'userfiles')
accepted_algs = ['novor', 'peaks', 'pn']
seqs_reported_per_alg_dict = {'novor': 1, 'peaks': 20, 'pn': 20}
accepted_mass_tols = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
proton_mass = 1.007276
seconds_in_min = 60
novor_dropped_chars = {ord(char): None for char in
                       ''.join([str(i) for i in range(10)] + ['(', ')'])}
pn_dropped_chars = {ord(char): None for char in
                       ''.join([str(i) for i in range(10)] + ['+', '-', '.'])}
#aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
#aa_translation = dict(zip(map(ord, aa_list), range(19)))
unicode_decimal_A = 65
train_consensus_len = 6

prediction_dict_source_cols = {'novor': ['retention time', 'seq', 'aa score', 'avg aa score', 'encoded seq'],
                               'peaks': [],
                               'pn': ['seq', 'rank score', 'pn score', 'sqs', 'encoded seq']}
single_alg_prediction_dict_cols = {'general': ['scan', 'is top rank single alg', 'seq', 'len', 'avg rank'],
                                   'novor': ['retention time', 'is novor seq', 'avg novor aa score'],
                                   'peaks': [],
                                   'pn': ['is pn seq', 'rank score', 'pn score', 'pn rank', 'sqs']}
consensus_prediction_dict_cols = {'general': ['scan', 'seq', 'len', 'avg rank', 'is longest consensus', 'is top rank consensus'],
                                   'novor': ['retention time', 'is novor seq', 'fraction novor parent len', 'avg novor aa score'],
                                   'peaks': [],
                                   'pn': ['is pn seq', 'fraction pn parent len', 'rank score', 'pn score', 'pn rank', 'sqs']}

n_estimators = 150

subsample_size = 10000
accuracy_divisor = 10
accuracy_distribution_lower_bound = 0
accuracy_distribution_upper_bound = 3
accuracy_distribution_mu_location = 0.5
accuracy_distribution_sigma = 0.9
min_retained_features_target = 2
features_ordered_by_importance = ['rank score', 'pn score', 'avg novor aa score', 'avg rank',
                                                 'retention time', 'pn rank', 'sqs', 'fraction novor parent len',
                                                 'fraction pn parent len', 'len', 'is longest consensus', 'is top rank consensus',
                                                 '0.5', '0.4', '0.6', '0.3', '0.7', '0.2']
clustering_feature_retention_factor_dict = {1: 1400000, 2: 900000, 3: 900000}
birch_threshold = 1

seq_matching_multiprocessing_splits = 300