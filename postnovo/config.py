''' Variables used across project '''

from os.path import join, dirname, realpath
from itertools import product
from collections import OrderedDict


# standard constants
proton_mass = 1.007276
seconds_in_min = 60
# ms_aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
unicode_decimal_A = 65

# user args
# run level settings: predict (default), train, test, optimize
run_type = ['predict']
verbose = [True]
novor_files = []
novor_tols = []
peaks_files = []
peaks_tols = []
pn_files = []
pn_tols = []
min_prob = [0.5]
min_len = [6]
ref_file = [None]
cores = [1]

# directories
postnovo_parent_dir = dirname(dirname(realpath(__file__)))
test_dir = join(postnovo_parent_dir, 'test')
training_dir = join(postnovo_parent_dir, 'training')
userfiles_dir = join(postnovo_parent_dir, 'userfiles')
output_dir = join(postnovo_parent_dir, 'output')

# program constraints
accepted_algs = ['novor', 'peaks', 'pn']
possible_alg_combos = []
for numerical_alg_combo in list(product((0, 1), repeat = len(accepted_algs)))[1:]:
    possible_alg_combos.append(tuple([alg for i, alg in enumerate(accepted_algs) if numerical_alg_combo[i]]))
seqs_reported_per_alg_dict = {'novor': 1, 'peaks': 20, 'pn': 20}
accepted_mass_tols = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']

# global info from user input
alg_list = []
alg_combo_list = []
## example
## alg_combo_list = [('novor', 'peaks'), ('novor', 'pn'), ('peaks', 'pn'), ('novor', 'peaks', 'pn')]
is_alg_col_names = []
is_alg_col_multiindex_keys = []
tol_list = []
alg_tols_dict = OrderedDict()
## example
## alg_tols_dict = odict('novor': odict('0.4': 'proteome-0.4.novor.csv', '0.5': 'proteome-0.5.novor.csv'),
##                     'pn': odict('0.4': 'proteome-0.4.mgf.out', '0.5': 'proteome-0.5.mgf.out'))
tol_alg_dict = OrderedDict()
## example
## tol_alg_dict = odict('0.4': ['novor', 'pn'], '0.5': ['novor', 'pn'])
tol_basenames_dict = OrderedDict()
## example
## tol_basenames_dict = odict('0.4': ['proteome-0.4.novor.csv', 'proteome-0.4.mgf.out'],
##                              '0.5': ['proteome-0.5.novor.csv', 'proteome-0.5.mgf.out'])

# de novo output characteristics
novor_dropped_chars = {ord(char): None for char in
                       ''.join([str(i) for i in range(10)] + ['(', ')'])}
pn_dropped_chars = {ord(char): None for char in
                       ''.join([str(i) for i in range(10)] + ['+', '-', '.'])}
precursor_mass_tol = [4.0]

# training parameters
train_consensus_len = 6
rf_n_estimators = 150
rf_default_params = {('novor',): {'max_depth': 19, 'max_features': None},
                     ('pn',): {'max_depth': 14, 'max_features': None},
                     ('novor', 'pn'): {'max_depth': 19, 'max_features': None}}

# feature selection from input data
prediction_dict_source_cols = {'novor': ['retention time', 'measured mass', 'seq', 'aa score', 'avg aa score', 'encoded seq'],
                               'peaks': [],
                               'pn': ['measured mass', 'seq', 'rank score', 'pn score', 'sqs', 'encoded seq']}
single_alg_prediction_dict_cols = {'general': ['scan', 'measured mass', 'is top rank single alg', 'seq', 'len', 'avg rank'],
                                   'novor': ['retention time', 'is novor seq', 'avg novor aa score'],
                                   'peaks': [],
                                   'pn': ['is pn seq', 'rank score', 'pn score', 'pn rank', 'sqs']}
consensus_prediction_dict_cols = {'general': ['scan', 'measured mass', 'seq', 'len', 'avg rank', 'is longest consensus', 'is top rank consensus'],
                                  'novor': ['retention time', 'is novor seq', 'fraction novor parent len', 'avg novor aa score'],
                                  'peaks': [],
                                  'pn': ['is pn seq', 'fraction pn parent len', 'rank score', 'pn score', 'pn rank', 'sqs']}

# parameters for subsampling training data
subsample_size = 20000
subsample_accuracy_divisor = 10
subsample_accuracy_distribution_lower_bound = 0
subsample_accuracy_distribution_upper_bound = 3
subsample_accuracy_distribution_sigma = 0.9
subsample_accuracy_distribution_mu_location = 0.5
clustering_min_retained_features = 2
clustering_feature_retention_factor_dict = {1: 1400000, 2: 900000, 3: 900000}
clustering_birch_threshold = 1
# feature ranking for subsampling
features_ordered_by_importance = ['rank score', 'pn score', 'avg novor aa score', 'avg rank',
                                  'retention time', 'pn rank', 'sqs',
                                  '0.2 match', '0.7 match', '0.3 match', '0.4 match', '0.5 match', '0.6 match',
                                  'fraction novor parent len', 'fraction pn parent len',
                                  'len', 'is longest consensus', 'is top rank consensus',
                                  '0.5', '0.4', '0.6', '0.3', '0.7', '0.2']

# report
reported_df_cols = ['seq', 'probability', 'ref match',
                    'is novor seq', 'is peaks seq', 'is pn seq',
                    '0.2', '0.3', '0.4', '0.5', '0.6', '0.7',
                    '0.2 match', '0.3 match', '0.4 match', '0.5 match', '0.6 match', '0.7 match',
                    'precursor seq agreement', 'precursor seq count',
                    'avg novor aa score', 'rank score', 'pn score',
                    'avg rank', 'peaks rank', 'pn rank',
                    'len', 'fraction novor parent len', 'fraction peaks parent len', 'fraction pn parent len',
                    'is longest consensus', 'is top rank consensus', 'is top rank single alg',
                    'sqs', 'retention time']