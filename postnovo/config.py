''' Variables used across project '''

import re
import os

from itertools import product
from collections import OrderedDict
from functools import partial


# websites
forest_dict_url = ''

# standard constants
proton_mass = 1.007276
seconds_in_min = 60
# ms_aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
unicode_decimal_A = 65

# aa substitutions
# M+15.9949 Da, C+57.0215 Da
mono_di_isobaric_subs = {
    'N': 'N_GG', 'GG': 'N_GG',
    'Q': 'Q_AG', 'AG': 'Q_AG', 'GA': 'Q_AG'
    }
di_isobaric_subs = {
    'AD': 'AD_EG', 'DA': 'AD_EG', 'EG': 'AD_EG', 'GE': 'AD_EG',
    'AN': 'AN_GQ', 'NA': 'AN_GQ', 'GQ': 'AN_GQ', 'QG': 'AN_GQ',
    'AS': 'AS_GT', 'SA': 'AS_GT', 'GT': 'AS_GT', 'TG': 'AS_GT',
    'AV': 'AV_GL', 'VA': 'AV_GL', 'GL': 'AV_GL', 'LG': 'AV_GL',
    'AY': 'AY_FS', 'YA': 'AY_FS', 'FS': 'AY_FS', 'SF': 'AY_FS',
    'CT': 'CT_MN', 'TC': 'CT_MN', 'MN': 'CT_MN', 'NM': 'CT_MN',
    'DL': 'DL_EV', 'LD': 'DL_EV', 'EV': 'DL_EV', 'VE': 'DL_EV',
    'DQ': 'DQ_EN', 'QD': 'DQ_EN', 'EN': 'DQ_EN', 'NE': 'DQ_EN',
    'DT': 'DT_ES', 'TD': 'DT_ES', 'ES': 'DT_ES', 'SE': 'DT_ES',
    'LN': 'LN_QV', 'NL': 'LN_QV', 'QV': 'LN_QV', 'VQ': 'LN_QV',
    'LS': 'LS_TV', 'SL': 'LS_TV', 'TV': 'LS_TV', 'VT': 'LS_TV',
    'NT': 'NT_QS', 'TN': 'NT_QS', 'QS': 'NT_QS', 'SQ': 'NT_QS'    
    }
mono_di_near_isobaric_subs = {
    'R': 'R_GV', 'GV': 'R_GV', 'VG': 'R_GV'
    }
di_near_isobaric_subs = {
    'CL': 'CL_SW', 'LC': 'CL_SW', 'SW': 'CL_SW', 'WS': 'CL_SW',
    'ER': 'ER_VW', 'RE': 'ER_VW', 'VW': 'ER_VW', 'WV': 'ER_VW',
    'FQ': 'FQ_KM', 'QF': 'FQ_KM', 'KM': 'FQ_KM', 'MK': 'FQ_KM',
    'LM': 'LM_PY', 'ML': 'LM_PY', 'PY': 'LM_PY', 'YP': 'LM_PY'
    }

# user args
getopt_opts = ['help',
               'quiet',
               'train',
               'test',
               'optimize',
               'iodir=',
               'denovogui_path=',
               'denovogui_mgf_path=',
               #'frag_mass_tols=',
               'novor_files=',
               'peaks_files=',
               'pn_files=',
               'min_len=',
               'min_prob=',
               'db_search_ref_file=',
               'fasta_ref_file=',
               'cores=',
               'param_file=']

help_str = '\n'.join(['postnovo.py',
                      '--iodir <"/home/postnovo_io">',
                      '--train',
                      '--test',
                      '--optimize',
                      #'--frag_mass_tols <"0.3, 0.5">',
                      '--novor_files <"novor_output_0.3.novor.csv, novor_output_0.5.novor.csv">',
                      '--peaks_files <"peaks_output_0.3.csv, peaks_output_0.5.csv">',
                      '--pn_files <"pn_output_0.3.mgf.out, pn_output_0.5.mgf.out">',
                      '--denovogui_path <"/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar">',
                      '--denovogui_mgf_path <"/home/ms_files/spectra.mgf">',
                      '--db_search_ref_file <"proteome_discoverer_psm_table.csv">',
                      '--fasta_ref_file <"fasta_file.faa">',
                      '--cores <3>',
                      '--min_len <9>',
                      '--min_prob <0.75>',
                      '--quiet',
                      '--param_file <"param.json">'])

# program constraints
accepted_algs = ['novor', 'peaks', 'pn']
possible_alg_combos = []
for numerical_alg_combo in list(product((0, 1), repeat = len(accepted_algs)))[1:]:
    possible_alg_combos.append(tuple([alg for i, alg in enumerate(accepted_algs) if numerical_alg_combo[i]]))
seqs_reported_per_alg_dict = {'novor': 1, 'peaks': 20, 'pn': 20}
frag_mass_tols = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
accepted_mass_tols = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
fixed_mod = 'Oxidation of M'
variable_mod = 'Carbamidomethylation of C'
frag_method = 'CID'
frag_mass_analyzer = 'Trap'
train_consensus_len = 8

# run level settings: predict (default), train, test, optimize
verbose = [True]
run_type = ['predict']
#frag_mass_tols = []
novor_files = []
peaks_files = []
pn_files = []
min_prob = [0.5]
min_len = [train_consensus_len]
min_ref_match_len = [8]
db_search_ref_file = [None]
fasta_ref_file = [None]
cores = [1]

# directories
iodir = []
training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'training')

# global info from user input
alg_list = []
alg_combo_list = []
## example
## alg_combo_list = [('novor', 'peaks'), ('novor', 'pn'), ('peaks', 'pn'), ('novor', 'peaks', 'pn')]
is_alg_col_names = []
is_alg_col_multiindex_keys = []
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
novor_seq_sub_fn = partial(re.sub, pattern = '\([^)]+\)|[0-9]', repl = '')
pn_seq_sub_fn = partial(re.sub, pattern = '[0-9\+\-\.\^]', repl = '')
precursor_mass_tol = [4.0]

# training parameters
min_fdr = 0.05
rf_n_estimators = 150
rf_default_params = {('novor',): {'max_depth': 16, 'max_features': 'sqrt'},
                     ('pn',): {'max_depth': 12, 'max_features': 'sqrt'},
                     ('novor', 'pn'): {'max_depth': 16, 'max_features': 'sqrt'}}

# feature selection from input data
prediction_dict_source_cols = {'novor': ['retention time', 'measured mass', 'seq', 'aa score', 'avg aa score', 'encoded seq'],
                               'peaks': [],
                               'pn': ['measured mass', 'seq', 'rank score', 'pn score', 'sqs', 'encoded seq']}
single_alg_prediction_dict_cols = {'general': ['scan', 'measured mass', 'is top rank single alg', 'seq', 'len'],
                                   'novor': ['retention time', 'is novor seq', 'avg novor aa score',
                                             'mono-di isobaric sub score', 'di isobaric sub score',
                                             'mono-di near-isobaric sub score', 'di near-isobaric sub score'],
                                   'peaks': [],
                                   'pn': ['is pn seq', 'rank score', 'pn score', 'sqs']}
consensus_prediction_dict_cols = {'general': ['scan', 'measured mass', 'seq', 'len', 'avg rank', 'is longest consensus', 'is top rank consensus'],
                                  'novor': ['retention time', 'is novor seq', 'fraction novor parent len', 'avg novor aa score',
                                            'mono-di isobaric sub score', 'di isobaric sub score',
                                            'mono-di near-isobaric sub score', 'di near-isobaric sub score'],
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
                                  '0.2', '0.7', '0.3', '0.4', '0.5', '0.6']

# feature groups
#feature_groups = {'novor score': ['avg novor aa score'],
#                  'pn scores': ['rank score', 'pn score'],
#                  'seq len': ['len'],
#                  'retention time': ['retention time'],
#                  'mass tolerance': ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
#                  'consensus info': ['avg rank', 'pn rank', 'peaks rank',
#                                     'fraction novor parent len', 'fraction pn parent len',
#                                     'is longest consensus', 'is top rank consensus'],
#                  'mass tolerance match info': ['0.2 seq match', '0.3 seq match', '0.4 seq match',
#                                          '0.5 seq match', '0.6 seq match', '0.7 seq match'],
#                  'substitution info': ['mono-di isobaric sub score', 'di isobaric sub score',
#                                        'mono-di near-isobaric sub score', 'di near-isobaric sub score'],
#                  'inter-spectrum info': ['precursor seq agreement', 'precursor seq count']
#                  }

feature_groups = {'novor score': ['avg novor aa score'],
                  'pn scores': ['rank score', 'pn score'],
                  'other': ['retention time', 'len', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
                  'consensus info': ['avg rank', 'pn rank', 'peaks rank',
                                     'fraction novor parent len', 'fraction pn parent len',
                                     'is longest consensus', 'is top rank consensus'],
                  'mass tolerance agreement': ['0.2 seq match', '0.3 seq match', '0.4 seq match',
                                          '0.5 seq match', '0.6 seq match', '0.7 seq match'],
                  'substitution info': ['mono-di isobaric sub score', 'di isobaric sub score',
                                        'mono-di near-isobaric sub score', 'di near-isobaric sub score'],
                  'inter-spectrum agreement': ['precursor seq agreement', 'precursor seq count']
                  }

# report
reported_df_cols = ['seq', 'probability', 'ref match',
                    'scan has db search PSM', 'de novo seq matches db search seq', 'correct de novo seq not found in db search',
                    'is novor seq', 'is peaks seq', 'is pn seq',
                    '0.2', '0.3', '0.4', '0.5', '0.6', '0.7',
                    '0.2 seq match', '0.3 seq match', '0.4 seq match', '0.5 seq match', '0.6 seq match', '0.7 seq match',
                    'precursor seq agreement', 'precursor seq count',
                    'mono-di isobaric sub score', 'di isobaric sub score',
                    'mono-di near-isobaric sub score', 'di near-isobaric sub score',
                    'avg novor aa score', 'rank score', 'pn score',
                    'avg rank', 'peaks rank', 'pn rank',
                    'len', 'fraction novor parent len', 'fraction peaks parent len', 'fraction pn parent len',
                    'is longest consensus', 'is top rank consensus', 'is top rank single alg',
                    'sqs', 'retention time']