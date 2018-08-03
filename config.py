''' Variables used across project '''

import re
import os

from itertools import product
from collections import OrderedDict
from functools import partial

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

#Global variables are stored in a mutable object.
globals = dict()

#Numeric representation of amino acids
dict([
    ('A', 0), 
    ('C', 1), 
    ('D', 2), 
    ('E', 3), 
    ('F', 4), 
    ('G', 5), 
    ('H', 6), 
    ('I', 7), 
    ('K', 8), 
    ('L', 9), 
    ('M', 10), 
    ('N', 11), 
    ('P', 12), 
    ('Q', 13), 
    ('R', 14), 
    ('S', 15), 
    ('T', 16), 
    ('V', 17), 
    ('W', 18), 
    ('Y', 19), 
    ('C+57.021', 20), 
    ('M+15.995', 21)
])
dict([
    ('Carbamidomethylation of C', 'C+57.021'), 
    ('Oxidation of M', 'M+15.995')
])

#Standard constants
proton_mass = 1.007276
seconds_in_min = 60
unicode_decimal_A = 65

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
#di_isobaric_subs = {
#    'AD': 'AD_EG', 'DA': 'AD_EG', 'EG': 'AD_EG', 'GE': 'AD_EG', 
#    'AN': 'AN_GQ', 'NA': 'AN_GQ', 'GQ': 'AN_GQ', 'QG': 'AN_GQ', 
#    'AS': 'AS_GT', 'SA': 'AS_GT', 'GT': 'AS_GT', 'TG': 'AS_GT', 
#    'AV': 'AV_GL', 'VA': 'AV_GL', 'GL': 'AV_GL', 'LG': 'AV_GL', 
#    'AY': 'AY_FS', 'YA': 'AY_FS', 'FS': 'AY_FS', 'SF': 'AY_FS', 
#    'C+57.021T': 'C+57.021T_M+15.995N', 
#    'TC+57.021': 'C+57.021T_M+15.995N', 
#    'M+15.995N': 'C+57.021T_M+15.995N', 
#    'NM+15.995': 'C+57.021T_M+15.995N', 
#    'DL': 'DL_EV', 'LD': 'DL_EV', 'EV': 'DL_EV', 'VE': 'DL_EV', 
#    'DQ': 'DQ_EN', 'QD': 'DQ_EN', 'EN': 'DQ_EN', 'NE': 'DQ_EN', 
#    'DT': 'DT_ES', 'TD': 'DT_ES', 'ES': 'DT_ES', 'SE': 'DT_ES', 
#    'LN': 'LN_QV', 'NL': 'LN_QV', 'QV': 'LN_QV', 'VQ': 'LN_QV', 
#    'LS': 'LS_TV', 'SL': 'LS_TV', 'TV': 'LS_TV', 'VT': 'LS_TV', 
#    'NT': 'NT_QS', 'TN': 'NT_QS', 'QS': 'NT_QS', 'SQ': 'NT_QS' 
#}
mono_di_near_isobaric_subs = {
    'R': 'R_GV', 'GV': 'R_GV', 'VG': 'R_GV'
}
di_near_isobaric_subs = {
    'CL': 'CL_SW', 'LC': 'CL_SW', 'SW': 'CL_SW', 'WS': 'CL_SW', 
    'ER': 'ER_VW', 'RE': 'ER_VW', 'VW': 'ER_VW', 'WV': 'ER_VW', 
    'FQ': 'FQ_KM', 'QF': 'FQ_KM', 'KM': 'FQ_KM', 'MK': 'FQ_KM', 
    'LM': 'LM_PY', 'ML': 'LM_PY', 'PY': 'LM_PY', 'YP': 'LM_PY'
}
#di_near_isobaric_subs = {
#    'C+57.021L': 'C+57.021L_SW', 
#    'LC+57.021': 'C+57.021L_SW', 
#    'SW': 'C+57.021L_SW', 
#    'WS': 'C+57.021L_SW', 
#    'ER': 'ER_VW', 'RE': 'ER_VW', 'VW': 'ER_VW', 'WV': 'ER_VW', 
#    'FQ': 'FQ_KM+15.995', 
#    'QF': 'FQ_KM+15.995', 
#    'KM+15.995': 'FQ_KM+15.995', 
#    'M+15.995K': 'FQ_KM+15.995', 
#    'LM+15.995': 'LM+15.995_PY', 
#    'M+15.995L': 'LM+15.995_PY', 
#    'PY': 'LM+15.995_PY', 
#    'YP': 'LM+15.995_PY'
#}

#Program constraints
seqs_reported_per_alg_dict = {'novor': 1, 'pn': 20, 'deepnovo': 20}
low_res_mass_tols = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
hi_res_mass_tols = ['0.01', '0.03', '0.05', '0.1', '0.5']
default_fixed_mods = ['Carbamidomethylation of C']
default_variable_mods = ['Oxidation of M']
train_consensus_len = 7
min_ref_match_len = 7
min_blast_query_len = 9

#PICK UP HERE, removing all instances of db_name_list
db_name_list = []
psm_fp_list = []
db_fp_list = []
## example
## is_alg_col_names = ['is novor seq', 'is pn seq', 'is deepnovo seq']
is_alg_col_names = []
is_alg_col_multiindex_keys = []
precursor_mass_tol = [10.0]

# de novo output characteristics
mod_chars = ['.', '|', '^', '+', '-'] + [str(i) for i in range(10)]

# training parameters
max_fdr = 0.01
rf_n_estimators = 150
rf_default_params = {
    ('novor',): {'max_depth': 16, 'max_features': 'sqrt'},
    ('pn',): {'max_depth': 12, 'max_features': 'sqrt'},
    ('deepnovo',): {'max_depth': 16, 'max_features': 'sqrt'},
    ('novor', 'pn'): {'max_depth': 16, 'max_features': 'sqrt'},
    ('novor', 'deepnovo'): {'max_depth': 16, 'max_features': 'sqrt'},
    ('pn', 'deepnovo'): {'max_depth': 16, 'max_features': 'sqrt'},
    ('novor', 'pn', 'deepnovo'): {'max_depth': 16, 'max_features': 'sqrt'}
    }

# feature selection from input data
prediction_dict_source_cols = {
    'novor': ['retention time', 'measured mass', 'seq', 'aa score', 'avg aa score', 'encoded seq'],
    'pn': ['retention time', 'measured mass', 'seq', 'rank score', 'pn score', 'sqs', 'encoded seq'],
    'deepnovo': [
        'retention time',
        'measured mass',
        'seq',
        'aa score',
        'avg aa score',
        'encoded seq'
        ]
    }
single_alg_prediction_dict_cols = {
    'general': [
        'scan',
        'measured mass',
        'is top rank single alg',
        'seq',
        'len'
        ],
    'novor': [
        'retention time',
        'is novor seq',
        'avg novor aa score',
        'novor low-scoring dipeptide count',
        'novor low-scoring tripeptide count',
        'novor mono-di isobaric sub score',
        'novor di isobaric sub score',
        'novor mono-di near-isobaric sub score',
        'novor di near-isobaric sub score'
        ],
    'pn': [
        'is pn seq',
        'rank score',
        'pn score',
        'sqs'
        ],
    'deepnovo': [
        'is deepnovo seq',
        'avg deepnovo aa score',
        'deepnovo low-scoring dipeptide count',
        'deepnovo low-scoring tripeptide count',
        'deepnovo mono-di isobaric sub score',
        'deepnovo di isobaric sub score',
        'deepnovo mono-di near-isobaric sub score',
        'deepnovo di near-isobaric sub score'
        ]
    }
consensus_prediction_dict_cols = {
    'general': [
        'scan',
        'measured mass',
        'seq',
        'len',
        'avg rank',
        'is longest consensus',
        'is top rank consensus'
        ],
    'novor': [
        'retention time',
        'is novor seq',
        'fraction novor parent len',
        'avg novor aa score',
        'novor low-scoring dipeptide count',
        'novor low-scoring tripeptide count',
        'novor mono-di isobaric sub score',
        'novor di isobaric sub score',
        'novor mono-di near-isobaric sub score',
        'novor di near-isobaric sub score'
        ],
    'pn': [
        'is pn seq',
        'fraction pn parent len',
        'rank score',
        'pn score',
        'pn rank',
        'sqs'
        ],
    'deepnovo': [
        'is deepnovo seq',
        'fraction deepnovo parent len',
        'deepnovo rank',
        'avg deepnovo aa score',
        'deepnovo low-scoring dipeptide count',
        'deepnovo low-scoring tripeptide count',
        'deepnovo mono-di isobaric sub score',
        'deepnovo di isobaric sub score',
        'deepnovo mono-di near-isobaric sub score',
        'deepnovo di near-isobaric sub score'
        ]
    }

# feature groups
feature_groups = OrderedDict([
    ('novor score', [
        'avg novor aa score'
        ]),
    ('pn scores', [
        'rank score', 
        'pn score', 
        'sqs'
        ]), 
    ('deepnovo score', [
        'avg deepnovo aa score'
        ]), 
    ('other', [
        'measured mass', 
        'retention time', 
        'is novor seq', 
        'is pn seq', 
        'is deepnovo seq', 
        'len', 
        '0.005', 
        '0.01', 
        '0.03', 
        '0.05', 
        '0.1', 
        '0.2', 
        '0.3', 
        '0.4', 
        '0.5', 
        '0.6', 
        '0.7'
        ]),
    ('consensus info', [
        'avg rank', 
        'pn rank', 
        'deepnovo rank', 
        'fraction novor parent len', 
        'fraction pn parent len', 
        'fraction deepnovo parent len', 
        'is longest consensus', 
        'is top rank consensus'
        ]),
    ('mass tolerance agreement', [
        '0.005 seq match', 
        '0.01 seq match', 
        '0.03 seq match', 
        '0.05 seq match', 
        '0.1 seq match', 
        '0.2 seq match', 
        '0.3 seq match', 
        '0.4 seq match', 
        '0.5 seq match', 
        '0.6 seq match', 
        '0.7 seq match'
        ]),
    ('substitution info', [
        'novor mono-di isobaric sub score', 
        'novor di isobaric sub score', 
        'novor mono-di near-isobaric sub score', 
        'novor di near-isobaric sub score', 
        'novor low-scoring dipeptide count', 
        'novor low-scoring tripeptide count', 
        'deepnovo mono-di isobaric sub score', 
        'deepnovo di isobaric sub score', 
        'deepnovo mono-di near-isobaric sub score', 
        'deepnovo di near-isobaric sub score', 
        'deepnovo low-scoring dipeptide count', 
        'deepnovo low-scoring tripeptide count'
        ]),
    ('inter-spectrum agreement', [
        'precursor seq agreement', 
        'precursor seq count'
        ])
    ])

# report
reported_df_cols = [
    'seq', 
    'probability', 
    'ref match', 
    'scan has db search PSM', 
    'de novo seq matches db search seq', 
    'correct de novo seq not found in db search', 
    'is novor seq', 
    'is pn seq', 
    'is deepnovo seq', 
    '0.005', 
    '0.01', 
    '0.03', 
    '0.05', 
    '0.1', 
    '0.2', 
    '0.3', 
    '0.4', 
    '0.5', 
    '0.6', 
    '0.7', 
    '0.005 seq match', 
    '0.01 seq match', 
    '0.03 seq match', 
    '0.05 seq match', 
    '0.1 seq match', 
    '0.2 seq match', 
    '0.3 seq match', 
    '0.4 seq match', 
    '0.5 seq match', 
    '0.6 seq match', 
    '0.7 seq match', 
    'precursor seq agreement', 
    'precursor seq count', 
    'novor mono-di isobaric sub score', 
    'novor di isobaric sub score', 
    'novor mono-di near-isobaric sub score', 
    'novor di near-isobaric sub score', 
    'novor low-scoring dipeptide count', 
    'novor low-scoring tripeptide count', 
    'deepnovo mono-di isobaric sub score', 
    'deepnovo di isobaric sub score', 
    'deepnovo mono-di near-isobaric sub score', 
    'deepnovo di near-isobaric sub score', 
    'deepnovo low-scoring dipeptide count', 
    'deepnovo low-scoring tripeptide count', 
    'avg novor aa score', 
    'rank score', 
    'pn score', 
    'avg deepnovo aa score', 
    'avg rank', 
    'pn rank', 
    'deepnovo rank', 
    'len', 
    'fraction novor parent len', 
    'fraction pn parent len', 
    'fraction deepnovo parent len', 
    'is longest consensus', 
    'is top rank consensus', 
    'is top rank single alg', 
    'sqs', 
    'retention time', 
    'measured mass', 
    'mass error'
    ]