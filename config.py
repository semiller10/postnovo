''' Variables used across Postnovo project. '''

#Copyright 2018, Samuel E. Miller. All rights reserved.
#Postnovo is publicly available for non-commercial uses.
#Licensed under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
#See postnovo/LICENSE.txt.

from collections import OrderedDict
from functools import partial

import glob
import os
import re
import sys

from itertools import product

#STANDARD CONSTANTS
#These are not specific to Postnovo.
PROTON_MASS = 1.007276
standard_aa_mass_dict = OrderedDict([
    ('A', 71.03711), 
    ('R', 156.10111), 
    ('N', 114.04293), 
    ('D', 115.02694), 
    ('C', 103.00919), 
    ('E', 129.04259), 
    ('Q', 128.05858), 
    ('G', 57.02146), 
    ('H', 137.05891), 
    ('L', 113.08406), 
    ('K', 128.09496), 
    ('M', 131.04049), 
    ('F', 147.06841), 
    ('P', 97.05276), 
    ('S', 87.03203), 
    ('T', 101.04768), 
    ('W', 186.07931), 
    ('Y', 163.06333), 
    ('V', 99.06841)])
standard_plus_mod_mass_dict = standard_aa_mass_dict.copy()
##Reference: isobaric and near-isobaric mono- and dipeptides.
##mono_di_isobaric_subs = {
##    'N': 'N_GG', 
##    'GG': 'N_GG', 
##    'Q': 'Q_AG', 
##    'AG': 'Q_AG', 'GA': 'Q_AG'}
##di_isobaric_subs = {
##    'AD': 'AD_EG', 'DA': 'AD_EG', 
##    'EG': 'AD_EG', 'GE': 'AD_EG', 
##    'AN': 'AN_GQ', 'NA': 'AN_GQ', 
##    'GQ': 'AN_GQ', 'QG': 'AN_GQ', 
##    'AS': 'AS_GT', 'SA': 'AS_GT', 
##    'GT': 'AS_GT', 'TG': 'AS_GT', 
##    'AV': 'AV_GL', 'VA': 'AV_GL', 
##    'GL': 'AV_GL', 'LG': 'AV_GL', 
##    'AY': 'AY_FS', 'YA': 'AY_FS', 
##    'FS': 'AY_FS', 'SF': 'AY_FS', 
##    'C+57.021T': 'C+57.021T_M+15.995N', 'TC+57.021': 'C+57.021T_M+15.995N', 
##    'M+15.995N': 'C+57.021T_M+15.995N', 'NM+15.995': 'C+57.021T_M+15.995N', 
##    'DL': 'DL_EV', 'LD': 'DL_EV', 
##    'EV': 'DL_EV', 'VE': 'DL_EV', 
##    'DQ': 'DQ_EN', 'QD': 'DQ_EN', 
##    'EN': 'DQ_EN', 'NE': 'DQ_EN', 
##    'DT': 'DT_ES', 'TD': 'DT_ES', 
##    'ES': 'DT_ES', 'SE': 'DT_ES', 
##    'LN': 'LN_QV', 'NL': 'LN_QV', 
##    'QV': 'LN_QV', 'VQ': 'LN_QV', 
##    'LS': 'LS_TV', 'SL': 'LS_TV', 
##    'TV': 'LS_TV', 'VT': 'LS_TV', 
##    'NT': 'NT_QS', 'TN': 'NT_QS', 
##    'QS': 'NT_QS', 'SQ': 'NT_QS'}
##mono_di_near_isobaric_subs = {
##    'R': 'R_GV', 
##    'GV': 'R_GV', 'VG': 'R_GV'}
##di_near_isobaric_subs = {
##    'C+57.021L': 'C+57.021L_SW', 'LC+57.021': 'C+57.021L_SW', 
##    'SW': 'C+57.021L_SW', 'WS': 'C+57.021L_SW', 
##    'ER': 'ER_VW', 'RE': 'ER_VW', 
##    'VW': 'ER_VW', 'WV': 'ER_VW', 
##    'FQ': 'FQ_KM+15.995', 'QF': 'FQ_KM+15.995', 
##    'KM+15.995': 'FQ_KM+15.995', 'M+15.995K': 'FQ_KM+15.995', 
##    'LM+15.995': 'LM+15.995_PY', 'M+15.995L': 'LM+15.995_PY', 
##    'PY': 'LM+15.995_PY', 'YP': 'LM+15.995_PY'}

#PROGRAM CONSTRAINTS
#These are specific to the context of Postnovo.
postnovo_dir = os.path.dirname(os.path.realpath(__file__))
download_ids_tsv = os.path.join(postnovo_dir, 'download_ids.tsv')
if not os.path.exists(download_ids_tsv):
    with open(download_ids_tsv, 'w') as f:
        f.write('Download Filename\tGoogle Drive ID\tSize\n')
postnovo_top_train_dir = os.path.join(postnovo_dir, 'train')
if not os.path.isdir(postnovo_top_train_dir):
    os.mkdir(postnovo_top_train_dir)
postnovo_train_dir_dict = {
    'Low': os.path.join(postnovo_top_train_dir, 'low'), 
    'High': os.path.join(postnovo_top_train_dir, 'high')}
POSTNOVO_TRAIN_RECORD_FILENAME = 'train_record.tsv'
BINNED_SCORES_FILENAME = 'predictions_binned_by_score.tsv'
#DeNovoGUI and DeepNovo are stored within subdirectories of the Postnovo directory.
denovogui_version = '1.16.2'
denovogui_dir = os.path.join(postnovo_dir, 'DeNovoGUI-' + denovogui_version)
denovogui_jar_fp = os.path.join(denovogui_dir, 'DeNovoGUI-' + denovogui_version + '.jar')
deepnovo_dir = os.path.join(postnovo_dir, 'deepnovo')
deepnovo_program_basenames = [
    'deepnovo_config_template.py', 
    'deepnovo_cython_modules.pyx', 
    'deepnovo_cython_setup.py', 
    'deepnovo_main.py', 
    'deepnovo_main_modules.py', 
    'deepnovo_misc.py', 
    'deepnovo_model.py', 
    'deepnovo_model_decoding.py', 
    'deepnovo_model_training.py', 
    'deepnovo_utils.py', 
    'deepnovo_worker_db.py', 
    'deepnovo_worker_io.py', 
    'deepnovo_worker_test.py', 
    'knapsack.npy', 
    'LICENSE', 
    'README.md']
deepnovo_program_fps = [
    os.path.join(deepnovo_dir, deepnovo_program_basename) 
    for deepnovo_program_basename in deepnovo_program_basenames]

GOOGLE_DRIVE_DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'
CURRENT_DOWNLOAD_GOOGLE_DRIVE_ID = '1UOV4RbF8Jl3wfslVV__ZnTRkKNHX4y5r'

default_fixed_mods = ['Carbamidomethylation of C']
default_variable_mods = ['Oxidation of M']

#Postnovo functions and user output use the following symbols for modified amino acids.
postnovo_mod_standard_aa_dict = OrderedDict([('C+57.021', 'C'), ('M+15.995', 'M')])
postnovo_mod_mass_dict = OrderedDict([
    ('C+57.021', 57.02146 + standard_aa_mass_dict['C']), 
    ('M+15.995', 15.99492 + standard_aa_mass_dict['M'])])
#DeNovoGUI and Postnovo user input use the following modification symbols.
postnovo_denovogui_mod_dict = OrderedDict([
    ('C+57.021', 'Carbamidomethylation of C'), 
    ('M+15.995', 'Oxidation of M')])
#Map DeNovoGUI (Postnovo user input) symbols to Postnovo output symbols.
denovogui_postnovo_mod_dict = OrderedDict(
    zip(postnovo_denovogui_mod_dict.values(), postnovo_denovogui_mod_dict.keys()))
#Map modification symbols in Novor output sequences to Postnovo symbols: 
#this depends on the modifications used and the order of the modifications.
novor_postnovo_mod_dict = OrderedDict()
#Map modification symbols in PepNovo+ output sequences to Postnovo symbols: 
#PepNovo+ modification codes include mass differences rounded to one digit.
pn_postnovo_mod_dict = OrderedDict()
#DeepNovo uses and outputs the following modification symbols.
postnovo_deepnovo_config_mod_dict = OrderedDict([
    ('C+57.021', 'Cmod'), 
    ('M+15.995', 'Mmod')])
#Map DeepNovo output symbols to Postnovo symbols.
deepnovo_config_postnovo_mod_dict = OrderedDict(
    [(deepnovo_config_code, postnovo_code) 
    for postnovo_code, deepnovo_config_code in postnovo_deepnovo_config_mod_dict.items()])
#Peptide header lines in DeepNovo training MGF files can use the following modification symbols.
postnovo_deepnovo_training_mod_dict = OrderedDict([
    ('C+57.021', 'C(+57.02)'), 
    ('M+15.995', 'M(+15.99)')])
#Map DeepNovo output symbols to DeepNovo training MGF symbols.
deepnovo_config_deepnovo_training_mod_dict = OrderedDict(
    [(deepnovo_config_code, postnovo_deepnovo_training_mod_dict[postnovo_code]) 
     for postnovo_code, deepnovo_config_code in postnovo_deepnovo_config_mod_dict.items()])
#Map DeNovoGUI (Postnovo user input) symbols to DeepNovo output symbols.
denovogui_deepnovo_config_mod_dict = OrderedDict(
    [(denovogui_code, postnovo_deepnovo_config_mod_dict[postnovo_code]) 
     for postnovo_code, denovogui_code in postnovo_denovogui_mod_dict.items()])

#Assign an integer number to each possible standard or modified amino acid.
mod_code_dict = OrderedDict(zip(
    list(postnovo_mod_standard_aa_dict.keys()), 
    range(
        len(standard_aa_mass_dict), 
        len(standard_aa_mass_dict) + len(postnovo_mod_standard_aa_dict))))
code_mod_dict = OrderedDict([(code, mod) for mod, code in mod_code_dict.items()])
aa_code_dict = OrderedDict(zip(
    list(standard_aa_mass_dict.keys()) + list(postnovo_mod_standard_aa_dict.keys()), 
    range(len(standard_aa_mass_dict) + len(postnovo_mod_standard_aa_dict))))
code_aa_dict = OrderedDict([(code, aa) for aa, code in aa_code_dict.items()])
#Map the mod code to the code for the unmodified amino acid.
mod_code_standard_code_dict = OrderedDict(
    [(mod_code, aa_code_dict[postnovo_mod_standard_aa_dict[code_mod_dict[mod_code]]]) 
     for mod_code in code_mod_dict])

#All potential subsequence substitutions under consideration: assigned in module "userargs".
all_permuted_isobaric_peps_dict = OrderedDict()
all_permuted_near_isobaric_peps_dict = OrderedDict()

#MSGF+ is stored within a subdirectory of the Postnovo directory.
msgf_dir = os.path.join(postnovo_dir, 'MSGFPlus')
msgf_jar = os.path.join(msgf_dir, 'MSGFPlus.jar')
msgf_mods_fp = os.path.join(msgf_dir, 'MSGFPlus_Mods.txt')

possible_algs = ['Novor', 'PepNovo', 'DeepNovo']
#The maximum number of sequence predictions that should be reported per spectrum from each tool.
seqs_reported_per_alg_dict = dict([('Novor', 1), ('PepNovo', 20), ('DeepNovo', 20)])

#The default minimum length of sequences considered by and reported by Postnovo.
DEFAULT_MIN_LEN = 7

#The sets of fragment mass tolerances required for low- and high-resolution spectra.
frag_mass_tol_dict = {
    'Low': ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], 
    'High': ['0.01', '0.03', '0.05', '0.1', '0.5']}
default_frag_mass_tol_dict = {'Low': '0.5', 'High': '0.05'}

#A breakpoint between near-isobaric dipeptides is a mass difference of 0.01124 Da.
#The next mass difference is >0.015 Da.
NEAR_ISOBARIC_WINDOW = 0.012 #Da
#Maximum length of subsequences to consider for isobaric substitutions.
MAX_SUBSEQ_LEN = 2

#A minimum sequence length of 9 is required for a strong direct match to a reference proteome.
MIN_REF_MATCH_LEN = 9
##IN PROGRESS
##MIN_BLAST_QUERY_LEN = 9

#Algorithm evaluation scores are used to compare Postnovo results to individual algorithms.
alg_evaluation_score_name_dict = {
    'Novor': 'Novor Peptide Score', 
    'PepNovo': 'PepNovo Rank Score', 
    'DeepNovo': 'DeepNovo Average Amino Acid Score'}

#Parameters for training Postnovo random forest models.
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 16
RF_MAX_FEATURES = 'sqrt'

#Assume a 1% FDR to plot a database search PSM star in precision-recall/yield plots.
DEFAULT_PRECISION = 0.99
#Score bounds are used to scale scores to the colorbar in precision-recall/yield plots.
upper_score_bound_dict = {'Novor': 100, 'PepNovo': 15, 'DeepNovo': 1, 'Peaks': 100}
lower_score_bound_dict = {'Novor': 0, 'PepNovo': -10, 'DeepNovo': 0, 'Peaks': 0}
#Define the colorbar tick positions in precision-recall/yield plots.
colorbar_tick_dict = {
    'Novor': [0, 20, 40, 60, 80, 100], 
    'PepNovo': [-10, -5, 0, 5, 10, 15], 
    'DeepNovo': [0, 0.2, 0.4, 0.6, 0.8, 1], 
    'Peaks': [0, 20, 40, 60, 80, 100]}

#Postnovo score is compared to sequence correctness by binning and averaging predictions.
SCORE_BIN_COUNT = 100
SCORE_BIN_SIZE = 1 / SCORE_BIN_COUNT

#Precision thresholds to report recall and yield statistics in test mode with "test_plots" option.
reported_precision_thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]

#Global user variables are stored in a mutable object: 
#the dict is constructed in the config module.
globals = dict()
#Peptide M/Z, charge, and retention time data for each spectrum is found from the MGF input: 
#the dict is constructed in module "userargs".
mgf_info_dict = OrderedDict()

#Output features from each tool that are reported by the input module 
#(spectrum ID and seq rank are indices).
alg_cols_dict = {
    'Novor': [
        'Retention Time', 
        'M/Z', 
        'Charge', 
        'Sequence', 
        'Encoded Sequence', 
        'Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Amino Acid Scores', 
        'Novor Average Amino Acid Score'], 
    'PepNovo': [
        'Retention Time', 
        'M/Z', 
        'Charge', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Sequence', 
        'Encoded Sequence', 
        'Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    'DeepNovo': [
        'Retention Time', 
        'M/Z', 
        'Charge', 
        'Sequence', 
        'Encoded Sequence', 
        'Sequence Length', 
        'DeepNovo Amino Acid Scores', 
        'DeepNovo Average Amino Acid Score']}

#The feature set used in Postnovo models is given by an ID and can be set by the user.
#The ID is stored in config.globals['Feature Set ID'] 
#and the feature set in config.globals['Model Features Dict'].
model_features_dicts = []
#FEATURE SET 0
#This set uses features taken directly from mass spectral data and source de novo algorithm output.
#It does not use consensus sequences.
#Each fragment mass tolerance parameterization is used, but parameterizations are not compared.
model_features_dicts.append({
    ('Novor', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score'], 
    ('PepNovo', ): [
        'M/Z', 
        'Charge', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    ('DeepNovo', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'DeepNovo Average Amino Acid Score']})

#FEATURE SET 1
#This set adds potential substitutional error features to Feature Set 0.
#It does not use consensus sequences.
#Each fragment mass tolerance parameterization is used, but parameterizations are not compared.
model_features_dicts.append({
    ('Novor', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position'], 
    ('PepNovo', ): [
        'M/Z', 
        'Charge', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    ('DeepNovo', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position']})

#FEATURE SET 2
#This set adds consensus sequence models and features to Feature Set 1.
model_features_dicts.append({
    ('Novor', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position'], 
    ('PepNovo', ): [
        'M/Z', 
        'Charge', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    ('DeepNovo', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    ('Novor', 'PepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    ('Novor', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    ('PepNovo', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    ('Novor', 'PepNovo', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position']})

#FEATURE SET 3
#This set adds fragment mass tolerance agreement features to Feature Set 2.
#These are added in the userargs module after MS2 resolution is specified by the user.
#Therefore, this dict here appears identical to the dict for Feature Set 2.
model_features_dicts.append({
    ('Novor', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position'], 
    ('PepNovo', ): [
        'M/Z', 
        'Charge', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    ('DeepNovo', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    ('Novor', 'PepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    ('Novor', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    ('PepNovo', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    ('Novor', 'PepNovo', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position']})

#FEATURE SET 4
#This set adds features comparing sequences from different spectra but the same peptide species.
model_features_dicts.append({
    ('Novor', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence'], 
    ('PepNovo', ): [
        'M/Z', 
        'Charge', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence'], 
    ('DeepNovo', ): [
        'M/Z', 
        'Charge', 
        'Sequence Length', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence'], 
    ('Novor', 'PepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence'], 
    ('Novor', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence'], 
    ('PepNovo', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence'], 
    ('Novor', 'PepNovo', 'DeepNovo'): [
        'M/Z', 
        'Charge', 
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence', 
        'Novor Fraction Parent Sequence Length', 
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count', 
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position', 
        'PepNovo Source Sequence Rank', 
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Fraction Parent Sequence Length', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)', 
        'DeepNovo Source Sequence Rank', 
        'DeepNovo Fraction Parent Sequence Length', 
        'DeepNovo Source Average Amino Acid Score', 
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position', 
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence']})

#Feature groups are used in the production of feature importance bar charts in "train" mode.
feature_group_dict = {
    'MS Data': [
        'M/Z', 
        'Charge'], 
    'Seq Length': [
        'Sequence Length'], 
    'Fragment Mass Tolerances': [], 
    'Novor Metrics': [
        'De Novo Peptide Ion Mass', 
        'De Novo Peptide Ion Mass Error (ppm)', 
        'Novor Peptide Score', 
        'Novor Average Amino Acid Score', 
        'Novor Source Average Amino Acid Score'], 
    'PepNovo Metrics': [
        'PepNovo N-terminal Mass Gap', 
        'PepNovo C-terminal Mass Gap', 
        'PepNovo Rank Score', 
        'PepNovo Score', 
        'PepNovo Spectrum Quality Score (SQS)'], 
    'DeepNovo Metrics': [
        'DeepNovo Average Amino Acid Score', 
        'DeepNovo Source Average Amino Acid Score'], 
    'Novor Low-Scoring Subseqs': [
        'Novor Low-Scoring Dipeptide Count', 
        'Novor Low-Scoring Tripeptide Count'], 
    'DeepNovo Low-Scoring Subseqs': [
        'DeepNovo Low-Scoring Dipeptide Count', 
        'DeepNovo Low-Scoring Tripeptide Count'], 
    'Novor Isobaric Subseqs': [        
        'Novor Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Isobaric Dipeptide Substitution Score', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
        'Novor Near-Isobaric Dipeptide Substitution Score', 
        'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Isobaric Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'Novor Near-Isobaric Dipeptide Substitution Average Position'], 
    'DeepNovo Isobaric Subseqs': [
        'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Isobaric Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
        'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Isobaric Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
        'DeepNovo Near-Isobaric Dipeptide Substitution Average Position'], 
    'Consensus Sequence Type': [
        'Is Consensus Top-Ranked Sequence', 
        'Is Consensus Longest Sequence'], 
    'Source Rank': [
        'PepNovo Source Sequence Rank', 
        'DeepNovo Source Sequence Rank'], 
    'Fraction Source Length': [
        'Novor Fraction Parent Sequence Length', 
        'PepNovo Fraction Parent Sequence Length', 
        'DeepNovo Fraction Parent Sequence Length'], 
    'Fragment Mass Tolerance Matches': [], 
    'Spectrum Comparison': [
        'Clustered Spectra', 
        'Clustered Spectra with a Sequence Containing This Sequence', 
        'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
        'Clustered Spectra with a Sequence Contained in This Sequence', 
        'Clustered Spectra with a Consensus Sequence Contained in This Sequence']}
#Color-code the bars by group in the feature importance plots.
feature_group_color_dict = {
    'MS Data': 'sienna', 
    'Seq Length': 'chocolate', 
    'Fragment Mass Tolerances': 'sandybrown', 
    'Novor Metrics': 'darksalmon', 
    'PepNovo Metrics': 'tomato', 
    'DeepNovo Metrics': 'lightcoral', 
    'Novor Low-Scoring Subseqs': 'darkorange', 
    'DeepNovo Low-Scoring Subseqs': 'goldenrod', 
    'Novor Isobaric Subseqs': 'darkkhaki', 
    'DeepNovo Isobaric Subseqs': 'olive', 
    'Consensus Sequence Type': 'darkgreen', 
    'Source Rank': 'limegreen', 
    'Fraction Source Length': 'mediumaquamarine', 
    'Fragment Mass Tolerance Matches': 'darkslategrey', 
    'Spectrum Comparison': 'dodgerblue'}

#All potential columns in the reported prediction table are listed in order of occurrence.
#The potential index columns are 
#'Spectrum ID', 'Is Novor Sequence', 'Is PepNovo Sequence', 'Is DeepNovo Sequence', 
#'0.2', '0.3', '0.4', '0.5', '0.6', '0.7' OR '0.01', '0.03', '0.05', '0.1', '0.5'
reported_df_cols = [
    'Is Consensus Top-Ranked Sequence', 
    'Is Consensus Longest Sequence', 
    'Estimated Probability', 
    'Sequence', 
    'Sequence Length', 
    'Reference Sequence', 
    'Reference Sequence Match', 
    'Sequence Matches Database Search PSM', 
    'Exclusive Reference Fasta Match', 
    'Novor Source Sequence', 
    'Novor Fraction Parent Sequence Length', 
    'PepNovo Source Sequence', 
    'PepNovo Source Sequence Rank', 
    'PepNovo Fraction Parent Sequence Length', 
    'DeepNovo Source Sequence', 
    'DeepNovo Source Sequence Rank', 
    'DeepNovo Fraction Parent Sequence Length', 
    'Novor Peptide Score', 
    'Novor Average Amino Acid Score', 
    'Novor Amino Acid Scores', 
    'Novor Consensus Amino Acid Scores', 
    'De Novo Peptide Ion Mass', 
    'De Novo Peptide Ion Mass Error (ppm)', 
    'Novor Isobaric Dipeptide Substitution Score', 
    'Novor Isobaric Dipeptide Substitution Average Position', 
    'Novor Isobaric Mono-Dipeptide Substitution Score', 
    'Novor Isobaric Mono-Dipeptide Substitution Average Position', 
    'Novor Near-Isobaric Dipeptide Substitution Score', 
    'Novor Near-Isobaric Dipeptide Substitution Average Position', 
    'Novor Near-Isobaric Mono-Dipeptide Substitution Score', 
    'Novor Near-Isobaric Mono-Dipeptide Substitution Average Position', 
    'Novor Low-Scoring Dipeptide Count', 
    'Novor Low-Scoring Tripeptide Count', 
    'PepNovo N-terminal Mass Gap', 
    'PepNovo C-terminal Mass Gap', 
    'PepNovo Rank Score', 
    'PepNovo Score', 
    'PepNovo Spectrum Quality Score (SQS)', 
    'DeepNovo Average Amino Acid Score', 
    'DeepNovo Amino Acid Scores', 
    'DeepNovo Source Average Amino Acid Score', 
    'DeepNovo Consensus Amino Acid Scores', 
    'DeepNovo Isobaric Dipeptide Substitution Score', 
    'DeepNovo Isobaric Dipeptide Substitution Average Position', 
    'DeepNovo Isobaric Mono-Dipeptide Substitution Score', 
    'DeepNovo Isobaric Mono-Dipeptide Substitution Average Position', 
    'DeepNovo Near-Isobaric Dipeptide Substitution Score', 
    'DeepNovo Near-Isobaric Dipeptide Substitution Average Position', 
    'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Score', 
    'DeepNovo Near-Isobaric Mono-Dipeptide Substitution Average Position', 
    'DeepNovo Low-Scoring Dipeptide Count', 
    'DeepNovo Low-Scoring Tripeptide Count', 
    '0.01 Match Value', 
    '0.03 Match Value', 
    '0.05 Match Value', 
    '0.2 Match Value', 
    '0.3 Match Value', 
    '0.4 Match Value', 
    '0.5 Match Value', 
    '0.6 Match Value', 
    '0.7 Match Value', 
    'Spectrum Cluster ID', 
    'Clustered Spectra', 
    'Clustered Spectra with a Sequence Containing This Sequence', 
    'Clustered Spectra with a Consensus Sequence Containing This Sequence', 
    'Clustered Spectra with a Sequence Contained in This Sequence', 
    'Clustered Spectra with a Consensus Sequence Contained in This Sequence', 
    'M/Z', 
    'Charge']