''' Process user input and run DeNovoGUI and DeepNovo as needed. '''

import config
import utils

import argparse
import csv
import glob
import numpy as np
import os
import pandas as pd
import requests
import shutil
import subprocess
import sys
import time
import zipfile

from collections import OrderedDict
from config import postnovo_deepnovo_training_mod_dict
from io import StringIO
from itertools import combinations, product
from math import ceil
from multiprocessing import cpu_count
    
def setup(test_argv=None):
    '''
    Entry point to userargs module from main module.

    Parameters
    ----------
    test_argv : list
        List of command line arguments for testing specified in main module.

    Return
    ------
    None
    '''
    
    #Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Postnovo post-processes peptide de novo sequences to improve their accuracy.')
    subparsers = parser.add_subparsers(dest='subparser_name')

    setup_parser = subparsers.add_parser(
        'setup', 
        help='Sets up the default Postnovo and DeepNovo models.')
    setup_parser.add_argument(
        '--check_updates', 
        action='store_true', 
        default=False, 
        help='Check for updates to downloadable files.')
    setup_parser.add_argument(
        '--denovogui', 
        action='store_true', 
        default=False, 
        help='Download DeNovoGUI.')
    setup_parser.add_argument(
        '--msgf', 
        action='store_true', 
        default=False, 
        help='Download MSGF+.')
    setup_parser.add_argument(
        '--postnovo_low', 
        action='store_true', 
        default=False, 
        help='Download the Postnovo model for low-resolution fragmentation spectra.')
    setup_parser.add_argument(
        '--postnovo_high', 
        action='store_true', 
        default=False, 
        help='Download the Postnovo model for high-resolution fragmentation spectra.')
    setup_parser.add_argument(
        '--deepnovo_low', 
        action='store_true', 
        default=False, 
        help=(
            'Download the DeepNovo model for low-resolution fragmentation spectra. '
            'Warning: this option overwrites '
            'the relevant fragment mass tolerance directories in postnovo/deepnovo.'))
    setup_parser.add_argument(
        '--deepnovo_high', 
        action='store_true', 
        default=False, 
        help=(
            'Download the DeepNovo model for high-resolution fragmentation spectra.'
            'Warning: this option overwrites '
            'the relevant fragment mass tolerance directories in postnovo/deepnovo.'))
    setup_parser.add_argument(
        '--container', 
        help=(
            'Path to Singularity/Docker TensorFlow container image. '
            'Required for "deepnovo_low" and "deepnovo_high" options.'))
    setup_parser.add_argument(
        '--bind_point', 
        help=(
            'Directory within the scope of the singularity container '
            'used to bind a Postnovo/DeepNovo directory outside the scope of the container. '
            'By default, bind points are subdirectories created in the directory '
            'containing the Singularity container and given the name, '
            '"DeepNovo.<frag_resolution>.<frag_mass_tol>".'))
    setup_parser.add_argument(
        '--postnovo_low_spectra', 
        action='store_true', 
        default=False, 
        help='Download MGF files of spectra used to train the Postnovo low-resolution model.')
    setup_parser.add_argument(
        '--postnovo_high_spectra', 
        action='store_true', 
        default=False, 
        help='Download MGF files of spectra used to train the Postnovo high-resolution model.')
    setup_parser.add_argument(
        '--deepnovo_low_spectra', 
        action='store_true', 
        default=False, 
        help='Download MGF files of spectra used to train the DeepNovo low-resolution model.')
    setup_parser.add_argument(
        '--deepnovo_high_spectra', 
        action='store_true', 
        default=False, 
        help='Download MGF files of spectra used to train the DeepNovo high-resolution model.')

    #Subcommand: Reformat mgf files for Postnovo compatability.
    format_mgf_parser = subparsers.add_parser(
        'format_mgf', 
        help=(
            'Reformats MGF input file(s) '
            'to be compatible with de novo sequencing tools and Postnovo. '
            'If multiple MGF input files are specified, '
            'then input spectra will be concatenated into a single file.'))
    format_mgf_parser.add_argument(
        '--mgfs', 
        nargs='+', 
        help=(
            'Path(s) to MGF file(s). '
            'A specific format is required, as can be generated with the following command: '
            'msconvert YOUR_FILE.raw --mgf '
            '--filter "titleMaker Run: <RunID>, Index: <Index>, Scan: <ScanNumber>"'))
    format_mgf_parser.add_argument(
        '--out', 
        help='Path to reformatted MGF output file.')
    format_mgf_parser.add_argument(
        '--subsample', 
        type=utils.check_positive_nonzero_int, 
        help='Number of spectra to randomly subsample from all MGF input files.')
    format_mgf_parser.add_argument(
        '--remove_low_mass', 
        type=float, 
        default=200, 
        help=(
            'Remove spectra with a precursor mass below the specified cutoff. ' 
            'Novor has problems with low masses. '
            'Specify a value <= 0 to retain spectra with any mass.'))
    format_mgf_parser.add_argument(
        '--db_searches', 
        nargs='+', 
        help=(
            'Path(s) to table(s) of selected database search PSMs, '
            'each corresponding to input MGF(s) (same order required). '
            'Confident peptide sequences are placed '
            'in each spectrum header for DeepNovo training. '
            'MSGF+ TSV output is accepted. '
            'The table must contain a column of scans labeled \'ScanNum\' '
            'and a column of peptide sequences labeled \'Peptide\'. '
            'Postnovo recognizes the following PTM symbols: "C+57.021" and "M+15.995". '
            'If screening to a specified FDR with decoy hits (screen_fdr option), '
            'the table must also contain '
            '\'ScanNum\', \'Protein\' and \'SpecEValue\' MSGF+ columns or the equivalent. '
            'Decoy hits are indicated by the prefix of "XXX_" before a "Protein" entry.'))
    format_mgf_parser.add_argument(
        '--screen_fdr', 
        type=utils.check_between_zero_and_one, 
        help=(
            'Filter MSGF+ formatted database search results to the specified FDR. '
            'I have found that reported MSGF+ PSM q-values (FDRs) are inaccurate.'))

    #DeepNovo subcommands.
    train_deepnovo_parser = subparsers.add_parser(
        'train_deepnovo', 
        help=(
            'Train DeepNovo. '
            'The "setup" subcommand with options "deepnovo_low" or "deepnovo_high" '
            'should already have been run.'))
    train_deepnovo_parser.add_argument(
        '--mgf', 
        help=(
            'MGF file to be used for training. '
            'This file will be split into training, validation, and testing fractions.'))
    train_deepnovo_parser.add_argument(
        '--container', 
        help='Path to Singularity/Docker TensorFlow container image.')
    train_deepnovo_parser.add_argument(
        '--bind_point', 
        help=(
            'Directory within the scope of the singularity container '
            'used to bind a Postnovo/DeepNovo directory outside the scope of the container. '
            'By default, bind points are subdirectories created in the directory '
            'of the Singularity container and given the name, '
            '"DeepNovo.<frag_resolution>.<frag_mass_tol>".'))
    train_deepnovo_parser.add_argument(
        '--frag_resolution', 
        choices=['low', 'high'], 
        help=(
            'Resolution of fragmentation spectra in dataset. '
            'If frag_mass_tols option is not specified, '
            'then default fragment mass tolerances for "low" or "high" resolution are used.'))
    train_deepnovo_parser.add_argument(
        '--frag_mass_tols', 
        nargs='+', 
        help=(
            'Individual fragment mass tolerances with which to train DeepNovo. '
            'If this option is used, it overrides the default values from frag_resolution.'))
    train_deepnovo_parser.add_argument(
        '--fixed_mods', 
        nargs='+', 
        default=config.default_fixed_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Carbamidomethylation of C".'))
    train_deepnovo_parser.add_argument(
        '--variable_mods', 
        nargs='+', 
        default=config.default_variable_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Oxidation of M".'))
    train_deepnovo_parser.add_argument(
        '--cpus', 
        type=utils.check_positive_nonzero_int, 
        help='Number of CPUs to use per prediction task (used with and without slurm option).')
    train_deepnovo_parser.add_argument(
        '--slurm', 
        action='store_true', 
        default=False, 
        help='Flag to train DeepNovo at each mass tolerance on a compute cluster using Slurm.')
    train_deepnovo_parser.add_argument(
        '--partition', 
        help='Partition to use with Slurm.')
    train_deepnovo_parser.add_argument(
        '--time_limit', 
        type=utils.check_positive_nonzero_float, 
        help='Time limit for each mass tolerance training task in Slurm, in hours.')
    train_deepnovo_parser.add_argument(
        '--mem', 
        default='MaxMemPerNode', 
        help=(
            'Memory to allocate per training task, in GB: '
            '0.01 Da requires ~48 GB, '
            '0.03, 0.05 Da require ~32 GB, '
            '>0.05 Da require ~16 GB or less. '
            'This defaults to the maximum memory available on the node.'))

    predict_deepnovo_parser = subparsers.add_parser(
        'predict_deepnovo', 
        help='Use DeepNovo to predict de novo sequences.')
    predict_deepnovo_parser.add_argument(
        '--mgf', 
        help='Path to input MGF file.')
    predict_deepnovo_parser.add_argument(
        '--out', 
        help=(
            'Output directory for DeepNovo sequencing results. '
            'If not specified, defaults to the directory of the input MGF.'))
    predict_deepnovo_parser.add_argument(
        '--container', 
        help='Path to Singularity/Docker TensorFlow container image.')
    predict_deepnovo_parser.add_argument(
        '--bind_point', 
        help=(
            'Directory within the scope of the singularity container '
            'used to bind a Postnovo/DeepNovo directory outside the scope of the container. '
            'By default, bind points are subdirectories created in the directory '
            'of the Singularity container and given the name, '
            '"DeepNovo.<frag_resolution>.<frag_mass_tol>".'))
    predict_deepnovo_parser.add_argument(
        '--frag_resolution', 
        choices=['low', 'high'], 
        help=(
            'Resolution of fragmentation spectra in dataset. '
            'If frag_mass_tols option is not specified, '
            'then default fragment mass tolerances for "low" or "high" resolution are used.'))
    predict_deepnovo_parser.add_argument(
        '--frag_mass_tols', 
        nargs='+', 
        help=(
            'Individual fragment mass tolerances at which DeepNovo sequences are predicted. '
            'If this option is used, it overrides the default values from frag_resolution.'))
    predict_deepnovo_parser.add_argument(
        '--fixed_mods', 
        nargs='+', 
        default=config.default_fixed_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Carbamidomethylation of C".'))
    predict_deepnovo_parser.add_argument(
        '--variable_mods', 
        nargs='+', 
        default=config.default_variable_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Oxidation of M".'))
    predict_deepnovo_parser.add_argument(
        '--cpus', 
        type=utils.check_positive_nonzero_int, 
        help='Number of CPUs to use per prediction task (used with and without slurm option).')
    predict_deepnovo_parser.add_argument(
        '--slurm', 
        action='store_true', 
        default=False, 
        help='Flag to run DeepNovo at each mass tolerance on a compute cluster using Slurm.')
    predict_deepnovo_parser.add_argument(
        '--partition', 
        help='Partition to use with Slurm.')
    predict_deepnovo_parser.add_argument(
        '--time_limit', 
        type=utils.check_positive_nonzero_float, 
        help='Time limit for each mass tolerance prediction task in Slurm, in hours.')
    predict_deepnovo_parser.add_argument(
        '--mem', 
        default='MaxMemPerNode', 
        help=(
            'Memory to allocate per prediction task in Slurm, in GB. '
            '0.01 Da requires ~48 GB, '
            '0.03, 0.05 Da require ~32 GB, '
            '>0.05 Da require ~16 GB or less. '
            'This defaults to the maximum memory available on the node.'))

    #Postnovo subcommands.
    predict_parser = subparsers.add_parser(
        'predict', help='Predict de novo sequences.')
    predict_parser.add_argument(
        '--mgf', 
        help='Input MGF filepath.')
    predict_parser.add_argument(
        '--clusters', 
        help=(
            'MaRaCluster clusters tsv filepath. '
            'The file should be named MaRaCluster.clusters_p2.tsv by default, '
            'and is generated with a MaRaCluster log p-value threshold setting of -2. '
            'If this argument is not provided, '
            'Postnovo looks for a file called MaRaCluster.clusters_p2.tsv '
            'in the directory of the input MGF file.'))
    predict_parser.add_argument(
        '--out', 
        help=(
            'Output directory. If not specified, defaults to the directory of the input MGF.'))
    predict_parser.add_argument(
        '--precursor_mass_tol', 
        type=utils.check_positive_nonzero_float, 
        default=10, 
        help='Precursor mass tolerance in ppm.')
    predict_parser.add_argument(
        '--frag_method', 
        choices=['CID', 'HCD'], 
        help='Fragmentation method.')
    predict_parser.add_argument(
        '--frag_resolution', 
        choices=['low', 'high'], 
        help='Resolution of fragmentation spectra in dataset.')
    predict_parser.add_argument(
        '--fixed_mods', 
        nargs='+', 
        default=config.default_fixed_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Carbamidomethylation of C".'))
    predict_parser.add_argument(
        '--variable_mods', 
        nargs='+', 
        default=config.default_variable_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Oxidation of M".'))
    predict_parser.add_argument(
        '--denovogui', 
        action='store_true', 
        default=False, 
        help='Flag to run Novor and PepNovo+ via DeNovoGUI on the input MGF file.')
    predict_parser.add_argument(
        '--filename',
        help=(
            'If DeNovoGUI output files over the range of fragment mass tolerances '
            'were already generated from the input MGF file, '
            'provide the filename prefix for these files, '
            'which should be in the same directory as the MGF file: '
            'e.g., <filename>.0.2.novor.csv, <filename>.0.2.mgf.out, <filename>.0.2.tsv.'))
    predict_parser.add_argument(
        '--deepnovo', 
        action='store_true', 
        default=False, 
        help=(
            'Flag for use of de novo sequence predictions from DeepNovo: '
            'DeepNovo output files, e.g., <filename>.0.2.tsv, '
            'should be in the directory of the MGF file.'))
    predict_parser.add_argument(
        '--cpus', 
        type=int, 
        default=1, 
        help='Number of CPUs to use.')
    predict_parser.add_argument(
        '--quiet', 
        action='store_true', 
        default=False, 
        help='No messages to standard output.')
    predict_parser.add_argument(
        '--min_len', 
        type=int, 
        default=config.DEFAULT_MIN_LEN, 
        help=(
            'The minimum length of sequences reported by Postnovo, '
            'with the absolute minimum being {0} amino acids.'.format(config.DEFAULT_MIN_LEN)))
    predict_parser.add_argument(
        '--min_prob', 
        type=utils.check_positive_nonzero_float, 
        default=0.5, 
        help='Minimum estimated probability of reported Postnovo sequences.')
    predict_parser.add_argument(
        '--max_total_sacrifice', 
        type=float, 
        default=1, 
        help=(
            'The maximum estimated sequence probability that can be sacrificed '
            'to select a longer reported sequence '
            '(containing the most probable sequence as a subsequence) for the spectrum.'))
    predict_parser.add_argument(
        '--max_sacrifice_per_percent_extension', 
        type=float, 
        default=0.0035, 
        help=(
            'The maximum estimated sequence probability that can be sacrificed '
            'per percent change in sequence length: '
            'the default is 0.0035, or a max probability sacrifice of 0.05 '
            'to add an amino acid to a length 7 sequence (0.0035 = 0.05 / (1/7 * 100)); '
            'this is equivalent to a max score sacrifice of 0.025 '
            'to add an amino acid to a length 14 sequence.'))
    predict_parser.add_argument(
        '--feature_set', 
        type=int, 
        default=4, 
        help=(
            'The set of features (variables) used in Postnovo models. '
            'By default, the largest feature set (4) is used. '
            'Feature Set 0: features taken directly from mass spectral data '
            'and source de novo algorithm output. '
            'Feature Set 1: adds potential substitutional error features. '
            'Feature Set 2: adds consensus sequence models and features. '
            'Feature Set 3: adds fragment mass tolerance agreement features. '
            'Feature Set 4: adds features comparing sequences from different spectra '
            'but the same peptide species.'))

    test_parser = subparsers.add_parser('test', help='Test the Postnovo model.')
    test_parser.add_argument(
        '--mgf', 
        help='Input MGF filepath.')
    test_parser.add_argument(
        '--clusters', 
        help=(
            'MaRaCluster clusters tsv filepath. '
            'The file should be named MaRaCluster.clusters_p2.tsv by default, '
            'and is generated with a MaRaCluster log p-value threshold setting of -2. '
            'If this argument is not provided, '
            'Postnovo looks for a file called MaRaCluster.clusters_p2.tsv '
            'in the directory of the input MGF file.'))
    test_parser.add_argument(
        '--out', 
        help=(
            'Output directory. If not specified, defaults to the directory of the input MGF.'))
    test_parser.add_argument(
        '--precursor_mass_tol', 
        type=float, 
        default=10, 
        help='Precursor mass tolerance in ppm.')
    test_parser.add_argument(
        '--frag_method', 
        choices=['CID', 'HCD'], 
        help='Fragmentation method.')
    test_parser.add_argument(
        '--frag_resolution', 
        choices=['low', 'high'], 
        help='Resolution of fragmentation spectra in dataset.')
    test_parser.add_argument(
        '--fixed_mods', 
        nargs='+', 
        default=config.default_fixed_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Carbamidomethylation of C".'))
    test_parser.add_argument(
        '--variable_mods', 
        nargs='+', 
        default=config.default_variable_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Oxidation of M".'))
    test_parser.add_argument(
        '--denovogui', 
        action='store_true', 
        default=False, 
        help='Flag to run Novor and PepNovo+ via DeNovoGUI on the input MGF file.')
    test_parser.add_argument(
        '--filename', 
        help=(
            'If DeNovoGUI output files over the range of fragment mass tolerances '
            'were already generated from the input MGF file, '
            'provide the filename prefix for these files, '
            'which should be in the same directory as the MGF file: '
            'e.g., <filename>.0.2.novor.csv, <filename>.0.2.mgf.out, <filename>.0.2.tsv.'))
    test_parser.add_argument(
        '--deepnovo', 
        action='store_true', 
        default=False, 
        help=(
            'Flag for use of de novo sequence predictions from DeepNovo: '
            'DeepNovo output files, e.g., <filename>.0.2.tsv, '
            'should be in the directory of the MGF file.'))
    test_parser.add_argument(
        '--ref_fasta', 
        help=(
            'FASTA reference file that was used in database search, '
            'which should be in the directory of the MGF file.'))
    test_parser.add_argument(
        '--db_search', 
        help=('Table of PSMs produced by MSGF+ in TSV format, '
              'which should be in the directory of the MGF file.'))
    test_parser.add_argument(
        '--reconcile_spectrum_ids', 
        action='store_true', 
        default=False, 
        help=('Use this flag if the input MGF file produced by Postnovo format_mgf ' 
              'is not the same as the MGF file used in database search.'))
    test_parser.add_argument(
        '--msgf', 
        action='store_true', 
        default=False, 
        help='Flag to search the MGF file against the FASTA file using MSGF+.')
    test_parser.add_argument(
        '--qexactive', 
        action='store_true', 
        default=False, 
        help='Flag to use the MSGF+ Q-Exactive model tailored to those instruments.')
    test_parser.add_argument(
        '--fdr_cutoff', 
        type=float, 
        default=0.01, 
        help=(
            'Filter MSGF+ formatted database search results to the specified FDR. '
            'For no FDR screening, specify a value of 1. '
            'I have found that reported MSGF+ PSM q-values (FDRs) are inaccurate.'))
    test_parser.add_argument(
        '--test_plots', 
        action='store_true', 
        default=False, 
        help=(
            'Make plots to judge the performance of Postnovo '
            'and its constituent random forest models. '
            'These include precision-recall and precision-yield plots, '
            'and plots of true versus predicted accuracy.'))
    test_parser.add_argument(
        '--cpus', 
        type=utils.check_positive_nonzero_int, 
        default=1, 
        help='Number of CPUs to use.')
    test_parser.add_argument(
        '--quiet', 
        action='store_true', 
        default=False, 
        help='No messages to standard output.')
    test_parser.add_argument(
        '--min_len', 
        type=int, 
        default=config.DEFAULT_MIN_LEN, 
        help=(
            'The minimum length of sequences reported by Postnovo, '
            'with the absolute minimum being {0} amino acids.'.format(config.DEFAULT_MIN_LEN)))
    test_parser.add_argument(
        '--feature_set', 
        type=int, 
        default=4, 
        help=(
            'The set of features (variables) used in Postnovo models. '
            'By default, the largest feature set (4) is used. '
            'Feature Set 0: features taken directly from mass spectral data '
            'and source de novo algorithm output. '
            'Feature Set 1: adds potential substitutional error features. '
            'Feature Set 2: adds consensus sequence models and features. '
            'Feature Set 3: adds fragment mass tolerance agreement features. '
            'Feature Set 4: adds features comparing sequences from different spectra '
            'but the same peptide species.'))

    train_parser = subparsers.add_parser('train', help='Train the Postnovo random forest model.')
    train_parser.add_argument(
        '--mgf', 
        help='Input MGF filepath.')
    train_parser.add_argument(
        '--clusters', 
        help=(
            'MaRaCluster clusters tsv filepath. '
            'The file should be named MaRaCluster.clusters_p2.tsv by default, '
            'and is generated with a MaRaCluster log p-value threshold setting of -2. '
            'If this argument is not provided, '
            'Postnovo looks for a file called MaRaCluster.clusters_p2.tsv '
            'in the directory of the input MGF file.'))
    train_parser.add_argument(
        '--out', 
        help=(
            'Output directory. If not specified, defaults to the directory of the input MGF.'))
    train_parser.add_argument(
        '--precursor_mass_tol', 
        type=float, 
        default=10, 
        help='Precursor mass tolerance in ppm.')
    train_parser.add_argument(
        '--frag_method', 
        choices=['CID', 'HCD'], 
        help='Fragmentation method.')
    train_parser.add_argument(
        '--frag_resolution', 
        choices=['low', 'high'], 
        help='Resolution of fragmentation spectra in dataset.')
    train_parser.add_argument(
        '--fixed_mods', 
        nargs='+', 
        default=config.default_fixed_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Carbamidomethylation of C".'))
    train_parser.add_argument(
        '--variable_mods', 
        nargs='+', 
        default=config.default_variable_mods, 
        help=(
            'This cannot currently be changed from the default value, '
            '"Oxidation of M".'))
    train_parser.add_argument(
        '--denovogui', 
        action='store_true', 
        default=False, 
        help='Flag to run Novor and PepNovo+ via DeNovoGUI on the input MGF file.')
    train_parser.add_argument(
        '--filename', 
        help=(
            'If DeNovoGUI output files over the range of fragment mass tolerances '
            'were already generated from the input MGF file, '
            'provide the filename prefix for these files, '
            'which should be in the same directory as the MGF file: '
            'e.g., <filename>.0.2.novor.csv, <filename>.0.2.mgf.out, <filename>.0.2.tsv.'))
    train_parser.add_argument(
        '--deepnovo', 
        action='store_true', 
        default=False, 
        help=(
            'Flag for use of de novo sequence predictions from DeepNovo: '
            'DeepNovo output files, e.g., <filename>.0.2.tsv, '
            'should be in the directory of the MGF file.'))
    train_parser.add_argument(
        '--ref_fasta', 
        help=(
            'FASTA reference file that was used in database search, '
            'which should be in the directory of the MGF file.'))
    train_parser.add_argument(
        '--db_search', 
        help=(
            'Table of PSMs produced by MSGF+ in TSV format, '
            'which should be in the directory of the MGF file.'))
    train_parser.add_argument(
        '--reconcile_spectrum_ids', 
        action='store_true', 
        default=False, 
        help=(
            'Use this flag if the input MGF file produced by Postnovo format_mgf ' 
            'is not the same as the MGF file used in database search.'))
    train_parser.add_argument(
        '--msgf', 
        action='store_true', 
        default=False, 
        help='Flag to search the MGF file against the FASTA file using MSGF+.')
    train_parser.add_argument(
        '--qexactive', 
        action='store_true', 
        default=False, 
        help='Flag to use the MSGF+ Q-Exactive model tailored to those instruments.')
    train_parser.add_argument(
        '--fdr_cutoff', 
        type=float, 
        default=0.01, 
        help=(
            'Filter MSGF+ formatted database search results to the specified FDR. '
            'For no FDR screening, specify a value of 1. '
            'I have found that reported MSGF+ PSM q-values (FDRs) are inaccurate.'))
    train_parser.add_argument(
        '--stop_before_training', 
        action='store_true', 
        default=False, 
        help=(
            'Create the training files for the added dataset, '
            'but do not create new random forest models with all of the training datasets. '
            'This flag is useful for adding multiple new datasets before training the model.'))
    train_parser.add_argument(
        '--leave_one_out', 
        action='store_true', 
        default=False, 
        help=(
            'Conduct a leave-one-out analysis of model accuracy with each training dataset. '
            'This produces plots of predicted versus true accuracy '
            'both for the overall model and individual random forest models.'))
    train_parser.add_argument(
        '--plot_feature_importance', 
        action='store_true', 
        default=False, 
        help='Plot random forest feature importances.')
    train_parser.add_argument(
        '--retrain', 
        action='store_true', 
        default=False, 
        help=(
            'Create Postnovo random forest training models '
            'with the list of training datasets found in train_record.tsv. '
            'No new dataset can be added when this flag is used. '
            'The only other options that should be used with this flag are '
            '--frag_resolution (mandatory; Q-Exactive datasets are high-resoluion), '
            '--out (needed with --leave_one_out and --plot_feature_importance), '
            '--deepnovo (if DeepNovo was used as a de novo algorithm for the datasets), '
            '--feature_set (if a different feature set was used for the datasets), '
            '--leave_one_out, --plot_feature_importance, --cpus, and --quiet.'))
    train_parser.add_argument(
        '--cpus', 
        type=int, 
        default=1, 
        help='Number of CPUs to use.')
    train_parser.add_argument(
        '--quiet', 
        action='store_true', 
        default=False, 
        help='No messages to standard output.')
    train_parser.add_argument(
        '--feature_set', 
        type=int, 
        default=4, 
        help=(
            'The set of features (variables) used in Postnovo models. '
            'By default, the largest feature set (4) is used. '
            'Feature Set 0: features taken directly from mass spectral data '
            'and source de novo algorithm output. '
            'Feature Set 1: adds potential substitutional error features. '
            'Feature Set 2: adds consensus sequence models and features. '
            'Feature Set 3: adds fragment mass tolerance agreement features. '
            'Feature Set 4: adds features comparing sequences from different spectra '
            'but the same peptide species.'))

    if test_argv:
        #User arguments were not specified on the command line, but in "main" module.
        args = parser.parse_args(test_argv)
    else:
        args = parser.parse_args()

    ##REMOVE: There are only two default modifications, to C and M.
    #if args.subparser_name == 'mods_list':
    #    print_accepted_mods()
    #    sys.exit(0)
    if args.subparser_name == 'setup':
        set_up_postnovo(
            args.check_updates, 
            args.denovogui, 
            args.msgf, 
            args.postnovo_low, 
            args.postnovo_high, 
            args.deepnovo_low, 
            args.deepnovo_high, 
            args.container, 
            args.bind_point, 
            args.postnovo_low_spectra, 
            args.postnovo_high_spectra, 
            args.deepnovo_low_spectra, 
            args.deepnovo_high_spectra)
        sys.exit(0)
    if args.subparser_name == 'format_mgf':
        format_mgf(
            args.mgfs, 
            args.out, 
            args.db_searches, 
            args.screen_fdr, 
            args.subsample, 
            args.remove_low_mass)
        sys.exit(0)
    elif args.subparser_name == 'train_deepnovo' or args.subparser_name == 'predict_deepnovo':
        #Perform checks and get the necessary parameters for DeepNovo, regardless of mode.
        frag_mass_tols, fixed_mod_aas, variable_mod_aas, time_limit = prepare_deepnovo(
            args.mgf, 
            args.container, 
            args.frag_resolution, 
            args.frag_mass_tols, 
            args.fixed_mods, 
            args.variable_mods, 
            args.bind_point, 
            args.time_limit, 
            args.mem)
        if args.subparser_name == 'train_deepnovo':
            train_deepnovo(
                args.mgf, 
                args.container, 
                args.bind_point, 
                args.frag_resolution, 
                frag_mass_tols, 
                fixed_mod_aas, 
                variable_mod_aas, 
                args.slurm, 
                args.partition, 
                time_limit, 
                args.cpus, 
                args.mem)
            sys.exit(0)
        elif args.subparser_name == 'predict_deepnovo':
            if args.out == None:
                out_dir = os.path.dirname(args.mgf)
            else:
                utils.check_path(args.out)
                out_dir = args.out

            #Check that DeepNovo was trained at each fragment mass tolerance.
            for frag_mass_tol in frag_mass_tols:
                #The Postnovo training routine creates a directory for each parameterization.
                target_deepnovo_dir = os.path.join(
                    config.deepnovo_dir, 
                    'DeepNovo.' + args.frag_resolution + '.' + frag_mass_tol)
                if not os.path.exists(target_deepnovo_dir):
                    raise RuntimeError(target_deepnovo_dir + ' does not exist.')

            predict_deepnovo(
                args.mgf, 
                out_dir, 
                args.container, 
                args.bind_point, 
                args.frag_resolution, 
                frag_mass_tols, 
                fixed_mod_aas, 
                variable_mod_aas, 
                args.slurm, 
                args.partition, 
                time_limit, 
                args.cpus, 
                args.mem)
            sys.exit(0)

    if args.subparser_name == 'predict':
        config.globals['Mode'] = 'predict'
    elif args.subparser_name == 'test':
        config.globals['Mode'] = 'test'
    elif args.subparser_name == 'train':
        config.globals['Mode'] = 'train'

    inspect_args(args)
    if not config.globals['Retrain']:
        if config.globals['Run DeNovoGUI']:
            run_denovogui()
        if config.globals['Run MSGF']:
            run_msgf()
    make_additional_globals()

    return

##REMOVE: There are only two default modifications, to C and M.
#def print_accepted_mods():

#    print('Accepted DeNovoGUI (Novor and PepNovo+) mods:')
#    for denovogui_mod_code in config.denovogui_postnovo_mod_dict:
#        print(denovogui_mod_code)
#    print('Accepted DeepNovo mods:')
#    for denovogui_mod_code in config.denovogui_deepnovo_config_mod_dict:
#        print(denovogui_mod_code)

#    return

def set_up_postnovo(
    check_updates, 
    denovogui, 
    msgf, 
    postnovo_low, 
    postnovo_high, 
    deepnovo_low, 
    deepnovo_high, 
    container_fp, 
    bind_point, 
    postnovo_low_spectra, 
    postnovo_high_spectra, 
    deepnovo_low_spectra, 
    deepnovo_high_spectra):
    '''
    Sets up the default Postnovo and DeepNovo models.

    Parameters
    ----------
    check_updates : bool
    denovogui : bool
    msgf : bool
    postnovo_low : bool
    postnovo_high : bool
    deepnovo_low : bool
    deepnovo_high : bool
    container_fp : str
        Filepath to TensorFlow container image.
    bind_point : str
        Singularity bind point (directory) within the scope of the TensorFlow container.
    postnovo_low_spectra : bool
    postnovo_high_spectra : bool
    deepnovo_low_spectra : bool
    deepnovo_high_spectra : bool

    Returns
    -------
    None
    '''

    #Download a table of the most up-to-date files that can be downloaded from Google Drive.
    #The maintainer of these downloadable files should try to ensure that 
    #the Google Drive ID changes with each change in the file, 
    #but this requires deleting the previous version of the file on Google Drive 
    #before uploading the new version.
    #Both ID and file size are compared between the new download 
    #and the user's version of the file, if present.
    #This should help ensure that only true updates, not redundant updates, can occur.
    try:
        current_downloads_df = pd.read_csv(
            StringIO(requests.get(
                config.GOOGLE_DRIVE_DOWNLOAD_URL, 
                params={'id': config.CURRENT_DOWNLOAD_GOOGLE_DRIVE_ID}).text), 
            sep='\t', 
            index_col='Download Filename')
    except requests.ConnectionError:
        raise RuntimeError('An internet connection is needed to download files for setup.')
    #Load the table of the Google Drive files that are in use.
    user_downloads_df = pd.read_csv(
        config.download_ids_tsv, sep='\t', index_col='Download Filename')

    if check_updates:
        for filename in current_downloads_df.index:
            if filename == 'knapsack.npy':
                continue
            if filename in user_downloads_df.index:
                if (
                    current_downloads_df.loc[filename]['Google Drive ID'] != 
                    user_downloads_df.loc[filename]['Google Drive ID']) or (
                        current_downloads_df.loc[filename]['Size'] != 
                        user_downloads_df.loc[filename]['Size']):
                    print('The previously downloaded version of', filename, 'is not up-to-date.')
            else:
                print(filename, 'has not been downloaded.')

        return

    if denovogui:
        current_download_id = current_downloads_df.loc['DeNovoGUI.zip']['Google Drive ID']
        current_download_size = current_downloads_df.loc['DeNovoGUI.zip']['Size']

        continue_with_download = True
        if 'DeNovoGUI.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['DeNovoGUI.zip']['Google Drive ID']
            user_download_size = user_downloads_df.loc['DeNovoGUI.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded DeNovoGUI was already up-to-date. '
                    'Delete the line for "DeNovoGUI.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            denovogui_zip_fp = os.path.join(config.postnovo_dir, 'DeNovoGUI.zip')

            print('Downloading DeNovoGUI.zip')
            utils.download_file_from_google_drive(
                current_download_id, denovogui_zip_fp, current_download_size)

            print('Unzipping DeNovoGUI.zip')
            with zipfile.ZipFile(denovogui_zip_fp) as f:
                f.extractall(config.postnovo_dir)
            os.remove(denovogui_zip_fp)

            #Update the user download record.
            if 'DeNovoGUI.zip' in user_downloads_df.index:
                user_downloads_df.drop('DeNovoGUI.zip', inplace=True)
            user_downloads_df.loc['DeNovoGUI.zip'] = [current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if msgf:
        current_download_id = current_downloads_df.loc['MSGFPlus.zip']['Google Drive ID']
        current_download_size = current_downloads_df.loc['MSGFPlus.zip']['Size']

        continue_with_download = True
        if 'MSGFPlus.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['MSGFPlus.zip']['Google Drive ID']
            user_download_size = user_downloads_df.loc['MSGFPlus.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded MSGF+ was already up-to-date. '
                    'Delete the line for "MSGFPlus.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            msgf_zip_fp = os.path.join(config.postnovo_dir, 'MSGFPlus.zip')

            print('Downloading MSGFPlus.zip')
            utils.download_file_from_google_drive(
                current_download_id, msgf_zip_fp, current_download_size)

            print('Unzipping MSGFPlus.zip')
            with zipfile.ZipFile(msgf_zip_fp) as f:
                f.extractall(config.postnovo_dir)
            os.remove(msgf_zip_fp)

            #Update the user download record.
            if 'MSGFPlus.zip' in user_downloads_df.index:
                user_downloads_df.drop('MSGFPlus.zip', inplace=True)
            user_downloads_df.loc['MSGFPlus.zip'] = [current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if postnovo_low:
        current_download_id = current_downloads_df.loc['postnovo_low_default_models.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['postnovo_low_default_models.zip']['Size']

        continue_with_download = True
        if 'postnovo_low_default_models.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['postnovo_low_default_models.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['postnovo_low_default_models.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded Postnovo low-resolution model '
                    'was already up-to-date. '
                    'Delete the line for "postnovo_low_default_models.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            postnovo_low_default_models_zip_fp = os.path.join(
                config.postnovo_train_dir_dict['Low'], 'postnovo_low_default_models.zip')
            if not os.path.isdir(config.postnovo_train_dir_dict['Low']):
                os.mkdir(config.postnovo_train_dir_dict['Low'])

            print('Downloading postnovo_low_default_models.zip')
            utils.download_file_from_google_drive(
                current_download_id, postnovo_low_default_models_zip_fp, current_download_size)

            print('Unzipping postnovo_low_default_models.zip')
            with zipfile.ZipFile(postnovo_low_default_models_zip_fp) as f:
                f.extractall(config.postnovo_train_dir_dict['Low'])
            os.remove(postnovo_low_default_models_zip_fp)

            #Download the low-resolution training record, 
            #which is required to run "train" mode with the default model.
            utils.download_file_from_google_drive(
                current_downloads_df.loc['train_record_low.tsv']['Google Drive ID'], 
                os.path.join(config.postnovo_train_dir_dict['Low'], 'train_record.tsv'), 
                current_downloads_df.loc['train_record_low.tsv']['Size'])

            #Update the user download record.
            if 'postnovo_low_default_models.zip' in user_downloads_df.index:
                user_downloads_df.drop('postnovo_low_default_models.zip', inplace=True)
            user_downloads_df.loc['postnovo_low_default_models.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if postnovo_high:
        current_download_id = current_downloads_df.loc['postnovo_high_default_models.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['postnovo_high_default_models.zip'][
            'Size']

        continue_with_download = True
        if 'postnovo_high_default_models.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['postnovo_high_default_models.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['postnovo_high_default_models.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded Postnovo high-resolution model '
                    'was already up-to-date. '
                    'Delete the line for "postnovo_high_default_models.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            postnovo_high_default_models_zip_fp = os.path.join(
                config.postnovo_train_dir_dict['High'], 'postnovo_high_default_models.zip')
            if not os.path.isdir(config.postnovo_train_dir_dict['High']):
                os.mkdir(config.postnovo_train_dir_dict['High'])

            print('Downloading postnovo_high_default_models.zip')
            utils.download_file_from_google_drive(
                current_download_id, postnovo_high_default_models_zip_fp, current_download_size)

            print('Unzipping postnovo_high_default_models.zip')
            with zipfile.ZipFile(postnovo_high_default_models_zip_fp) as f:
                f.extractall(config.postnovo_train_dir_dict['High'])
            os.remove(postnovo_high_default_models_zip_fp)

            #Download the high-resolution training record, 
            #which is required to run "train" mode with the default model.
            utils.download_file_from_google_drive(
                current_downloads_df.loc['train_record_high.tsv']['Google Drive ID'], 
                os.path.join(config.postnovo_train_dir_dict['High'], 'train_record.tsv'), 
                current_downloads_df.loc['train_record_high.tsv']['Size'])

            #Update the user download record.
            if 'postnovo_high_default_models.zip' in user_downloads_df.index:
                user_downloads_df.drop('postnovo_high_default_models.zip', inplace=True)
            user_downloads_df.loc['postnovo_high_default_models.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if deepnovo_low:
        current_download_id = current_downloads_df.loc['deepnovo_low_default_models.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['deepnovo_low_default_models.zip']['Size']

        continue_with_download = True
        if 'deepnovo_low_default_models.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['deepnovo_low_default_models.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['deepnovo_low_default_models.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded DeepNovo low-resolution model '
                    'was already up-to-date. '
                    'Delete the line for "deepnovo_low_default_models.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            deepnovo_low_default_models_zip_fp = os.path.join(
                config.deepnovo_dir, 'deepnovo_low_default_models.zip')

            print('Downloading deepnovo_low_default_models.zip')
            utils.download_file_from_google_drive(
                current_download_id, deepnovo_low_default_models_zip_fp, current_download_size)

            print('Unzipping deepnovo_low_default_models.zip')
            with zipfile.ZipFile(deepnovo_low_default_models_zip_fp) as f:
                f.extractall(config.deepnovo_dir)
            os.remove(deepnovo_low_default_models_zip_fp)

            knapsack_fp = os.path.join(config.deepnovo_dir, 'knapsack.npy')
            if not os.path.exists(knapsack_fp):
                print('Downloading knapsack.npy')
                utils.download_file_from_google_drive(
                    current_downloads_df.loc['knapsack.npy']['Google Drive ID'], 
                    knapsack_fp, 
                    current_downloads_df.loc['knapsack.npy']['Size'])
            #Create a model directory for each fragment mass tolerance.
            set_up_deepnovo_model('low', container_fp, bind_point)
            shutil.rmtree(os.path.join(config.deepnovo_dir, 'deepnovo_low_default_models'))

            #Update the user download record.
            if 'deepnovo_low_default_models.zip' in user_downloads_df.index:
                user_downloads_df.drop('deepnovo_low_default_models.zip', inplace=True)
            user_downloads_df.loc['deepnovo_low_default_models.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if deepnovo_high:
        current_download_id = current_downloads_df.loc['deepnovo_high_default_models.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['deepnovo_high_default_models.zip'][
            'Size']

        continue_with_download = True
        if 'deepnovo_high_default_models.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['deepnovo_high_default_models.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['deepnovo_high_default_models.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded DeepNovo high-resolution model '
                    'was already up-to-date. '
                    'Delete the line for "deepnovo_high_default_models.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            deepnovo_high_default_models_zip_fp = os.path.join(
                config.deepnovo_dir, 'deepnovo_high_default_models.zip')

            print('Downloading deepnovo_high_default_models.zip')
            utils.download_file_from_google_drive(
                current_download_id, deepnovo_high_default_models_zip_fp, current_download_size)

            print('Unzipping deepnovo_high_default_models.zip')
            with zipfile.ZipFile(deepnovo_high_default_models_zip_fp) as f:
                f.extractall(config.deepnovo_dir)
            os.remove(deepnovo_high_default_models_zip_fp)

            knapsack_fp = os.path.join(config.deepnovo_dir, 'knapsack.npy')
            if not os.path.exists(knapsack_fp):
                print('Downloading knapsack.npy')
                utils.download_file_from_google_drive(
                    current_downloads_df.loc['knapsack.npy']['Google Drive ID'], 
                    knapsack_fp, 
                    current_downloads_df.loc['knapsack.npy']['Size'])
            set_up_deepnovo_model('high', container_fp, bind_point)
            shutil.rmtree(os.path.join(config.deepnovo_dir, 'deepnovo_high_default_models'))

            #Update the user download record.
            if 'deepnovo_high_default_models.zip' in user_downloads_df.index:
                user_downloads_df.drop('deepnovo_high_default_models.zip', inplace=True)
            user_downloads_df.loc['deepnovo_high_default_models.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if postnovo_low_spectra:
        current_download_id = current_downloads_df.loc['postnovo_low_default_spectra.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['postnovo_low_default_spectra.zip'][
            'Size']

        continue_with_download = True
        if 'postnovo_low_default_spectra.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['postnovo_low_default_spectra.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['postnovo_low_default_spectra.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded Postnovo low-resolution training spectra '
                    'were already up-to-date. '
                    'Delete the line for "postnovo_low_default_spectra.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            postnovo_low_default_spectra_zip_fp = os.path.join(
                config.postnovo_train_dir_dict['Low'], 'postnovo_low_default_spectra.zip')

            print('Downloading postnovo_low_default_spectra.zip')
            utils.download_file_from_google_drive(
                current_download_id, postnovo_low_default_spectra_zip_fp, current_download_size)

            print('Unzipping postnovo_low_default_spectra.zip')
            with zipfile.ZipFile(postnovo_low_default_spectra_zip_fp) as f:
                f.extractall(config.postnovo_train_dir_dict['Low'])
            os.remove(postnovo_low_default_spectra_zip_fp)

            #Update the user download record.
            if 'postnovo_low_default_spectra.zip' in user_downloads_df.index:
                user_downloads_df.drop('postnovo_low_default_spectra.zip', inplace=True)
            user_downloads_df.loc['postnovo_low_default_spectra.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if postnovo_high_spectra:
        current_download_id = current_downloads_df.loc['postnovo_high_default_spectra.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['postnovo_high_default_spectra.zip'][
            'Size']

        continue_with_download = True
        if 'postnovo_high_default_spectra.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['postnovo_high_default_spectra.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['postnovo_high_default_spectra.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded Postnovo high-resolution training spectra '
                    'were already up-to-date. '
                    'Delete the line for "postnovo_high_default_spectra.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            postnovo_high_default_spectra_zip_fp = os.path.join(
                config.postnovo_train_dir_dict['High'], 'postnovo_high_default_spectra.zip')

            print('Downloading postnovo_high_default_spectra.zip')
            utils.download_file_from_google_drive(
                current_download_id, postnovo_high_default_spectra_zip_fp, current_download_size)

            print('Unzipping postnovo_high_default_spectra.zip')
            with zipfile.ZipFile(postnovo_high_default_spectra_zip_fp) as f:
                f.extractall(config.postnovo_train_dir_dict['High'])
            os.remove(postnovo_high_default_spectra_zip_fp)

            #Update the user download record.
            if 'postnovo_high_default_spectra.zip' in user_downloads_df.index:
                user_downloads_df.drop('postnovo_high_default_spectra.zip', inplace=True)
            user_downloads_df.loc['postnovo_high_default_spectra.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if deepnovo_low_spectra:
        current_download_id = current_downloads_df.loc['deepnovo_low_default_spectra.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['deepnovo_low_default_spectra.zip'][
            'Size']

        continue_with_download = True
        if 'deepnovo_low_default_spectra.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['deepnovo_low_default_spectra.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['deepnovo_low_default_spectra.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded DeepNovo low-resolution training spectra '
                    'were already up-to-date. '
                    'Delete the line for "deepnovo_low_default_spectra.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            deepnovo_low_default_spectra_zip_fp = os.path.join(
                config.deepnovo_dir, 'deepnovo_low_default_spectra.zip')

            print('Downloading deepnovo_low_default_spectra.zip')
            utils.download_file_from_google_drive(
                current_download_id, deepnovo_low_default_spectra_zip_fp, current_download_size)

            print('Unzipping deepnovo_low_default_spectra.zip')
            with zipfile.ZipFile(deepnovo_low_default_spectra_zip_fp) as f:
                f.extractall(config.deepnovo_dir)
            os.remove(deepnovo_low_default_spectra_zip_fp)

            #Update the user download record.
            if 'deepnovo_low_default_spectra.zip' in user_downloads_df.index:
                user_downloads_df.drop('deepnovo_low_default_spectra.zip', inplace=True)
            user_downloads_df.loc['deepnovo_low_default_spectra.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    if deepnovo_high_spectra:
        current_download_id = current_downloads_df.loc['deepnovo_high_default_spectra.zip'][
            'Google Drive ID']
        current_download_size = current_downloads_df.loc['deepnovo_high_default_spectra.zip'][
            'Size']

        continue_with_download = True
        if 'deepnovo_high_default_spectra.zip' in user_downloads_df.index:
            user_download_id = user_downloads_df.loc['deepnovo_high_default_spectra.zip'][
                'Google Drive ID']
            user_download_size = user_downloads_df.loc['deepnovo_high_default_spectra.zip']['Size']
            if (current_download_id == user_download_id) and \
                (current_download_size == user_download_size):
                continue_with_download = False
                print(
                    'The previously downloaded DeepNovo high-resolution training spectra '
                    'were already up-to-date. '
                    'Delete the line for "deepnovo_high_default_spectra.zip" in the file, '
                    '"postnovo/download_ids.tsv", and re-run this command to download it again.')

        if continue_with_download:
            deepnovo_high_default_spectra_zip_fp = os.path.join(
                config.deepnovo_dir, 'deepnovo_high_default_spectra.zip')

            print('Downloading deepnovo_high_default_spectra.zip')
            utils.download_file_from_google_drive(
                current_download_id, deepnovo_high_default_spectra_zip_fp, current_download_size)

            print('Unzipping deepnovo_high_default_spectra.zip')
            with zipfile.ZipFile(deepnovo_high_default_spectra_zip_fp) as f:
                f.extractall(config.deepnovo_dir)
            os.remove(deepnovo_high_default_spectra_zip_fp)

            #Update the user download record.
            if 'deepnovo_high_default_spectra.zip' in user_downloads_df.index:
                user_downloads_df.drop('deepnovo_high_default_spectra.zip', inplace=True)
            user_downloads_df.loc['deepnovo_high_default_spectra.zip'] = [
                current_download_id, current_download_size]
            user_downloads_df.to_csv(config.download_ids_tsv, sep='\t')

    #Remove the large file, knapsack.npy, that was placed in the top DeepNovo directory 
    #to copy into the subdirectories that are actually used.
    if os.path.exists(os.path.join(config.deepnovo_dir, 'knapsack.npy')):
        os.remove(os.path.join(config.deepnovo_dir, 'knapsack.npy'))

    return

def set_up_deepnovo_model(frag_resolution, container_fp, bind_point):
    '''
    Sets up default DeepNovo models in directories for each fragment mass tolerance.

    Parameters
    ----------
    frag_resolution : <'High' or 'Low'>
    container_fp : str
        Filepath to TensorFlow image.
    bind_point : str
        Filepath to bind point within scope of TensorFlow container.

    Returns
    -------
    None
    '''

    encountered_cython_compile_err = False
    default_model_dir = os.path.join(
        config.deepnovo_dir, 'deepnovo_' + frag_resolution + '_default_models')

    for frag_mass_tol in config.frag_mass_tol_dict[frag_resolution.capitalize()]:            
        target_deepnovo_dir = os.path.join(
            config.deepnovo_dir, 'DeepNovo.' + frag_resolution + '.' + frag_mass_tol)
        target_train_dir = os.path.join(target_deepnovo_dir, 'train')

        try:
            #Try to load a Singularity environment module, 
            #which may be needed for Cython compilation.
            with open(os.devnull, 'w') as null_f:
                subprocess.call('module load singularity', shell=True, stderr=null_f)
        except FileNotFoundError:
            pass

        if os.path.isdir(target_deepnovo_dir):
            #The target DeepNovo directory already exists.
            #Check for each program file, including compiled Cython files.
            for program_file_fp in config.deepnovo_program_fps + \
                [os.path.join(target_deepnovo_dir, 'deepnovo_cython_modules.so'), 
                    os.path.join(target_deepnovo_dir, 'deepnovo_cython_modules.c')]:
                if not os.path.exists(program_file_fp):
                    print(
                        'Not every program file was found in ' + target_deepnovo_dir + 
                        ', so the program files there are being refreshed.')
                    subprocess.call(['cp'] + config.deepnovo_program_fps + [target_deepnovo_dir])

                    #Compile Cython modules.
                    #Create a bind point for the proper DeepNovo directory 
                    #within the Singularity container.
                    if container_fp == None:
                        raise RuntimeError(
                            'For DeepNovo Cython files to be compiled, '
                            'please use the "container" option.')
                    if bind_point == None:
                        #Create a bind point for the proper DeepNovo directory 
                        #within the Singularity container, by default.
                        bind_point = os.path.join(
                            os.path.dirname(container_fp), 
                            os.path.basename(target_deepnovo_dir))
                    if not os.path.isdir(bind_point):
                        os.mkdir(bind_point)

                    os.chdir(bind_point)
                    p = subprocess.Popen([
                        'singularity', 
                        'exec', 
                        '--bind', 
                        target_deepnovo_dir + ':' + bind_point, 
                        container_fp, 
                        'python', 
                        os.path.join(bind_point, 'deepnovo_cython_setup.py'), 
                        'build_ext', 
                        '--inplace'], stderr=subprocess.PIPE)
                    err = p.communicate()[1]
                    if err.decode() != '':
                        encountered_cython_compile_err = True

                    break
            if os.path.isdir(target_train_dir):
                pass
            else:
                os.mkdir(target_train_dir)
        else:
            #The target DeepNovo directory does not exist.
            os.mkdir(target_deepnovo_dir)
            os.mkdir(target_train_dir)
            subprocess.call(['cp'] + config.deepnovo_program_fps + [target_deepnovo_dir])

            #Compile Cython modules.
            #Create a bind point for the proper DeepNovo directory 
            #within the Singularity container.
            if container_fp == None:
                raise RuntimeError(
                    'For DeepNovo Cython files to be compiled, '
                    'please use the "container" option.')
            if bind_point == None:
                #Create a bind point for the proper DeepNovo directory 
                #within the Singularity container, by default.
                bind_point = os.path.join(
                    os.path.dirname(container_fp), 
                    os.path.basename(target_deepnovo_dir))
            if not os.path.isdir(bind_point):
                os.mkdir(bind_point)

            os.chdir(bind_point)
            p = subprocess.Popen([
                'singularity', 
                'exec', 
                '--bind', 
                target_deepnovo_dir + ':' + bind_point, 
                container_fp, 
                'python', 
                os.path.join(bind_point, 'deepnovo_cython_setup.py'), 
                'build_ext', 
                '--inplace'], stderr=subprocess.PIPE)
            err = p.communicate()[1]
            if err.decode() != '':
                encountered_cython_compile_err = True

        #Move model files into the training directory for the fragment mass tolerance.
        frag_mass_tol_default_model_dir = os.path.join(
            default_model_dir, 'DeepNovo.' + frag_resolution + '.' + frag_mass_tol)
        for unzipped_filename in os.listdir(frag_mass_tol_default_model_dir):
            os.rename(
                os.path.join(frag_mass_tol_default_model_dir, unzipped_filename), 
                os.path.join(target_train_dir, unzipped_filename))

        if encountered_cython_compile_err:
            print(
                'DeepNovo Cython modules could not be compiled. '
                'Try using Python 2.7 to compile in each DeepNovo directory. '
                'The necessary "Cython" package may not be installed in Python -- '
                'to install, run the command, "pip install --user Cython". '
                'To compile the DeepNovo Cython code, run the following command in each directory, '
                '"python deepnovo_cython_setup.py build_ext --inplace".')

    return

def format_mgf(mgf_fps, out_mgf_fp, db_search_fps, fdr_cutoff, subsample_size, min_mass):
    '''
    Write new MGF files compatible with Postnovo.

    Parameters
    ----------
    mgf_fps : list of str
        Input MGF filepaths.
        MGF files must have a specific format of the spectrum title line, 
        TITLE=Run: <RUN ID>, Index: <SPECTRUM INDEX>, Scan: <SCAN NUMBER>
    out_mgf_fp : str
        Output MGF filepath.
    db_search_fps : list of str
        Database search results in MSGF+ tsv format for each MGF file.
        Only required for DeepNovo training, as the sequences are placed in each spectrum header.
    fdr_cutoff : float
        FDR cutoff for filtering spectra by database search results.
    subsample_size : int
        Number of spectra to randomly subsample from each MGF file.
    min_mass : float
        Precursor mass cutoff below which spectra are ignored.

    Returns
    -------
    None
    '''

    #Check the existence of MGF and database search result files.
    nonexistent_fps = []
    for mgf_fp in mgf_fps:
        nonexistent_fps.append(utils.check_path(mgf_fp, return_str=True))
        if nonexistent_fps[-1] == None:
            nonexistent_fps.pop()
    if nonexistent_fps:
        raise RuntimeError(', '.join(nonexistent_fps) + ' are not files.')

    if db_search_fps != None:
        nonexistent_fps = []
        for db_search_fp in db_search_fps:
            nonexistent_fps.append(utils.check_path(db_search_fp, return_str=True))
            if nonexistent_fps[-1] == None:
                nonexistent_fps.pop()
        if nonexistent_fps:
            raise RuntimeError(', '.join(nonexistent_fps) + ' are not files.')

    #Check that the input MGF files meet the minimum formatting requirements.
    unrecognized_mgf_fps = []
    for mgf_fp in mgf_fps:
        if not check_mgf(mgf_fp):
            unrecognized_mgf_fps.append(mgf_fp)
    if unrecognized_mgf_fps:
        raise RuntimeError(
            ', '.join(unrecognized_mgf_fps) + ' are not properly formatted MGF files. '
            'Properly formatted MGF files can be generated with the msconvert command, '
            'substituting your file for <FILE>: '
            'msconvert <FILE>.raw --mgf --filter "peakPicking vendor" '
            '--filter "titleMaker Run: <RunId>, Index: <Index>, Scan: <ScanNumber>" '
            '--filter "zeroSamples removeExtra"')

    #Filter db search PSMs to FDR.
    if fdr_cutoff:
        if db_search_fps == None:
            parser.error(
                'FDR screening requires database search results, '
                'specified with db_search_fps.')
        db_search_dfs = []
        filtered_db_search_fps = []
        for db_search_fp in db_search_fps:
            #Get each database search results table, with a column of PSM q-values added.
            db_search_df = pd.read_csv(db_search_fp, sep='\t', header=0)
            db_search_df = utils.calculate_qvalues(db_search_df)
            #Filter to PSMs with a q-value meeting the cutoff.
            db_search_df = db_search_df[db_search_df['psm_qvalue'] <= fdr_cutoff]
            #NOTE: The following line may be redundant, but in this file, 
            #do not assume uniquely paired Spectrum IDs and Scan Numbers, so be cautious.
            #If there are PSMs to the same spectrum, remove the one with the worse score.
            db_search_df = db_search_df.sort_values('psm_qvalue').drop_duplicates('ScanNum')
            db_search_dfs.append(db_search_df)
            #Write the filtered table of PSMs to file.
            filtered_db_search_fp = \
                os.path.splitext(db_search_fp)[0] + '.' + str(fdr_cutoff) + '.tsv'
            filtered_db_search_fps.append(filtered_db_search_fp)
            db_search_df.to_csv(
                filtered_db_search_fp, sep='\t', index=False, quoting=csv.QUOTE_NONE)
        del(db_search_df)
        #Henceforth, only consider the sets of FDR-controlled PSMs.
        db_search_fps = filtered_db_search_fps
        
    #To make DeepNovo training files, 
    #only those spectra with FDR-controlled PSMs from database search are retained in the MGF.
    if db_search_fps != None:
        filtered_mgf_fps = []
        for mgf_fp, db_search_fp in zip(mgf_fps, db_search_fps):
            db_search_df = pd.read_csv(db_search_fp, sep='\t', header=0)
            filtered_scans = db_search_df['ScanNum'].tolist()
            filtered_mgf_fp = os.path.splitext(mgf_fp)[0] + '.fdr_controlled.mgf'
            filtered_mgf_fps.append(filtered_mgf_fp)
            with open(mgf_fp) as in_f, open(filtered_mgf_fp, 'w') as out_f:
                spectrum_lines = []
                for line in in_f:
                    spectrum_lines.append(line)
                    #First, assume msconvert output with properly formatted title lines.
                    if 'TITLE=' == line[:6]:
                        scan_number = line.split(', Scan: ')[1].rstrip('\n')
                    #A scans line indicates that the MGF was reformatted by Postnovo.
                    #This assumes that db search was performed with the Postnovo-formatted MGF.
                    #Attempt to retrieve the scan number again.
                    if 'SCANS=' == line[:6]:
                        scan_number = line.replace('SCANS=', '').rstrip('\n')
                    if 'END IONS\n' == line:
                        #Only retain spectra with a db search PSM meeting FDR threshold.
                        if int(scan_number) in filtered_scans:
                            out_f.write(''.join(spectrum_lines))
                        spectrum_lines = []
        mgf_fps = filtered_mgf_fps

    #Ignore spectra with precursor mass below threshold, since Novor has problems with low masses.
    if min_mass > 0:
        high_mass_mgf_fps = []
        temp_high_mass_mgf_fps = []
        for mgf_fp in mgf_fps:
            low_mass_count = 0
            high_mass_mgf_fp = os.path.splitext(mgf_fp)[0] + '.' + str(min_mass) + 'Da.mgf'
            with open(mgf_fp) as in_f, open(high_mass_mgf_fp, 'w') as out_f:
                spectrum_lines = []
                for line in in_f:
                    spectrum_lines.append(line)
                    if 'PEPMASS=' == line[:8]:
                        #Recover the precursor mass, 
                        #making sure to separate the value from the intensity, if present.
                        pepmass = float(line.rstrip().split('PEPMASS=')[1].split(' ')[0])
                    if 'END IONS\n' == line:
                        if pepmass >= min_mass:
                            out_f.write(''.join(spectrum_lines))
                        else:
                            low_mass_count += 1
                        spectrum_lines = []
            if low_mass_count > 0:
                print(
                    os.path.basename(mgf_fp) + ' contains ' + str(low_mass_count) + 
                    ' spectra with precursor mass < ' + str(min_mass))
                high_mass_mgf_fps.append(high_mass_mgf_fp)
                #These are new files that will later be deleted.
                temp_high_mass_mgf_fps.append(high_mass_mgf_fp)
            else:
                subprocess.call(['rm', high_mass_mgf_fp])
                high_mass_mgf_fps.append(mgf_fp)
        mgf_fps = high_mass_mgf_fps

    #Check that there are enough spectra for a subsample of the given size.
    if subsample_size == None:
        subsampled_scans = None
    else:
        total_spectrum_count = 0
        for mgf_fp in mgf_fps:
            #Instead of slowly loading files into Python and parsing, count spectra with grep.
            p = subprocess.Popen(['grep', '-c', 'BEGIN IONS', mgf_fp], stdout=subprocess.PIPE)
            mgf_spectrum_count = p.communicate()[0]
            total_spectrum_count += int(mgf_spectrum_count)
        if subsample_size > total_spectrum_count:
            raise ValueError(
                'The subsample is greater than the number of spectra in the input MGF files: ' + 
                str(total_spectrum_count))
        #Identify spectra to subsample.
        subsampled_indices = np.sort(np.random.choice(
            range(1, total_spectrum_count + 1), subsample_size, replace=False))

    #Subsample and concatenate spectra from all input files.
    #Assume msconvert output with properly formatted title lines.
    if len(mgf_fps) == 1:
        if subsample_size == None:
            full_mgf_fp = mgf_fps[0]
        #Subsample spectra.
        else:
            temp_subsample_mgf_fp = os.path.splitext(out_mgf_fp)[0] + '.subsample_tmp.mgf'
            spectrum_count = 0
            selection_count = 0
            selection_index = subsampled_indices[selection_count]
            with open(mgf_fps[0]) as in_f, open(temp_subsample_mgf_fp, 'w') as out_f:
                for line in in_f:
                    if line == 'BEGIN IONS\n':
                        spectrum_count += 1
                    if spectrum_count == selection_index:
                        out_f.write(line)
                        if line == 'END IONS\n':
                            selection_count += 1
                            try:
                                selection_index = subsampled_indices[selection_count]
                            except IndexError:
                                #All spectra have been selected.
                                break
            full_mgf_fp = temp_subsample_mgf_fp
    else:
        if subsample_size == None:
            temp_concat_mgf_fp = os.path.splitext(out_mgf_fp)[0] + '.concat_tmp.mgf'
            with open(temp_concat_mgf_fp, 'w') as out_f:
                for mgf_fp in mgf_fps:
                    with open(mgf_fp) as in_f:
                        for line in in_f:
                            out_f.write(line)
            full_mgf_fp = temp_concat_mgf_fp
        #Subsample spectra and concatenate them.
        else:
            temp_subsample_concat_mgf_fp = \
                os.path.splitext(out_mgf_fp)[0] + '.subsample_concat_tmp.mgf'
            spectrum_count = 0
            selection_count = 0
            selection_index = subsampled_indices[selection_count]
            with open(temp_subsample_concat_mgf_fp, 'w') as out_f:
                #Loop through each dataset.
                for mgf_fp in mgf_fps:
                    with open(mgf_fp) as in_f:
                        for line in in_f:
                            if line == 'BEGIN IONS\n':
                                spectrum_count += 1
                            if spectrum_count == selection_index:
                                out_f.write(line)
                                if line == 'END IONS\n':
                                    selection_count += 1
                                    try:
                                        selection_index = subsampled_indices[selection_count]
                                    except IndexError:
                                        #All spectra have been selected.
                                        break
            full_mgf_fp = temp_subsample_concat_mgf_fp

    #If creating a file for DeepNovo training, get the peptide sequences
    #from database search PSMs that must be assigned to each spectrum.
    if db_search_fps != None:
        #First, concatenate database search results from all datasets under consideration, 
        #adding a column indicating the dataset corresponding to each PSM.
        concat_db_search_df = pd.DataFrame()
        for mgf_fp, db_search_fp in zip(mgf_fps, db_search_fps):
            db_search_df = pd.read_csv(db_search_fp, sep='\t', header=0)
            #Recover the run ID from a title line of the MGF.
            with open(mgf_fp) as f_in:
                f_in.readline()
                db_search_df['run'] = f_in.readline().split('TITLE=Run: ')[1].split(', Index: ')[0]
            db_search_df['ScanNum'] = db_search_df['ScanNum'].apply(str)
            concat_db_search_df = pd.concat([concat_db_search_df, db_search_df], ignore_index=True)
        concat_db_search_df.set_index(['run', 'ScanNum'], inplace=True)
        seqs_mgf_fp = os.path.splitext(out_mgf_fp)[0] + '.seqs_tmp.mgf'
        with open(full_mgf_fp) as in_f, open(seqs_mgf_fp, 'w') as out_f:
            for line in in_f:
                if 'TITLE=' == line[:6]:
                    split_line = line.split('TITLE=Run: ')[1]
                    run_id, split_line = split_line.split(', Index: ')
                    scan = split_line.split(', Scan: ')[1].rstrip('\n')
                    pep_seq = concat_db_search_df.loc[(run_id, scan)]['Peptide']
                    for postnovo_mod_symbol, deepnovo_mod_symbol in \
                        postnovo_deepnovo_training_mod_dict.items():
                        pep_seq = pep_seq.replace(postnovo_mod_symbol, deepnovo_mod_symbol)
                out_f.write(line)
                #Add the peptide sequence line at the end of the spectrum header.
                if 'RTINSECONDS=' == line[:12]:
                    out_f.write('SEQ=' + pep_seq + '\n')
        full_mgf_fp = seqs_mgf_fp

    #Reformat MGF files to be compatible with de novo sequencing tools.
    #The one exception here is that MGF files used for de novo sequence prediction by DeepNovo 
    #are modified in a temporary file before running DeepNovo to satisfy that tool's requirements, 
    #adding a null peptide sequence (Seq=A) as the last line of each spectrum header.
    new_index = 1
    new_scan = 1
    with open(full_mgf_fp) as in_f, open(out_mgf_fp, 'w') as out_f:
        for line in in_f:
            if line == 'BEGIN IONS\n':
                ms2_peak_lines = []
            elif 'TITLE=' == line[:6]:
                split_line = line.split('TITLE=Run: ')[1]
                run_id, split_line = split_line.split(', Index: ')
                old_index, split_line = split_line.split(', Scan: ')
                old_scan = split_line.rstrip('\n')
            elif 'PEPMASS=' == line[:8]:
                #Remove intensity data: DeepNovo only looks for the mass and not intensity.
                if ' ' in line:
                    pepmass_line = line.split(' ')[0] + '\n'
                else:
                    pepmass_line = line
            elif 'CHARGE=' == line[:7]:
                charge_line = line
            elif 'RTINSECONDS=' == line[:12]:
                rt_line = line
            elif 'SEQ=' == line[:4]:
                seq_line = line
            elif line == 'END IONS\n':
                #Avoid peptides without MS2 peaks.
                if len(ms2_peak_lines) > 0:
                    out_f.write('BEGIN IONS\n')
                    out_f.write(
                        'TITLE=Run: ' + run_id + ', Index: ' + str(new_index) + \
                            ', Old index: ' + old_index + ', Old scan: ' + old_scan + '\n')
                    out_f.write(pepmass_line)
                    out_f.write(charge_line)
                    out_f.write('SCANS=' + str(new_scan) + '\n')
                    out_f.write(rt_line)
                    try:
                        #Written in DeepNovo training files.
                        out_f.write(seq_line)
                    except NameError:
                        pass
                    out_f.write(''.join(ms2_peak_lines))
                    out_f.write(line)
                    new_index += 1
                    new_scan += 1
            else:
                #For DeepNovo training files, the first of these lines is the db search PSM seq.
                ms2_peak_lines.append(line)

    #Delete temp files created to store spectra with high confidence db search PSMs.
    if 'filtered_mgf_fps' in locals():
        for filtered_mgf_fp in filtered_mgf_fps:
            subprocess.call(['rm', filtered_mgf_fp])

    #Delete temp files from which low precursor mass spectra were removed.
    if 'temp_high_mass_mgf_fps' in locals():
        if len(temp_high_mass_mgf_fps) > 0:
            for temp_high_mass_mgf_fp in temp_high_mass_mgf_fps:
                subprocess.call(['rm', temp_high_mass_mgf_fp])

    #Delete temp file of subsampled and/or concatenated, msconvert-formatted spectra.
    if 'temp_subsample_mgf_fp' in locals():
        subprocess.call(['rm', temp_subsample_mgf_fp])
    if 'temp_concat_mgf_fp' in locals():
        subprocess.call(['rm', temp_concat_mgf_fp])
    if 'temp_subsample_concat_mgf_fp' in locals():
        subprocess.call(['rm', temp_subsample_concat_mgf_fp])

    #Delete temp file of msconvert-formatted spectra with an added header line of the peptide seq.
    if 'seqs_mgf_fp' in locals():
        subprocess.call(['rm', seqs_mgf_fp])

    return

def check_mgf(mgf_fp):
    '''
    Determine whether the input MGF has the minimum requirements for the format_mgf command.

    Parameters
    ----------
    mgf_fp : str
        MGF filepath.

    Returns
    -------
    recognized_format : bool
        False if the MGF file is not formatted properly.
    '''

    header_line_count = 0
    with open(mgf_fp) as in_f:
        #Check the first spectrum block.
        for line in in_f:
            if 'BEGIN IONS' == line[:10]:
                header_line_count += 1
            elif 'TITLE=' == line[:6]:
                #The TITLE line in the header must contain the run ID, spectrum index, and scan.
                #TITLE=Run: <RunId>, Index: <SPECTRUM INDEX>, Scan: <SCAN NUMBER>
                if ('Run: ' in line) and (', Index: ' in line) and (', Scan: ' in line):
                    header_line_count += 1
                else:
                    return False
                header_line_count += 1
            elif 'RTINSECONDS=' == line[:12]:
                header_line_count += 1
            elif 'PEPMASS=' == line[:8]:
                header_line_count += 1
            elif 'CHARGE=' == line[:7]:
                header_line_count += 1
            elif 'END IONS' == line[:8]:
                #All of the preceding header lines must be present, 
                #but a specific order is not required.
                if header_line_count == 6:
                    return True
                else:
                    return False

    raise RuntimeError('Each MGF input file is assumed to contain at least one spectrum.')

    return

def prepare_deepnovo(
    mgf_fp, 
    tensorflow_fp, 
    frag_resolution, 
    frag_mass_tols, 
    fixed_mods, 
    variable_mods, 
    bind_point, 
    time_limit, 
    mem):
    
    #Check mandatory input filepaths.
    if mgf_fp == None:
        raise RuntimeError('Specify the input MGF filepath.')
    else:
        utils.check_path(mgf_fp)

    if tensorflow_fp == None:
        raise RuntimeError('The filepath to a TensorFlow image must be specified.')
    else:
        utils.check_path(tensorflow_fp)

    if frag_resolution == None:
        raise RuntimeError(
            'Use the frag_resolution option to specify whether the dataset '
            'has fragmentation spectra with "low" or "high" resolution.')

    if frag_mass_tols:
        #Check that individually specified fragment mass tolerances assume allowable values.
        valid_frag_mass_tols = config.frag_mass_tol_dict['Low'] + config.frag_mass_tol_dict['High']
        for frag_mass_tol in frag_mass_tols:
            if frag_mass_tol not in valid_frag_mass_tols:
                raise RuntimeError(
                    'Choose from the following valid mass tolerances: '
                    'Low-resolution: ' + ', '.join(config.frag_mass_tol_dict['Low']) + ' Da; '
                    'High-resolution: ' + ', '.join(config.frag_mass_tol_dict['High']) + ' Da.')
        frag_mass_tols = sorted(list(set(frag_mass_tols)))
    else:
        frag_mass_tols = config.frag_mass_tol_dict[frag_resolution.capitalize()]

    #Collect list of amino acids with fixed modifications.
    fixed_mod_aas = []
    for denovogui_mod_code in fixed_mods:
        if denovogui_mod_code not in config.denovogui_deepnovo_config_mod_dict:
            raise RuntimeError(
                'The only mods supported by DeepNovo are: '
                ', '.join(config.denovogui_deepnovo_config_mod_dict.keys()))
        #An example of the notation of a modification in DeepNovo is Cmod.
        deepnovo_config_mod_code = config.denovogui_deepnovo_config_mod_dict[
            denovogui_mod_code]
        #Record the amino acid with the fixed modification, e.g., C.
        fixed_mod_aas.append(deepnovo_config_mod_code.replace('mod', ''))

    #Collect list of amino acids with variable modifications.
    variable_mod_aas = []
    for denovogui_mod_code in variable_mods:
        if denovogui_mod_code not in config.denovogui_deepnovo_config_mod_dict:
            raise RuntimeError(
                'The only mods supported by DeepNovo are: '
                ', '.join(config.denovogui_deepnovo_config_mod_dict.keys()))
        deepnovo_config_mod_code = config.denovogui_deepnovo_config_mod_dict[
            denovogui_mod_code]
        variable_mod_aas.append(deepnovo_config_mod_code.replace('mod', ''))

    #Check the validity of a user-specified bind point (directory).
    if bind_point:
        utils.check_path(bind_point)

    if time_limit:
        hrs = int(time_limit)
        mins = (time_limit * 60) % 60
        secs = (time_limit * 3600) % 60
        time_limit = '%02d:%02d:%02d' % (hrs, mins, secs)

    if mem != 'MaxMemPerNode':
        if float(mem) <= 0:
            raise RuntimeError('Memory allocated to the Slurm job must be positive.')

    return frag_mass_tols, fixed_mod_aas, variable_mod_aas, time_limit

#REMOVE: There are only two default modifications, to C and M.
#def check_mods(parser, mod_input):
#    '''    
#    Check user-defined fixed and variable modification arguments.

#    Parameters
#    ----------
#    parser : argparse ArgumentParser object
#    mod_input : list

#    Returns
#    -------
#    None
#    '''

#    unrecognized_mods = []
#    for mod in mod_input:
#        if mod not in config.denovogui_postnovo_mod_dict:
#            unrecognized_mods.append(mod)
#    if unrecognized_mods:
#        parser.error(
#            'The following mods are not recognized: {0}'.format(
#                ', '.join(unrecognized_mods)))

#    return

def train_deepnovo(
    mgf_fp, 
    container_fp, 
    user_bind_point, 
    frag_resolution, 
    frag_mass_tols, 
    fixed_mod_aas, 
    variable_mod_aas, 
    use_slurm, 
    partition, 
    time_limit, 
    cpus, 
    mem):
    '''
    Train DeepNovo at each fragment mass tolerance.

    Parameters
    ----------
    mgf_fp : str
        Filepath to MGF file, with format concordant with Postnovo format_mgf output.
    container_fp : str
        Filepath to TensorFlow container image.
    user_bind_point : str
        Singularity bind point within the scope of the TensorFlow container.
    frag_resolution : str
        Resolution of fragmentation spectra: "low" or "high".
    frag_mass_tols : list of str
        Fragment mass tolerances at which to run DeepNovo.
    fixed_mod_aas : list of str
        Amino acids with fixed modifications.
    variable_mod_aas : list of str
        Amino acids with variable modifications.
    use_slurm : boolean
        True when running Slurm job on compute cluster.
    partition : str
        Name of Slurm partition to use on compute cluster.
    time_limit : str
        Time limit of Slurm job.
    cpus : int
        Number of CPUs to use in Slurm job on compute cluster.
    mem : float
        Memory allocated to Slurm job on compute cluster.

    Returns
    -------
    None
    '''

    file_prefix = os.path.splitext(mgf_fp)[0]
    dataset_name = os.path.basename(file_prefix)
    train_mgf_fp = file_prefix + '.train.mgf'
    valid_mgf_fp = file_prefix + '.valid.mgf'
    test_mgf_fp = file_prefix + '.test.mgf'

    if (os.path.exists(train_mgf_fp)) and \
        (os.path.exists(valid_mgf_fp)) and \
        (os.path.exists(test_mgf_fp)):
        #Do not make new files if properly named files already exist in the input directory.
        print('Existing training, validation and testing MGFs will be used.')
    elif not (
        (os.path.exists(train_mgf_fp) == False) and \
            (os.path.exists(valid_mgf_fp) == False) and \
            (os.path.exists(test_mgf_fp) == False)):
        raise RuntimeError(
            'Some but not all of train.mgf, valid.mgf, and test.mgf files already exist. '
            'Postnovo will not proceed unless none exist (new ones created) or all already exist.')
    else:
        #Construct the training, validation and testing datasets from the full dataset.
        validation_proportion = 0.18
        test_proportion = 0.02

        #Count the number of spectra in the input dataset.
        p = subprocess.Popen(['grep', '-c', 'BEGIN IONS', mgf_fp], stdout=subprocess.PIPE)
        spectrum_count = int(p.communicate()[0])

        #Select spectra for training, validation and testing files.
        validation_count = int(validation_proportion * spectrum_count)
        test_count = int(test_proportion * spectrum_count)
        train_count = spectrum_count - validation_count - test_count
        #Ensure spectrum selection is reproducible.
        np.random.seed(1)
        nontrain_sample_indices = np.random.choice(
            range(1, spectrum_count + 1), 
            validation_count + test_count, 
            replace=False)
        validation_sample_indices = nontrain_sample_indices[:validation_count]
        test_sample_indices = nontrain_sample_indices[validation_count:]

        #Partition the spectra from the input MGF.
        with open(mgf_fp) as in_f, \
            open(train_mgf_fp, 'w') as train_out_f, \
            open(valid_mgf_fp, 'w') as valid_out_f, \
            open(test_mgf_fp, 'w') as test_out_f:
            i = 1
            for line in in_f:
                if 'BEGIN IONS' == line[:10]:
                    if i in validation_sample_indices:
                        current_out_f = valid_out_f
                    elif i in test_sample_indices:
                        current_out_f = test_out_f
                    else:
                        current_out_f = train_out_f
                    i += 1
                current_out_f.write(line)

    #Loop through each mass tolerance.
    try:
        #Try to load a Singularity environment module.
        with open(os.devnull, 'w') as null_f:
            subprocess.call('module load singularity', shell=True, stderr=null_f)
    except FileNotFoundError:
        pass

    if not use_slurm:
        #Data needed to remove copied MGF files upon process completion.
        file_removal_data = []

    for frag_mass_tol in frag_mass_tols:
        target_deepnovo_dir = os.path.join(
            config.deepnovo_dir, 'DeepNovo.' + frag_resolution + '.' + frag_mass_tol)
        #Make a DeepNovo directory for each fragment mass tolerance, as each has its own model.
        if not os.path.isdir(target_deepnovo_dir):
            os.mkdir(target_deepnovo_dir)
            #Copy DeepNovo program files and template files to the new directory.
            subprocess.call(['cp'] + config.deepnovo_program_fps + [target_deepnovo_dir])

        #Make a train directory for the training files.
        target_train_dir = os.path.join(target_deepnovo_dir, 'train')
        if not os.path.isdir(target_train_dir):
            os.mkdir(target_train_dir)
        #Copy training and validation files to the folder.
        #The testing file is not needed for the training routine.
        subprocess.call(['cp', train_mgf_fp, valid_mgf_fp, test_mgf_fp, target_train_dir])

        if user_bind_point == None:
            #Create a bind point for the proper DeepNovo directory 
            #within the Singularity container.
            bind_point = os.path.join(
                os.path.dirname(container_fp), os.path.basename(target_deepnovo_dir))
        else:
            bind_point = user_bind_point
        if not os.path.isdir(bind_point):
            os.mkdir(bind_point)

        #Train DeepNovo in a Singularity container.
        run_id = dataset_name + '.' + frag_resolution + '.' + frag_mass_tol
        os.chdir(bind_point)
        #Compile Cython modules.
        encountered_cython_compile_err = False
        if os.path.exists(os.path.join(target_deepnovo_dir, 'deepnovo_cython_modules.c')) and \
            os.path.exists(os.path.join(target_deepnovo_dir, 'deepnovo_cython_modules.so')):
            pass
        else:
            p = subprocess.Popen([
                'singularity', 
                'exec', 
                '--bind', 
                target_deepnovo_dir + ':' + bind_point, 
                container_fp, 
                'python', 
                os.path.join(bind_point, 'deepnovo_cython_setup.py'), 
                'build_ext', 
                '--inplace'], stderr=subprocess.PIPE)
            err = p.communicate()[1]
            if err.decode() != '':
                encountered_cython_compile_err = True

        #DeepNovo options must be specified in a config file.
        template_config_fp = os.path.join(config.deepnovo_dir, 'deepnovo_config_template.py')
        target_config_fp = os.path.join(target_deepnovo_dir, 'deepnovo_config.py')
        with open(template_config_fp) as in_f, open(target_config_fp, 'w') as out_f:
            in_dataset_block = False
            for line in in_f:
                if in_dataset_block:
                    if 'fixed_mod_list = ' == line[:17]:
                        out_f.write('fixed_mod_list = ' + str(fixed_mod_aas) + '\n')
                    elif 'var_mod_list = ' == line[:15]:
                        out_f.write('var_mod_list = ' + str(variable_mod_aas) + '\n')
                    #No reference to the training MGF files are needed -- 
                    #all that matter is that the model is in the 'train' directory.
                    elif 'input_file_train = ' == line[:19]:
                        os.path.join('train', os.path.basename(train_mgf_fp))
                        out_f.write(
                            'input_file_train = \'' + 
                            os.path.join('train', os.path.basename(train_mgf_fp)) + '\'\n')
                    elif 'input_file_valid = ' == line[:19]:
                        out_f.write(
                            'input_file_valid = \'' + 
                            os.path.join('train', os.path.basename(valid_mgf_fp)) + '\'\n')
                    elif 'decode_test_file = ' == line[:19]:
                        out_f.write(
                            'decode_test_file = \'' + 
                            os.path.join('train', os.path.basename(test_mgf_fp)) + '\'\n')
                    elif 'decode_output_file = ' == line[:21]:
                        #This variable is used de novo sequence output in DeepNovo prediction mode.
                        pass
                    else:
                        out_f.write(line)
                elif '# DATASET' == line[:9]:
                    in_dataset_block = True
                    out_f.write(line)
                elif 'SPECTRUM_RESOLUTION = ' == line[:22]:
                    res = 1 / float(frag_mass_tol)
                    if res > 10:
                        #Ex. 0.03 Da -> 33
                        out_f.write('SPECTRUM_RESOLUTION = ' + str(int(res)) + '\n')
                    else:
                        #Ex. 0.3 Da -> 3.3
                        out_f.write('SPECTRUM_RESOLUTION = ' + str(round(res, 1)) + '\n')
                else:
                    out_f.write(line)

        if not encountered_cython_compile_err:
            input_train_mgf_fp = os.path.join(
                target_deepnovo_dir, 
                os.path.join('train', os.path.basename(train_mgf_fp)))
            input_valid_mgf_fp = os.path.join(
                target_deepnovo_dir, 
                os.path.join('train', os.path.basename(valid_mgf_fp)))
            input_test_mgf_fp = os.path.join(
                target_deepnovo_dir, 
                os.path.join('train', os.path.basename(test_mgf_fp)))
            if use_slurm:
                sbatch_script_fp = os.path.join(bind_point, run_id + '.train.sbatch')
                with open(sbatch_script_fp, 'w') as out_f:
                    out_f.write('#!/bin/bash\n')
                    out_f.write('#SBATCH --job-name=' + run_id + '\n')
                    out_f.write('#SBATCH --output=' + run_id + '.deepnovo_train.out\n')
                    out_f.write('#SBATCH --error=' + run_id + '.deepnovo_train.err\n')
                    out_f.write('#SBATCH --time=' + time_limit + '\n')
                    out_f.write('#SBATCH --partition=' + partition + '\n')
                    out_f.write('#SBATCH --nodes=1\n')
                    out_f.write('#SBATCH --cpus-per-task=' + str(cpus) + '\n')
                    out_f.write('#SBATCH --mem=' + mem + 'G\n')
                    out_f.write('\n')
                    #Try to load a Singularity environment module.
                    out_f.write('module load singularity\n')
                    out_f.write(' '.join([
                        'singularity', 
                        'exec', 
                        '--bind', 
                        target_deepnovo_dir + ':' + bind_point, 
                        container_fp, 
                        'python', 
                        os.path.join(bind_point, 'deepnovo_main.py'), 
                        '--train_dir', 
                        'train', 
                        '--train\n']))
                    out_f.write('rm ' + input_train_mgf_fp + '\n')
                    out_f.write('rm ' + input_valid_mgf_fp + '\n')
                    out_f.write('rm ' + input_test_mgf_fp + '\n')
                subprocess.Popen(['sbatch', sbatch_script_fp])
            else:
                p = subprocess.Popen([
                    'nohup', 
                    'singularity', 
                    'exec', 
                    '--bind', 
                    target_deepnovo_dir + ':' + bind_point, 
                    container_fp, 
                    'python', 
                    os.path.join(bind_point, 'deepnovo_main.py'), 
                    '--train_dir', 
                    'train', 
                    '--train'])
                file_removal_data.append(
                    (p, [input_train_mgf_fp, input_valid_mgf_fp, input_test_mgf_fp]))

                #Rename nohup.out to reflect the dataset name and fragment mass tolerance.
                stdout_fp = os.path.join(bind_point, 'nohup.out')
                new_stdout_fp = os.path.join(
                    bind_point, 
                    dataset_name + '.' + frag_resolution + '.' + frag_mass_tol + \
                        '.train.nohup.out')
                while True:
                    if os.path.exists(stdout_fp):
                        os.rename(stdout_fp, new_stdout_fp)
                        break

    if encountered_cython_compile_err:
        print(
            'DeepNovo Cython modules could not be compiled. '
            'Try using Python 2.7 to compile in each DeepNovo directory. '
            'The necessary "Cython" package may not be installed in Python -- '
            'to install, run the command, "pip install --user Cython". '
            'To compile the DeepNovo Cython code, run the following command in each directory, '
            '"python deepnovo_cython_setup.py build_ext --inplace".')

    if not use_slurm:
        #No DeepNovo process has been found to be completed.
        completed_process_index = None
        while file_removal_data:
            time.sleep(2)
            #Cycle through the running processes.
            for i, process_input_mgf_fps_tuple in enumerate(file_removal_data):
                print(process_input_mgf_fps_tuple, flush=True)
                print(process_input_mgf_fps_tuple[0].poll(), flush=True)
                #The poll method returns None if the process is still running.
                if process_input_mgf_fps_tuple[0].poll() != None:
                    #Remove the copied input MGF files for the fragment mass tolerance.
                    for input_mgf_fp in process_input_mgf_fps_tuple[1]:
                        os.remove(input_mgf_fp)
                    completed_process_index = i
                    break
            if completed_process_index != None:
                #Update the list of running processes.
                file_removal_data.pop(completed_process_index)
                completed_process_index = None

    return

def predict_deepnovo(
    mgf_fp, 
    out_dir, 
    container_fp, 
    user_bind_point, 
    frag_resolution, 
    frag_mass_tols, 
    fixed_mod_aas, 
    variable_mod_aas, 
    use_slurm, 
    partition, 
    time_limit, 
    cpus, 
    mem):
    '''
    Predict DeepNovo de novo sequences at each fragment mass tolerance.

    Parameters
    ----------
    mgf_fp : str
        Filepath to MGF file, with format concordant with Postnovo format_mgf output.
    out_dir : str
        DeepNovo output path.
    container_fp : str
        Filepath to TensorFlow container image.
    user_bind_point : str
        Singularity bind point within the scope of the TensorFlow container.
    frag_resolution : str
        Resolution of fragmentation spectra: "low" or "high".
    frag_mass_tols : list of str
        Individual fragment mass tolerances at which to run DeepNovo, instead of a predefined set.
    fixed_mod_aas : list of str
        Amino acids with fixed modifications.
    variable_mod_aas : list of str
        Amino acids with variable modifications.
    use_slurm : boolean
        True when running Slurm job on compute cluster.
    partition : str
        Name of Slurm partition to use on compute cluster.
    time_limit : str
        Time limit of Slurm job.
    cpus : int
        Number of CPUs to use in Slurm job on compute cluster.
    mem : float
        Memory allocated to Slurm job on compute cluster.

    Returns
    -------
    None
    '''

    #Add a placeholder peptide sequence line to the MGF file to satisfy DeepNovo.
    mgf_basename = os.path.basename(mgf_fp)
    dataset_name = os.path.splitext(mgf_basename)[0]
    master_mgf_fp = mgf_fp + '.tmp'
    with open(mgf_fp) as in_f, open(master_mgf_fp, 'w') as out_f:
        for line in in_f:
            if 'RTINSECONDS=' == line[:12]:
                out_f.write(line)
                out_f.write('SEQ=A\n')
            else:
                out_f.write(line)

    if not use_slurm:
        try:
            #Try to load a Singularity environment module.
            with open(os.devnull, 'w') as null_f:
                subprocess.call('module load singularity', shell=True, stderr=null_f)
        except FileNotFoundError:
            pass

        #Data needed to remove copied MGF files upon process completion.
        file_removal_data = []

    for frag_mass_tol in frag_mass_tols:
        target_deepnovo_dir = os.path.join(
            config.deepnovo_dir, 'DeepNovo.' + frag_resolution + '.' + frag_mass_tol)
        #Create a prediction directory.
        target_predict_dir = os.path.join(target_deepnovo_dir, 'predict')
        if not os.path.isdir(target_predict_dir):
            os.mkdir(target_predict_dir)
        #Copy the MGF file to the proper prediction directory.
        input_mgf_fp = os.path.join(target_predict_dir, mgf_basename)
        shutil.copy2(master_mgf_fp, input_mgf_fp)

        #DeepNovo options must be specified in a config file.
        template_config_fp = os.path.join(config.deepnovo_dir, 'deepnovo_config_template.py')
        #The DeepNovo config file is first constructed 
        #and then renamed to deepnovo_config.py when DeepNovo is run.
        target_config_fp = os.path.join(
            target_deepnovo_dir, dataset_name + '.' + frag_mass_tol + '.deepnovo_config.py')
        with open(template_config_fp) as in_f, open(target_config_fp, 'w') as out_f:
            in_dataset_block = False
            for line in in_f:
                if in_dataset_block:
                    if 'fixed_mod_list = ' == line[:17]:
                        out_f.write('fixed_mod_list = ' + str(fixed_mod_aas) + '\n')
                    elif 'var_mod_list = ' == line[:15]:
                        out_f.write('var_mod_list = ' + str(variable_mod_aas) + '\n')
                    #No reference to the training MGF files are needed -- 
                    #all that matters is that the model is in the "train" directory.
                    elif 'input_file_train = ' == line[:19]:
                        pass
                    elif 'input_file_valid = ' == line[:19]:
                        pass
                    elif 'decode_test_file = ' == line[:19]:
                        out_f.write('decode_test_file = \'predict/' + mgf_basename + '\'\n')
                    elif 'decode_output_file = ' == line[:21]:
                        out_f.write(
                            'decode_output_file = \'predict/' + dataset_name + '.' + 
                            frag_mass_tol + '.tsv\'\n')
                    else:
                        out_f.write(line)
                elif '# DATASET' == line[:9]:
                    in_dataset_block = True
                    out_f.write(line)
                elif 'SPECTRUM_RESOLUTION = ' == line[:22]:
                    res = 1 / float(frag_mass_tol)
                    if res > 10:
                        #Ex. 0.03 Da -> 33
                        out_f.write('SPECTRUM_RESOLUTION = ' + str(int(res)) + '\n')
                    else:
                        #Ex. 0.3 Da -> 3.3
                        out_f.write('SPECTRUM_RESOLUTION = ' + str(round(res, 1)) + '\n')
                else:
                    out_f.write(line)

        #Create a bind point for the proper DeepNovo directory within the Singularity container.
        if user_bind_point == None:
            bind_point = os.path.join(
                os.path.dirname(container_fp), os.path.basename(target_deepnovo_dir))
        else:
            bind_point = user_bind_point
        if not os.path.isdir(bind_point):
            os.mkdir(bind_point)

        #Run DeepNovo in a Singularity container.
        run_id = dataset_name + '.' + frag_mass_tol
        deepnovo_output_basename = \
            dataset_name + '.' + frag_mass_tol + '.tsv'
        deepnovo_output_fp = os.path.join(
            target_deepnovo_dir, os.path.join('predict', deepnovo_output_basename))
        new_deepnovo_output_fp = os.path.join(out_dir, deepnovo_output_basename)
        os.chdir(bind_point)
        if use_slurm:
            sbatch_script_fp = os.path.join(bind_point, run_id + '.predict.sbatch')
            with open(sbatch_script_fp, 'w') as out_f:
                out_f.write('#!/bin/bash\n')
                out_f.write('#SBATCH --job-name=' + run_id + '\n')
                out_f.write('#SBATCH --output=' + run_id + '.deepnovo_predict.out\n')
                out_f.write('#SBATCH --error=' + run_id + '.deepnovo_predict.err\n')
                out_f.write('#SBATCH --time=' + time_limit + '\n')
                out_f.write('#SBATCH --partition=' + partition + '\n')
                out_f.write('#SBATCH --nodes=1\n')
                out_f.write('#SBATCH --cpus-per-task=' + str(cpus) + '\n')
                out_f.write('#SBATCH --mem=' + mem + 'G\n')
                out_f.write('\n')
                #Try to load a Singularity environment module.
                out_f.write('module load singularity\n')
                out_f.write('cp ' + target_config_fp + ' ' + \
                    os.path.join(target_deepnovo_dir, 'deepnovo_config.py') + '\n')
                out_f.write(' '.join([
                    'singularity', 
                    'exec', 
                    '--bind', 
                    target_deepnovo_dir + ':' + bind_point + ',' + out_dir + ':' + bind_point, 
                    container_fp, 
                    'python', 
                    os.path.join(bind_point, 'deepnovo_main.py'), 
                    '--train_dir', 
                    'train', 
                    '--decode', 
                    '--beam_search', 
                    '--beam_size', 
                    '20\n']))
                out_f.write('rm ' + target_config_fp + '\n')
                #Move the output file to the user-specified output directory.
                out_f.write('mv ' + deepnovo_output_fp + ' ' + new_deepnovo_output_fp + '\n')
                #Remove the MGF input file copied for use with the fragment mass tolerance.
                out_f.write('rm ' + input_mgf_fp + '\n')
            subprocess.Popen(['sbatch', sbatch_script_fp])
        else:
            shutil.copy2(target_config_fp, os.path.join(target_deepnovo_dir, 'deepnovo_config.py'))
            #Run on the present server.
            nohup_f = open(
                os.path.join(bind_point, dataset_name + '.' + frag_mass_tol + '.nohup.out'), 'w')
            p = subprocess.Popen([
                'nohup', 
                'singularity', 
                'exec', 
                '--bind', 
                target_deepnovo_dir + ':' + bind_point + ',' + out_dir + ':' + bind_point, 
                container_fp, 
                'python', 
                os.path.join(bind_point, 'deepnovo_main.py'), 
                '--train_dir', 
                'train', 
                '--decode', 
                '--beam_search', 
                '--beam_size', 
                '20'], stdout=nohup_f, stderr=nohup_f)
            file_removal_data.append((p, input_mgf_fp))

            #Move the DeepNovo output from the default destination to the output directory.
            while True:
                if os.path.exists(deepnovo_output_fp):
                    os.rename(deepnovo_output_fp, new_deepnovo_output_fp)
                    break

        os.chdir(config.postnovo_dir)

    if not use_slurm:
        #No DeepNovo process has been found to be completed.
        completed_process_index = None
        while file_removal_data:
            time.sleep(2)
            #Cycle through the running processes.
            for i, process_input_mgf_fp_duple in enumerate(file_removal_data):
                #The poll method returns None if the process is still running.
                if process_input_mgf_fp_duple[0].poll() != None:
                    #Remove the copied input MGF file for the fragment mass tolerance.
                    os.remove(process_input_mgf_fp_duple[1])
                    completed_process_index = i
                    break
            if completed_process_index != None:
                #Update the list of running processes.
                file_removal_data.pop(completed_process_index)
                completed_process_index = None

    os.remove(master_mgf_fp)

    return

def inspect_args(args):
    '''
    Inspects user arguments for predict, test and train subcommands, assigning global variables.

    Parameters
    ----------
    args : argparse NameSpace object
        Command line arguments.

    Returns
    -------
    None
    '''

    #Retraining the Postnovo random forest models requires minimal input 
    #beside fragment mass resolution.
    config.globals['De Novo Algorithms'] = ['Novor', 'PepNovo']

    if args.frag_resolution == None:
        raise RuntimeError('Please use the frag_resolution argument ("low" or "high").')
    config.globals['Fragment Mass Resolution'] = args.frag_resolution.capitalize()
    config.globals['Postnovo Training Directory'] = postnovo_train_dir = \
        config.postnovo_train_dir_dict[config.globals['Fragment Mass Resolution']]
    config.globals['Fragment Mass Tolerances'] = config.frag_mass_tol_dict[
        config.globals['Fragment Mass Resolution']]
    
    #This variable is later set to true in this function if the user specified "test_plots".
    #The initialization here is needed for the "retrain" option.
    config.globals['Make Test Plots'] = False

    config.globals['Leave-One-Out Data Filepath'] = os.path.join(
        config.globals['Postnovo Training Directory'], config.BINNED_SCORES_FILENAME)
    config.globals['Leave One Out'] = False
    if 'leave_one_out' in args:
        if args.leave_one_out:
            config.globals['Leave One Out'] = True

    if 'plot_feature_importance' in args:
        if args.plot_feature_importance:
            config.globals['Plot Feature Importance'] = True
        else:
            config.globals['Plot Feature Importance'] = False

    if args.cpus > cpu_count() or args.cpus < 1:
        raise RuntimeError(str(cpu_count()) + ' CPUs are available.')
    config.globals['CPU Count'] = args.cpus

    config.globals['Feature Set ID'] = args.feature_set
    config.globals['Model Features Dict'] = config.model_features_dicts[args.feature_set]
    #Fragment mass tolerances and fragment mass tolerance agreement 
    #are features of the Postnovo random forest models.
    for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
        for rf_model_name in config.globals['Model Features Dict']:
            config.globals['Model Features Dict'][rf_model_name].append(frag_mass_tol)
            #Fragment mass tolerance agreement features are not used in Feature Sets 0, 1, and 2.
            if config.globals['Feature Set ID'] >= 3:
                config.globals['Model Features Dict'][rf_model_name].append(frag_mass_tol + ' Match Value')
        config.feature_group_dict['Fragment Mass Tolerances'].append(frag_mass_tol)
        if config.globals['Feature Set ID'] >= 3:
            config.feature_group_dict['Fragment Mass Tolerance Matches'].append(
                frag_mass_tol + ' Match Value')

    if args.quiet:
        config.globals['Verbose'] = False
    else:
        config.globals['Verbose'] = True

    config.globals['Retrain'] = False
    if 'retrain' in args:
        if args.retrain:
            config.globals['Retrain'] = True
            if args.deepnovo:
                config.globals['De Novo Algorithms'].append('DeepNovo')
            config.globals['Postnovo Training Record Filepath'] = os.path.join(
                postnovo_train_dir, config.POSTNOVO_TRAIN_RECORD_FILENAME)
            #Check that all of the recorded training files actually exist.
            if os.path.exists(config.globals['Postnovo Training Record Filepath']):
                for train_dataset_name in pd.read_csv(
                    config.globals['Postnovo Training Record Filepath'], sep='\t', header=0)[
                        'Dataset'].tolist():
                    if not os.path.exists(
                        os.path.join(postnovo_train_dir, train_dataset_name + '.tsv')):
                        raise RuntimeError(
                            'Not all of the Postnovo training files recorded in ' + 
                            config.globals['Postnovo Training Record Filepath'] + 
                            ' were found in ' + 
                            postnovo_train_dir)
            if config.globals['Leave One Out'] or config.globals['Plot Feature Importance']:
                if args.out == None:
                    raise RuntimeError('Please specify an output directory using --out.')
                else:
                    if os.path.isdir(args.out):
                        config.globals['Output Directory'] = args.out
                    else:
                        raise RuntimeError(args.out + ' is not a directory.')
            return

    if not args.mgf:
        raise RuntimeError('A properly formatted MGF file must be provided.')
    #The full filepath must be provided.
    utils.check_path(args.mgf)
    config.globals['MGF Filepath'] = args.mgf
    config.globals['Dataset Name'] = os.path.splitext(os.path.basename(args.mgf))[0]
    config.globals['Input Directory'] = os.path.dirname(args.mgf)

    #If the MaRaCluster file is not specified, 
    #assume it has the default name in the directory of the MGF file.
    if not args.clusters:
        args.clusters = os.path.join(
            config.globals['Input Directory'], 'MaRaCluster.clusters_p2.tsv')
    utils.check_path(args.clusters)
    config.globals['Clusters Filepath'] = args.clusters

    #Check validity of output directory.
    #If this was not provided by the user, it is assigned as the MGF file's directory.
    if args.out == None:
        config.globals['Output Directory'] = os.path.dirname(config.globals['MGF Filepath'])
    else:
        if os.path.isdir(args.out):
            config.globals['Output Directory'] = args.out
        else:
            raise RuntimeError(args.out + ' is not a directory.')

    config.globals['Precursor Mass Tolerance'] = args.precursor_mass_tol
    if args.frag_method == None:
        raise RuntimeError('Please use the frag_method argument ("CID" or "HCD").')
    config.globals['Fragmentation Method'] = args.frag_method
    if config.globals['Mode'] == 'train':
        if not os.path.isdir(postnovo_train_dir):
            os.mkdir(postnovo_train_dir)
        config.globals['Postnovo Training Dataset Filepath'] = os.path.join(
            postnovo_train_dir, config.globals['Dataset Name'] + '.tsv')
        if os.path.exists(config.globals['Postnovo Training Dataset Filepath']):
            raise RuntimeError(
                'A training dataset of the same name already exists. '
                'If this is truly a new training dataset, change its name to proceed.')
        config.globals['Postnovo Training Record Filepath'] = os.path.join(
            postnovo_train_dir, config.POSTNOVO_TRAIN_RECORD_FILENAME)
        #Check that all of the recorded training files actually exist.
        if os.path.exists(config.globals['Postnovo Training Record Filepath']):
            for train_dataset_name in pd.read_csv(
                config.globals['Postnovo Training Record Filepath'], sep='\t', header=0)[
                    'Dataset'].tolist():
                if not os.path.exists(
                    os.path.join(postnovo_train_dir, train_dataset_name + '.tsv')):
                    raise RuntimeError(
                        'Not all of the Postnovo training files recorded in ' + 
                        config.globals['Postnovo Training Record Filepath'] + 
                        ' were found in ' + 
                        postnovo_train_dir)

    #Novor, PepNovo+, and DeepNovo output filenames must have a first part identical to the MGF.
    if args.filename == None:
        if args.denovogui:
            config.globals['Run DeNovoGUI'] = True
        else:
            raise RuntimeError(
                'Novor and PepNovo+ files were not specified by "filename". '
                'DeNovoGUI should be run with the "denovogui" flag to generate these files.')
        config.globals['MGF Filename'] = os.path.splitext(os.path.basename(args.mgf))[0]
    else:
        config.globals['Run DeNovoGUI'] = False
        config.globals['MGF Filename'] = args.filename
        #Check for the required Novor and PepNovo+ input files.
        missing_files = []
        for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
            try:
                missing_file = utils.check_path(
                    config.globals['MGF Filename'] + '.' + frag_mass_tol + '.novor.csv', 
                    config.globals['Input Directory'], 
                    return_str=True)
                if missing_file != None:
                    missing_files.append(missing_file)
            except TypeError:
                pass
            try:
                missing_file = utils.check_path(
                    config.globals['MGF Filename'] + '.' + frag_mass_tol + '.mgf.out', 
                    config.globals['Input Directory'], 
                    return_str=True)
                if missing_file != None:
                    missing_files.append(missing_file)
            except TypeError:
                pass
                
        if missing_files:
            for missing_file in missing_files:
                if missing_file != None:
                    print(missing_file + ' was not found.')
            raise RuntimeError('To generate Novor and PepNovo+ files, use the flag, --denovogui.')

        for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
            novor_fp = os.path.join(
                config.globals['Input Directory'], 
                config.globals['MGF Filename'] + '.' + frag_mass_tol + '.novor.csv')
            file_mass_tol_line = pd.read_csv(novor_fp, nrows = 12).iloc[11][0]
            file_mass_tol = file_mass_tol_line.strip('# fragmentIonErrorTol = ').strip('Da')
            if frag_mass_tol != file_mass_tol:
                raise RuntimeError(
                    'Novor files do not have the fragment mass tolerance ' + 
                    'asserted in the filename.')

    if args.deepnovo:
        #If the DeepNovo flag was used, check for the needed files.
        config.globals['De Novo Algorithms'].append('DeepNovo')
        missing_files = []
        for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
            try:
                missing_file = utils.check_path(
                    config.globals['MGF Filename'] + '.' + frag_mass_tol + '.tsv', 
                    config.globals['Input Directory'], 
                    return_str=True)
                if missing_file != None:
                    missing_files.append(missing_filename)
            except TypeError:
                pass

        if missing_files:
            for missing_file in missing_files:
                if missing_file != None:
                    print(missing_file + ' was not found.')
            raise RuntimeError(
                'To generate DeepNovo files, use the subcommand, postnovo predict_deepnovo.')

    #Record the filepaths of Novor, PepNovo+, and DeepNovo output files.
    config.globals['Novor Output Filepaths'] = []
    config.globals['PepNovo Output Filepaths'] = []
    if 'DeepNovo' in config.globals['De Novo Algorithms']:
        config.globals['DeepNovo Output Filepaths'] = []
    if args.filename == None:
        dir = config.globals['Output Directory']
    else:
        #These files already exist in the directory of the MGF file.
        dir = config.globals['Input Directory']
    for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
        config.globals['Novor Output Filepaths'].append(
            os.path.join(
                dir, config.globals['MGF Filename'] + '.' + frag_mass_tol + '.novor.csv'))
        config.globals['PepNovo Output Filepaths'].append(
            os.path.join(
                dir, config.globals['MGF Filename'] + '.' + frag_mass_tol + '.mgf.out'))
        if 'DeepNovo' in config.globals['De Novo Algorithms']:
            config.globals['DeepNovo Output Filepaths'].append(
                os.path.join(
                    dir, config.globals['MGF Filename'] + '.' + frag_mass_tol + '.tsv'))

    if (config.globals['Mode'] == 'test' or config.globals['Mode'] == 'train'):
        #Test and train modes require a reference FASTA file.
        if args.ref_fasta == None:
            raise RuntimeError(
                'Test and train modes require the option, ref_fasta.')
        #Test and train modes require MSGF+ formatted database search PSMs.
        if (args.db_search == None) and (args.msgf == False):
            raise RuntimeError(
                'Test and train modes require one of the options, db_search or msgf.')

    #Record the fixed and variable modifications, 
    #translating the ID string used in DeNovoGUI to the ID string used in Postnovo.
    config.globals['Fixed Modifications'] = []
    for denovogui_mod_code in args.fixed_mods:
        config.globals['Fixed Modifications'].append(
            config.denovogui_postnovo_mod_dict[denovogui_mod_code])
    config.globals['Variable Modifications'] = []
    for denovogui_mod_code in args.variable_mods:
        config.globals['Variable Modifications'].append(
            config.denovogui_postnovo_mod_dict[denovogui_mod_code])

    if 'test_plots' in args:
        if args.test_plots:
            config.globals['Make Test Plots'] = True
            config.globals['Reported Binary Classification Statistics Filepath'] = os.path.join(
                config.globals['Output Directory'], 'reported_classification_stats.tsv')
            if os.path.exists(
                config.globals['Reported Binary Classification Statistics Filepath']):
                os.remove(config.globals['Reported Binary Classification Statistics Filepath'])
            config.globals['Binned Scores Filepath'] = os.path.join(
                config.globals['Output Directory'], config.BINNED_SCORES_FILENAME)

    if 'stop_before_training' in args:
        if args.stop_before_training:
            config.globals['Stop Before Training'] = True
        else:
            config.globals['Stop Before Training'] = False

    if 'min_len' in args:
        if args.min_len < config.DEFAULT_MIN_LEN:
            raise RuntimeError(
                'min_len must be >= {0} amino acids.'.format(config.DEFAULT_MIN_LEN))
        config.globals['Minimum Postnovo Sequence Length'] = args.min_len
    else:
        config.globals['Minimum Postnovo Sequence Length'] = config.DEFAULT_MIN_LEN

    if 'min_prob' in args:
        if not 0 <= args.min_prob < 1:
            raise RuntimeError('min_prob must be between 0 and 1.')
        config.globals['Minimum Postnovo Sequence Probability'] = args.min_prob

    if 'ref_fasta' in args:
        utils.check_path(args.ref_fasta, config.globals['Input Directory'])
        config.globals['Reference Fasta Filepath'] = os.path.join(
            config.globals['Input Directory'], args.ref_fasta)
        if not 0 < args.fdr_cutoff <= 1:
            raise RuntimeError('fdr_cutoff must be > 0 and <= 1.')
        config.globals['FDR Cutoff'] = args.fdr_cutoff

    #MSGF+ can be run, or MSGF+ formatted output can be provided.
    config.globals['Reconcile Spectrum IDs'] = False
    config.globals['Run MSGF'] = False
    #Only the "test" and "train" modes have the "msgf" argument, 
    #so all of the following global variables that are set are only applicable to those modes.
    if 'msgf' in args:
        if args.msgf:
            #The newer versions of MSGF+ require Java 8.
            #Check that Java 8 is available.
            try:
                java_version = subprocess.Popen(
                    ['java', '-version'], stderr=subprocess.PIPE).communicate()[1].decode().split('.')[1]
                if int(java_version) < 8:
                    load_java8 = True
                else:
                    load_java8 = False
            except FileNotFoundError:
                load_java8 = True
            if load_java8:
                raise RuntimeError('Load Java 8 (java/1.8) or higher to run MSGF+.')

            config.globals['Run MSGF'] = True
            config.globals['Database Search Output Filepath'] = os.path.splitext(os.path.join(
                config.globals['Output Directory'], 
                os.path.basename(os.path.splitext(config.globals['MGF Filepath'])[0]) + '.' + \
                    os.path.basename(os.path.splitext(
                        config.globals['Reference Fasta Filepath'])[0]) + '.mzid'))[0] + '.tsv'
            config.globals['Is Q-Exactive'] = args.qexactive
        else:
            if 'db_search' in args:
                utils.check_path(args.db_search, config.globals['Input Directory'])
                config.globals['Database Search Output Filepath'] = os.path.join(
                    config.globals['Input Directory'], args.db_search)
            if 'reconcile_spectrum_ids' in args:
                if args.reconcile_spectrum_ids:
                    config.globals['Reconcile Spectrum IDs'] = True

    config.globals['Maximum Postnovo Sequence Probability Sacrifice'] = 0
    if 'max_total_sacrifice' in args:
        if args.max_total_sacrifice != None:
            if not 0 <= args.max_total_sacrifice <= 1:
                raise RuntimeError(
                    'max_sacrifice must be in the closed interval of 0 to 1.')
            if not 0 < args.max_sacrifice_per_percent_extension < 1:
                raise RuntimeError(
                    'max_sacrifice_per_percent_extension must be in the open interval of 0 and 1.')
            config.globals['Maximum Postnovo Sequence Probability Sacrifice'] = \
                args.max_total_sacrifice
            config.globals[
                'Maximum Postnovo Sequence Probability Sacrifice Per Percent Length Extension'] = \
                args.max_sacrifice_per_percent_extension

    return

def run_denovogui():
    '''
    Run Novor and PepNovo+ via DeNovoGUI.

    Parameters
    ----------
    None, specified in global variables.

    Returns
    -------
    None
    '''

    denovogui_jar_fp = glob.glob(os.path.join(config.denovogui_dir, '*.jar'))[0]

    #Create strings to be used as DeNovoGUI arguments.
    fixed_mods = '"' + ', '.join(
        [config.postnovo_denovogui_mod_dict[postnovo_mod] 
         for postnovo_mod in config.globals['Fixed Modifications']]) + '"'
    variable_mods = '"' + ', '.join(
        [config.postnovo_denovogui_mod_dict[postnovo_mod] 
         for postnovo_mod in config.globals['Variable Modifications']]) + '"'
    if config.globals['Fragment Mass Resolution'] == 'Low':
        frag_analyzer = 'Trap'
    elif config.globals['Fragment Mass Resolution'] == 'High':
        frag_analyzer = 'FT'

    #Create a temporary output directory for de novo sequencing in progress.
    #This separates the current Postnovo process from others run at the same time.
    temp_out_dir = os.path.join(
        config.globals['Output Directory'], config.globals['MGF Filename'] + '.denovogui_temp')
    os.mkdir(temp_out_dir)
    denovogui_stdout_fp = os.path.join(temp_out_dir, 'denovogui.out')
    #Transfer the MGF file into the temporary directory, 
    #as the MGF.CUI file is produced in the MGF file directory.
    shutil.copy2(config.globals['MGF Filepath'], temp_out_dir)
    temp_mgf_fp = os.path.join(temp_out_dir, os.path.basename(config.globals['MGF Filepath']))
    for frag_mass_tol in config.globals['Fragment Mass Tolerances']:
        #Each fragment mass tolerance parameterization requires a different parameters file.
        param_fp = os.path.join(
            temp_out_dir, config.globals['MGF Filename'] + '.' + frag_mass_tol + '.par')

        #Create the parameters file.
        subprocess.call([
            'java', 
            '-cp', 
            denovogui_jar_fp, 
            'com.compomics.denovogui.cmd.IdentificationParametersCLI', 
            '-out', 
            param_fp, 
            '-prec_tol', 
            str(config.globals['Precursor Mass Tolerance']), 
            '-frag_tol', 
            frag_mass_tol, 
            '-fixed_mods', 
            fixed_mods, 
            '-variable_mods', 
            variable_mods, 
            '-pepnovo_hitlist_length', 
            str(config.seqs_reported_per_alg_dict['PepNovo']), 
            '-novor_fragmentation', 
            config.globals['Fragmentation Method'], 
            '-novor_mass_analyzer', 
            frag_analyzer])
        #Run Novor and PepNovo+.
        with open(denovogui_stdout_fp, 'w') as out_f:
            subprocess.call([
                'java', 
                '-cp', 
                denovogui_jar_fp,
                'com.compomics.denovogui.cmd.DeNovoCLI', 
                '-spectrum_files', 
                temp_mgf_fp, 
                '-output_folder', 
                temp_out_dir, 
                '-id_params', 
                param_fp, 
                '-novor', 
                '1', 
                '-pepnovo', 
                '1', 
                '-directag', 
                '0', 
                '-threads', 
                str(config.globals['CPU Count'])], stdout=out_f)

        #Rename the output file to indicate the fragment mass tolerance used.
        subprocess.call([
            'mv', 
            os.path.join(temp_out_dir, config.globals['MGF Filename'] + '.novor.csv'), 
            os.path.join(
                config.globals['Output Directory'], 
                config.globals['MGF Filename'] + '.' + frag_mass_tol + '.novor.csv')])
        subprocess.call([
            'mv', 
            os.path.join(temp_out_dir, config.globals['MGF Filename'] + '.mgf.out'), 
            os.path.join(
                config.globals['Output Directory'], 
                config.globals['MGF Filename'] + '.' + frag_mass_tol + '.mgf.out')])
        subprocess.call(['rm', param_fp])
    #Remove temporary folder, MGF.CUI and DeNovoGUI output files.
    subprocess.call(['rm', '-r', temp_out_dir])

    return

def run_msgf():
    '''
    Perform database search with MSGF+.

    Parameters
    ----------
    None, specified in global variables.

    Returns
    -------
    None
    '''

    #Create strings to be used as MSGF+ arguments.
    if config.globals['Fragmentation Method'] == 'CID':
        frag_method = '1'
    elif config.globals['Fragmentation Method'] == 'HCD':
        frag_method = '3'

    if config.globals['Is Q-Exactive']:
        instrument_method = '3'
    elif config.globals['Fragment Mass Resolution'] == 'Low':
        instrument_method = '0'
    elif config.globals['Fragment Mass Resolution'] == 'High':
        instrument_method = '1'

    #The output file contains both the filename of the MGF and FASTA inputs.
    mzid_out_fp = os.path.join(
        config.globals['Output Directory'], 
        os.path.basename(os.path.splitext(config.globals['MGF Filepath'])[0]) + '.' + \
        os.path.basename(
            os.path.splitext(config.globals['Reference Fasta Filepath'])[0]) + '.mzid')
    tsv_out_fp = os.path.splitext(mzid_out_fp)[0] + '.tsv'

    msgf_stdout_fp = os.path.join(config.postnovo_dir, 'msgf.out')

    #Run MSGF+.
    with open(msgf_stdout_fp, 'w') as out_f:
        subprocess.call([
            'java', 
            '-Xmx3500M', 
            '-jar', 
            config.msgf_jar, 
            '-s', 
            config.globals['MGF Filepath'], 
            '-d', 
            config.globals['Reference Fasta Filepath'], 
            '-o', 
            mzid_out_fp, 
            '-t', 
            str(config.globals['Precursor Mass Tolerance']) + 'ppm', 
            '-ti', 
            '-1,2', 
            '-thread', 
            str(config.globals['CPU Count']), 
            '-m', 
            frag_method, 
            '-inst', 
            instrument_method, 
            '-mod', 
            config.msgf_mods_fp], stdout=out_f, stderr=out_f)

    #Convert MSGF+ MZID output to TSV.
    with open(msgf_stdout_fp, 'a') as out_f:
        subprocess.call([
            'java', 
            '-Xmx3500M', 
            '-cp', 
            config.msgf_jar, 
            'edu.ucsd.msjava.ui.MzIDToTsv', 
            '-i', 
            mzid_out_fp, 
            '-o', 
            tsv_out_fp, 
            '-showDecoy', 
            '1'], stdout=out_f, stderr=out_f)

    ref_fasta_prefix = os.path.splitext(
        os.path.basename(config.globals['Reference Fasta Filepath']))[0]
    csarr_fp = os.path.join(config.globals['Output Directory'], ref_fasta_prefix + '.csarr')
    cnlcp_fp = os.path.join(config.globals['Output Directory'], ref_fasta_prefix + '.cnlcp')
    cseq_fp = os.path.join(config.globals['Output Directory'], ref_fasta_prefix + '.cseq')
    canno_fp = os.path.join(config.globals['Output Directory'], ref_fasta_prefix + '.canno')
    subprocess.call(['rm', mzid_out_fp, csarr_fp, cnlcp_fp, cseq_fp, canno_fp])

    return

def make_additional_globals():
    '''
    Make global variables derived from user argument globals.

    Parameters
    ----------
    None, specified in global variables.

    Returns
    -------
    None
    '''

    #Columns in the final prediction table 
    #record the algorithms that contribute the de novo sequence in the row.
    #Example:
    #Novor-DeepNovo consensus sequence:
    #'Is Novor Sequence' == 1, 'Is PepNovo Sequence' = 0, 'Is DeepNovo Sequence' = 1
    config.globals['De Novo Algorithm Origin Headers'] = []
    for alg in config.globals['De Novo Algorithms']:
        config.globals['De Novo Algorithm Origin Headers'].append('Is ' + alg + ' Sequence')
    #The algorithmic origin of de novo sequences is also numerically encoded.
    #Example:
    #Novor-DeepNovo consensus sequence: (1, 0, 1)
    #1 for Novor, 0 for PepNovo+, 1 for DeepNovo
    config.globals['De Novo Algorithm Origin Header Keys'] = []
    for key in list(product((0, 1), repeat=len(config.globals['De Novo Algorithms'])))[1:]:
        config.globals['De Novo Algorithm Origin Header Keys'].append(key)

    if config.globals['Retrain']:
        return

    #Map mod symbols in Novor output seqs to Postnovo symbols.
    for i, postnovo_mod_code in enumerate(
        #This will have to be checked with the addition of more mods: 
        #the mod symbols in Novor output using the default mods are M(0) and C(1).
        config.globals['Variable Modifications'] + config.globals['Fixed Modifications']):
        config.novor_postnovo_mod_dict[
            postnovo_mod_code[0] + '(' + str(i) + ')'] = postnovo_mod_code
    #Map mod symbols in PepNovo+ output seqs to Postnovo symbols.
    for postnovo_mod_code in config.globals['Fixed Modifications'] + \
        config.globals['Variable Modifications']:
        config.pn_postnovo_mod_dict[
            postnovo_mod_code[:2] + str(round(float(postnovo_mod_code[2:])))] = postnovo_mod_code

    #Determine the algorithm comparisons needed for consensus sequences.
    #Example: 
    #[('Novor', 'PepNovo'), 
    #('Novor', 'DeepNovo'), 
    #('PepNovo', 'DeepNovo'), 
    #('Novor', 'PepNovo', 'DeepNovo')]
    config.globals['De Novo Algorithm Comparisons'] = []
    for combo_level in range(2, len(config.globals['De Novo Algorithms']) + 1):
        config.globals['De Novo Algorithm Comparisons'] += [
            combo for combo in combinations(
                config.globals['De Novo Algorithms'], combo_level)]

    #Determine potential isobaric and near-isobaric sequences of different lengths.
    standard_plus_mod_mass_dict = config.standard_plus_mod_mass_dict
    #Example: mod == 'C+57.021' -> dict value == 131.04049 + 57.02146
    standard_plus_mod_mass_dict.update(OrderedDict([
        (mod, config.postnovo_mod_mass_dict[mod]) 
        for mod in config.globals['Fixed Modifications']]))
    standard_plus_mod_mass_dict.update(OrderedDict([
        (mod, config.postnovo_mod_mass_dict[mod]) 
        for mod in config.globals['Variable Modifications']]))
    #Remove standard amino acids with fixed mods from consideration.
    #Example: By default, remove the standard amino acid, "C", as it is always "C+57.021".
    for fixed_mod in config.globals['Fixed Modifications']:
        standard_plus_mod_mass_dict.pop(fixed_mod[0])
    all_permuted_isobaric_peps_dict, all_permuted_near_isobaric_peps_dict = \
        utils.find_isobaric(standard_plus_mod_mass_dict, config.MAX_SUBSEQ_LEN)
    #Numerically encode the isobaric peptides.
    #Update dicts in config to retain the persistence of mutable variables.
    aa_code_dict = config.aa_code_dict
    #First, consider exact isobaric sequences.
    for len_combo, peps in all_permuted_isobaric_peps_dict.items():
        config.all_permuted_isobaric_peps_dict[len_combo] = []
        for pep in peps:
            config.all_permuted_isobaric_peps_dict[len_combo].append(
                tuple([aa_code_dict[aa] for aa in list(pep)]))
    #Second, consider near-isobaric sequences.
    for len_combo, peps in all_permuted_near_isobaric_peps_dict.items():
        config.all_permuted_near_isobaric_peps_dict[len_combo] = []
        for pep in peps:
            config.all_permuted_near_isobaric_peps_dict[len_combo].append(
                tuple([aa_code_dict[aa] for aa in list(pep)]))

    #Retrieve peptide mass and retention time information from the MGF file for each spectrum.
    mgf_info_dict = OrderedDict()
    with open(config.globals['MGF Filepath']) as handle:
        for line in handle.readlines():
            if 'PEPMASS=' == line[:8]:
                #Despite the name, this is actually the m/z measurement.
                mz = float(line.replace('PEPMASS=', '').rstrip())
            elif 'CHARGE=' == line[:7]:
                charge = int(line.replace('CHARGE=', '').rstrip('+\n'))
            #Spectrum and Scan IDs are equal in Postnovo-formatted MGF files.
            elif 'SCANS=' == line[:6]:
                scan = int(line.replace('SCANS=', '').rstrip())
            elif 'RTINSECONDS=' == line[:12]:
                rt = float(line.replace('RTINSECONDS=', '').rstrip())
            elif 'END IONS' == line[:8]:
                mgf_info_dict[scan] = OrderedDict(
                    [('M/Z', mz), ('Charge', charge), ('Retention Time', rt)])
    config.mgf_info_dict.update(mgf_info_dict)

    return