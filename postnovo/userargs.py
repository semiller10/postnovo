''' Command line script: this module processes user input '''

import getopt
import argparse
import sys
import subprocess
import datetime
import os

import postnovo.config as config
import postnovo.utils as utils

from multiprocessing import cpu_count
from itertools import combinations, product
from collections import OrderedDict
from pkg_resources import resource_filename
from copy import deepcopy

def setup():

    args = parse_args()
    run_denovogui(args)
    set_global_vars(args)

    return args
    
def parse_args():
    
    parser = argparse.ArgumentParser(
        description='postnovo post-processes peptide de novo sequences to improve their accuracy'
        )

    parser.add_argument(
        '--filename',
        help=('specify the name of output and, if provided, Novor/PepNovo+ input files: '
              'if Novor and PepNovo+ {0}-{1} Da files are provided rather than an mgf file, '
              'provide the prefix as filename, '
              'e.g., <filename>.0.2.novor.csv, <filename>.0.2.mgf.out'
              .format(config.frag_mass_tols[0], config.frag_mass_tols[1]))
        )
    # Add 'novor, pn, peaks' choice when ready
    parser.add_argument(
        '--algs',
        choices=['novor, pn'],
        default='novor, pn',
        help=('list the de novo sequencing algorithms that should be considered')
        )
    parser.add_argument(
        '--iodir',
        help=('when specified, all input files should be in dir and output goes to dir: '
              'full filepaths for input files are not needed when this option is used')
        )
    parser.add_argument(
        '--mode',
        choices=['predict', 'test', 'train', 'optimize'],
        default='predict',
        help=('predict: screen "unknown" seqs, '
              'test: screen "known" seqs, '
              'train: train postnovo model with "known" seqs, '
              'optimize: like train, but includes model parameter optimization')
        )
    parser.add_argument(
        '--fixed_mods',
        default='Oxidation of M',
        help=('enter mods as comma-separated list in quotes, '
              'display list of accepted mods with python postnovo --mods_list')
        )
    parser.add_argument(
        '--variable_mods',
        default='Carbamidomethylation of C',
        help=('enter mods as comma-separated list in quotes, '
              'display list of accepted mods with python postnovo --mods_list')
        )
    parser.add_argument(
        '--denovogui_fp',
        help='denovogui jar filepath'
        )
    parser.add_argument(
        '--mgf_fp',
        help=('spectra mgf filepath: '
              'postnovo output stored in directory if no iodir is specified')
        )
    parser.add_argument(
        '--psm_fp_list',
        nargs='+',
        help=('provide any MSGF+ PSM tsv files to compare with postnovo output')
        )
    parser.add_argument(
        '--psm_name_list',
        nargs='+',
        help=('provide corresponding names for MSGF+ PSM datasets specified in psm_fp_list')
        )
    parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='number of cores to use'
        )
    parser.add_argument(
        '--min_len',
        type=int,
        default=9,
        help='min length of seqs reported by postnovo ({0} aa)'.format(config.min_len[0])
        )
    parser.add_argument(
        '--min_prob',
        type=float,
        default=0.5,
        help='min prob of seqs reported by postnovo'
        )
    parser.add_argument(
        '--db_search_psm_file',
        help=('table of psm info from db search needed for test, train and optimize modes: '
              'see GitHub for accepted formats')
        )
    parser.add_argument(
        '--db_search_ref_file',
        help='fasta reference file used in generating database for db search'
        )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='no messages to standard output'
        )
    parser.add_argument(
        '--mods_list',
        action='store_true',
        help='display list of accepted mods'
        )

    raw_args = parser.parse_args()
    report_info(raw_args)
    # config.iodir[0] is assigned before other package-wide variables
    determine_iodir(raw_args)
    raw_args = check_args(parser, raw_args)
    args = parse_mods_strings(raw_args)

    return args

def report_info(args):
    if args.mods_list:
        with open(resource_filename('postnovo', 'data/DeNovoGUI_mods.csv')) as mods_f:
            print(mods_f.read())
        sys.exit(0)

def determine_iodir(args):

    if args.iodir == None:
        config.iodir.append(os.path.dirname(args.mgf_fp))
        if config.iodir[0] == '':
            parser.error('provide the full mgf_fp')
    else:
        config.iodir.append(args.iodir)
        check_path(config.iodir[0])

def parse_mods_strings(args):
    fixed_mods_split1 = args.fixed_mods.split(', ')
    fixed_mods_split2 = []
    for fixed_mod in fixed_mods_split1:
        fixed_mods_split2 += fixed_mod.split(',')
    args.fixed_mods = fixed_mods_split2
    variable_mods_split1 = args.variable_mods.split(', ')
    variable_mods_split2 = []
    for variable_mod in variable_mods_split1:
        variable_mods_split2 += variable_mod.split(',')
    args.variable_mods = variable_mods_split2
    return args

def check_args(parser, args):

    if args.filename == None:
        args.filename = os.path.splitext(os.path.basename(args.mgf_fp))[0]
    else:
        if args.denovogui_fp == None:
            missing_files = []
            for mass_tol in config.frag_mass_tols:
                try:
                    missing_files.append(
                        check_path(args.filename + '.' + mass_tol + '.novor.csv',
                                   config.iodir[0], return_str = True)
                        )
                except TypeError:
                    pass
                try:
                    missing_files.append(
                        check_path(args.filename + '.' + mass_tol + '.mgf.out',
                                   config.iodir[0], return_str = True)
                        )
                except TypeError:
                    pass
        if missing_files:
            for missing_file in missing_files:
                if missing_file != None:
                    print(missing_file)

    if args.mode == 'predict' and \
        (args.db_search_psm_file != None or args.db_search_ref_file != None):
        parser.error('predict mode incompatible with db_search_psm_file and db_search_ref_file')
    if (args.mode == 'test' or args.mode == 'train' or args.mode == 'optimize') and \
        (args.db_search_psm_file == None or args.db_search_ref_file == None):
        parser.error('test, train and optimize modes mode require '
                     'db_search_psm_file and db_search_ref_file')

    if args.fixed_mods != config.fixed_mods[0]:
        check_mods(parser, args.fixed_mods, 'fixed')
    if args.variable_mods != config.variable_mods[0]:
        check_mods(parser, args.variable_mods, 'variable')

    if (args.denovogui_fp != None) ^ (args.mgf_fp != None):
        parser.error('both denovogui_fp and mgf_fp are needed')
    if args.denovogui_fp == None and args.filename == None:
        parser.error('run DeNovoGUI in postnovo to generate Novor and PepNovo+ files, '
                     'or supply these files')
    if args.denovogui_fp:
        check_path(args.denovogui_fp, args.iodir)
        check_path(args.mgf_fp, args.iodir)

    if (args.psm_fp_list != None) ^ (args.psm_name_list != None):
        parser.error('both psm_fp_list and psm_name_list are needed')
    if args.psm_fp_list:
        if len(args.psm_fp_list) != len(args.psm_name_list):
            parser.error('specify an equal number of inputs to psm_fp_list and psm_name_list')
        for psm_fp in args.psm_fp_list:
            check_path(psm_fp, args.iodir)

    if args.cores > cpu_count() or args.cores < 1:
        parser.error(str(cpu_count()) + ' cores are available')
    if args.min_len < 6:
        parser.error('min length of reported peptides must be >= {0} aa'.format(config.min_len[0]))
    if args.min_prob < 0 or args.min_prob > 1:
        parser.error('min reported prob must be between 0 and 1')
    if args.db_search_psm_file != None:
        check_path(args.db_search_psm_file, args.iodir)
    if args.db_search_ref_file != None:
        check_path(args.db_search_ref_file, args.iodir)

    return args

def check_mods(parser, mod_input, mod_type):
    with open(resource_filename('postnovo', 'data/DeNovoGUI_mods.csv')) as mods_f:
        recognized_mods = []
        for line in mods_f:
            for mod in mod_input:
                if mod in line:
                    recognized_mods.append(mod)
    unrecognized_mods = set(mod_input).intersection(recognized_mods)
    if unrecognized_mods != set(mod_input):
        parser.error('{0} mods not recognized: {1}'.format(mod_type, unrecognized_mods))

def check_path(path, iodir = None, return_str = False):
    if iodir is None:
        if os.path.exists(path) == False:
            if return_str:
                return path + ' does not exist'
            print(path + ' does not exist')
            sys.exit(1)
    else:
        full_path = os.path.join(iodir, path)
        if os.path.exists(full_path) == False:
            if return_str:
                return full_path + ' does not exist'
            print(full_path + ' does not exist')
            sys.exit(1)

def run_denovogui(args):
    
    if args.denovogui_fp == None:
        return

    denovogui_param_args = \
        OrderedDict().fromkeys(
            ['-out', '-frag_tol',
             '-fixed_mods', '-variable_mods',
             '-pepnovo_hitlist_length', '-novor_fragmentation', '-novor_mass_analyzer']
            )
    denovogui_param_args['-fixed_mods'] = '\"' + ', '.join(args.fixed_mods) + '\"'
    denovogui_param_args['-variable_mods'] = '\"' + ', '.join(args.variable_mods) + '\"'
    denovogui_param_args['-pepnovo_hitlist_length'] = str(config.seqs_reported_per_alg_dict['pn'])
    denovogui_param_args['-novor_fragmentation'] = '\"' + config.frag_method + '\"'
    denovogui_param_args['-novor_mass_analyzer'] = '\"' + config.frag_mass_analyzer + '\"'

    denovogui_args = OrderedDict().fromkeys(['-spectrum_files', '-output_folder', '-id_params',
                                             '-pepnovo', '-novor', '-directag', '-threads'])
    if os.path.dirname(args.mgf_fp) == '':
        denovogui_args['-spectrum_files'] = '\"' + os.path.join(config.iodir[0], args.mgf_fp) + '\"'
    else:
        denovogui_args['-spectrum_files'] = '\"' + args.mgf_fp + '\"'

    denovogui_args['-output_folder'] = '\"' + config.iodir[0] + '\"'
    denovogui_args['-pepnovo'] = '1'
    denovogui_args['-novor'] = '1'
    denovogui_args['-directag'] = '0'
    denovogui_args['-threads'] = str(args.cores)

    for tol in config.frag_mass_tols:

        denovogui_param_file_cmd = 'java -cp ' +\
            '\"' + args.denovogui_fp + '\"' +\
            ' com.compomics.denovogui.cmd.IdentificationParametersCLI '
        denovogui_param_args['-frag_tol'] = tol
        denovogui_param_args['-out'] = '\"' +\
            os.path.join(config.iodir[0],
                         args.filename + '.' + tol + '.par') + '\"'
        for opt, arg in denovogui_param_args.items():
            denovogui_param_file_cmd += opt + ' ' + arg + ' '
        subprocess.call(denovogui_param_file_cmd, shell = True)

        denovogui_cmd = 'java -cp ' +\
            '\"' + args.denovogui_fp + '\"' +\
            ' com.compomics.denovogui.cmd.DeNovoCLI '
        denovogui_args['-id_params'] = denovogui_param_args['-out']
        for opt, arg in denovogui_args.items():
            denovogui_cmd += opt + ' ' + arg + ' '
        subprocess.call(denovogui_cmd, shell = True)

        set_novor_output_filename_cmd = 'mv ' +\
            '\"' + os.path.join(config.iodir[0], args.filename + '.novor.csv') + '\" ' +\
            '\"' + os.path.join(config.iodir[0], args.filename + '.' + tol + '.novor.csv') + '\"'
        subprocess.call(set_novor_output_filename_cmd, shell = True)

        set_pn_output_filename_cmd = 'mv ' +\
            '\"' + os.path.join(config.iodir[0], args.filename + '.mgf.out') + '\" ' +\
            '\"' + os.path.join(config.iodir[0], args.filename + '.' + tol + '.mgf.out') + '\"'
        subprocess.call(set_pn_output_filename_cmd, shell = True)

def set_global_vars(args):

    for alg in args.algs.split(', '):
        config.alg_list.append(alg)
    for combo_level in range(2, len(config.alg_list) + 1):
        combo_level_combo_list = [combo for combo in combinations(config.alg_list, combo_level)]
        for alg_combo in combo_level_combo_list:
            config.alg_combo_list.append(alg_combo)
    # MultiIndex cols for prediction_df
    for alg in config.alg_list:
        is_alg_col_name = 'is ' + alg + ' seq'
        config.is_alg_col_names.append(is_alg_col_name)
    is_alg_col_multiindex_list = list(product((0, 1), repeat = len(config.alg_list)))
    for multiindex_key in is_alg_col_multiindex_list[1:]:
        config.is_alg_col_multiindex_keys.append(multiindex_key)

    for tol in config.frag_mass_tols:
        config.novor_files.append(
            os.path.join(config.iodir[0], args.filename + '.' + tol + '.novor.csv'))
        config.pn_files.append(
            os.path.join(config.iodir[0], args.filename + '.' + tol + '.mgf.out'))
    alg_fp_lists = {'novor': config.novor_files,
                    'peaks': config.peaks_files,
                    'pn': config.pn_files}
    for alg in config.alg_list:
        config.alg_tols_dict[alg] = OrderedDict(
            zip(config.frag_mass_tols,
                [os.path.basename(fp) for fp in alg_fp_lists[alg]]))

    tol_alg_dict_local = invert_dict_of_lists(config.alg_tols_dict)
    for k, v in tol_alg_dict_local.items():
        config.tol_alg_dict[k] = v
    for tol in config.frag_mass_tols:
        config.tol_basenames_dict[tol] = []
    for alg in config.alg_tols_dict:
        for tol in config.alg_tols_dict[alg]:
            config.tol_basenames_dict[tol] += [config.alg_tols_dict[alg][tol]]

    config.mode[0] = args.mode

    for fixed_mod in args.fixed_mods:
        config.fixed_mods.append(fixed_mod)
    for variable_mod in args.variable_mods:
        config.variable_mods.append(variable_mod)

    if args.psm_fp_list:
        for i, psm_fp in enumerate(args.psm_fp_list):
            config.psm_fp_list.append(
                os.path.join(config.iodir[0], psm_fp))
            config.psm_name_list.append(args.psm_name_list[i])

    config.cores[0] = args.cores
    config.min_len[0] = args.min_len
    config.min_prob[0] = args.min_prob
    if args.db_search_ref_file != None:
        if args.iodir == None:
            config.db_search_psm_file[0] = args.db_search_psm_file
            config.db_search_ref_file[0] = args.db_search_ref_file
        else:
            config.db_search_psm_file[0] = os.path.join(args.iodir, args.db_search_psm_file)
            config.db_search_ref_file[0] = os.path.join(args.iodir, args.db_search_ref_file)
    if args.quiet:
        config.verbose[0] = False

def invert_dict_of_lists(d):
    values = set(a for b in d.values() for a in b)
    values = sorted(list(values))
    invert_d = OrderedDict((new_k, [k for k, v in d.items() if new_k in v]) for new_k in values)
    return invert_d