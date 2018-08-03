'''
Process user input
'''

import argparse
import getopt
import datetime
import os
import pandas as pd
import re
import subprocess
import sys

from collections import OrderedDict
from copy import deepcopy
from itertools import combinations, product
from multiprocessing import cpu_count
from pkg_resources import resource_filename

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.utils as utils
else:
    import config
    import utils

def setup(test_argv=None):

    parse_args(test_argv)
    if 'denovogui_fp' in config.globals:
        run_denovogui()
    make_other_globals()

    return args
    
def parse_args(test_argv=None):

    parser = argparse.ArgumentParser(
        description='Postnovo post-processes peptide de novo sequences to improve their accuracy.'
    )
    subparsers = parser.add_subparsers()

    mods_parser = subparsers.add_parser(
        'mods_list', 
        dest='subparser_name', 
        help='Displays accepted modifications and their symbols'
    )

    mgf_format_parser = subparsers.add_parser(
        'format_mgf', 
        dest='subparser_name', 
        help='Reformats mgf input file to be compatible with de novo sequencing tools and Postnovo'
    )
    mgf_format_parser.add_argument('mgf', help='Path to mgf file')
    mgf_format_parser.add_argument('--deepnovo', help='Flag to make an mgf file for DeepNovo')

    msgf_format_parser.add_argument()

    #Arguments applicable to predict, test and train subparsers
    parser.add_argument(
        '--iodir',
        help=(
            'When specified, all input files should be in this directory: '
            'full filepaths for input files are not needed when this option is used.'
        )
    )
    parser.add_argument(
        '--pre_mass_tol', 
        default=10, 
        help='Precursor mass tolerance in ppm'
    )
    parser.add_argument(
        '--frag_method', 
        choices=['CID', 'HCD'], 
        default='CID', 
        help='Fragmentation method'
    )
    parser.add_argument(
        '--frag_resolution', 
        choices=['low', 'high'], 
        default='low', 
        help=('Fragment mass resolution')
    )
    parser.add_argument(
        '--fixed_mods', 
        nargs='+', 
        default=config.default_fixed_mods, 
        help=(
            'Enter modifications as a comma-separated list in quotes. '
            'Display the list of accepted mods with the Postnovo mods_list command.'
        )
    )
    parser.add_argument(
        '--variable_mods', 
        nargs='+', 
        default=config.default_variable_mods, 
        help=(
            'Enter modifications as a comma-separated list in quotes. '
            'Display the list of accepted mods with the Postnovo mods_list command.'
        )
    )
    parser.add_argument(
        '--denovogui',
        help='DeNovoGUI jar filepath'
    )
    parser.add_argument(
        '--mgf',
        help=(
            'Input mgf filepath: '
            'Postnovo output stored in this directory if the iodir option is not specified.'
        )
    )
    parser.add_argument(
        '--filename',
        help=(
            'If DeNovoGUI {0}-{1} Da output files have already been generated, '
            'provide the filename prefix rather than an mgf file: '
            'e.g., <filename>.0.2.novor.csv, <filename>.0.2.mgf.out', '<filename>.0.2.deepnovo.tab'
            .format(config.frag_mass_tols[0], config.frag_mass_tols[1])
        )
    )
    parser.add_argument(
        '--deepnovo',
        default=False,
        action='store_true',
        help=(
            'Flag for use of DeepNovo: '
            'DeepNovo output files, e.g., <filename>.0.2.deepnovo.tab, '
            'should be in iodir'
        )
    )
    parser.add_argument(
        '--cpus', 
        type=int, 
        default=1, 
        help='Number of CPUs to use'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='No messages to standard output'
    )

    #Predict mode subparser
    predict_parser = subparsers.add_parser('predict', dest='subparser_name')
    predict_parser.add_argument(
        '--min_len', 
        type=int, 
        default=9, 
        help=(
            'The minimum length of seqs reported by Postnovo, '
            'with the absolute minimum being {0} amino acids'.format(
                config.train_consensus_len
            )
        )
    )
    predict_parser.add_argument(
        '--min_prob', 
        type=float, 
        default=0.5, 
        help='Minimum Postnovo score (~probability) of reported Postnovo sequences.'
    )
    predict_parser.add_argument(
        '--max_total_sacrifice', 
        type=float, 
        help=(
            'The maximum Postnovo score (~probability) that can be sacrificed '
            'to extend the reported sequence'
        )
    )
    predict_parser.add_argument(
        '--max_sacrifice_per_percent_extension', 
        type=float, 
        default=0.0035, 
        help=(
            'Maximum Postnovo score (~probability) that can be sacrificed '
            'per percent change in sequence length: '
            'the default is 0.0035, or a max score sacrifice of 0.05 '
            'to add an amino acid to a length 7 seq (0.0035 = 0.05 / (1/7 * 100)) ; '
            'or a max score sacrifice of 0.025 '
            'to add an amino acid to a length 14 seq'
        )
    )

    #Test mode subparser
    test_parser = subparsers.add_parser('test', dest='subparser_name')
    test_parser.add_argument(
        '--db_search', 
        nargs='+', 
        help='Table of PSMs produced by MSGF+ in tsv format'
    )
    test_parser.add_argument(
        '--ref_fasta', 
        nargs='+', 
        help='Fasta reference file used in database search'
    )
    test_parser.add_argument(
        '--min_len', 
        type=int, 
        default=9, 
        help=(
            'The minimum length of seqs reported by Postnovo, '
            'with the absolute minimum being {0} amino acids'.format(
                config.train_consensus_len
            )
        )
    )
    test_parser.add_argument(
        '--max_total_sacrifice', 
        type=float, 
        help=(
            'The maximum probability score that can be sacrificed '
            'to extend the reported sequence'
        )
    )
    test_parser.add_argument(
        '--sacrifice_floor', 
        type=float, 
        default=0.5, 
        help=(
            'The minimum probability below which sequences will not be extended '
            'at the expense of probability'
        )
    )
    test_parser.add_argument(
        '--max_sacrifice_per_percent_extension', 
        type=float, 
        default=0.0035, 
        help=(
            'Maximum Postnovo score (~probability) that can be sacrificed '
            'per percent change in sequence length: '
            'the default is 0.0035, or a max score sacrifice of 0.05 '
            'to add an amino acid to a length 7 seq (0.0035 = 0.05 / (1/7 * 100)) ; '
            'or a max score sacrifice of 0.025 '
            'to add an amino acid to a length 14 seq'
        )
    )

    #Train mode subparser
    train_parser = subparsers.add_parser('train', dest='subparser_name')
    train_parser.add_argument(
        '--db_search', 
        nargs='+', 
        help='Table of PSMs produced by MSGF+ in tsv format'
    )
    train_parser.add_argument(
        '--ref_fasta', 
        nargs='+', 
        help='Fasta reference file used in database search'
    )

    if test_argv:
        args = parser.parse_args(test_argv)
    else:
        args = parser.parse_args()

    if args.subparser_name == 'mods_list':
        print(open(resource_filename('postnovo', 'data/DeNovoGUI_mods.csv')).read())
        sys.exit()
    elif args.subparser_name == 'format_mgf':
        check_path(args.mgf)
        check_mgf(args.mgf)
        format_mgf(args.mgf, args.deepnovo)
        sys.exit()

    #Determine the I/O directory.
    if args.iodir == None:
        args.iodir = os.path.dirname(args.mgf)
        if args.iodir == '':
            parser.error('Provide the full mgf filepath.')

    inspect_args(parser, args)

    if args.subparser_name == 'predict':
        config.globals['mode'] = 'predict'
    elif args.subparser_name == 'test':
        config.globals['mode'] = 'test'
    elif args.subparser_name == 'train':
        config.globals['mode'] = 'train'

    return

def inspect_args(parser, args):

    check_path(args.iodir)
    config.globals['iodir'] = args.iodir

    config.globals['pre_mass_tol'] = args.pre_mass_tol

    config.globals['frag_method'] = args.frag_method
    config.globals['frag_resolution'] = args.frag_resolution
    if config.globals['frag_resolution'] == 'low':
        config.globals['frag_mass_tols'] = config.low_res_mass_tols
    elif config.globals['frag_resolution'] == 'high':
        config.globals['frag_mass_tols'] = config.hi_res_mass_tols

    config.globals['algs'] = ['novor', 'pn']
    #If DeNovoGUI is run by Postnovo
    if args.filename == None:
        config.globals['filename'] = os.path.splitext(os.path.basename(args.mgf))[0]
    # Else DeNovoGUI output is provided
    else:
        config.globals['filename'] = args.filename
        if args.denovogui == None:
            missing_files = []
            for frag_mass_tol in config.globals['frag_mass_tols']:
                try:
                    missing_files.append(
                        check_path(
                            args.filename + '.' + frag_mass_tol + '.novor.csv',
                            args.iodir, 
                            return_str=True
                        )
                    )
                except TypeError:
                    pass
                try:
                    missing_files.append(
                        check_path(
                            args.filename + '.' + frag_mass_tol + '.mgf.out', 
                            args.iodir, 
                            return_str=True)
                        )
                except TypeError:
                    pass
                
            if missing_files:
                for missing_file in missing_files:
                    if missing_file != None:
                        print(missing_file)

        for frag_mass_tol in config.globals['frag_mass_tols']:
            novor_fp = os.path.join(args.iodir, args.filename + '.' + frag_mass_tol + '.novor.csv')
            file_mass_tol_line = pd.read_csv(novor_fp, nrows = 12).iloc[11][0]
            file_mass_tol = file_mass_tol_line.strip('# fragmentIonErrorTol = ').strip('Da')
            if frag_mass_tol != file_mass_tol:
                raise AssertionError(
                    'Novor files do not have the asserted order of fragment mass tolerances.'
                )

    if args.deepnovo:
        config.globals['algs'].append('deepnovo')
        missing_files = []
        for frag_mass_tol in config.globals['frag_mass_tols']:
            try:
                missing_files.append(
                    check_path(
                        args.filename + '.' + frag_mass_tol + '.deepnovo.tab', 
                        args.iodir, 
                        return_str=True
                        )
                    )
            except TypeError:
                pass
        if missing_files:
            for missing_file in missing_files:
                if missing_file != None:
                    print(missing_file)
            #At this point, missing Novor, PN and DeepNovo files have been reported.
            sys.exit()

    config.globals['novor_fps'] = []
    config.globals['pn_fps'] = []
    if args.deepnovo:
        config.globals['deepnovo_fps'] = []
    for frag_mass_tol in config.globals['frag_mass_tols']:
        config.globals['novor_fps'].append(
            os.path.join(
                config.globals['iodir'], 
                config.globals['filename'] + '.' + frag_mass_tol + '.novor.csv'
            )
        )
        config.globals['pn_fps'].append(
            os.path.join(
                config.globals['iodir'], 
                config.globals['filename'] + '.' + frag_mass_tol + '.mgf.out'
            )
        )
        if 'deepnovo' in config.globals['algs']:
            config.globals['deepnovo_fps'].append(
                os.path.join(
                    config.globals['iodir'], 
                    config.globals['filename'] + '.' + frag_mass_tol + '.deepnovo.tab'
                )
            )

    if (config.globals['mode'] == 'test' or config.globals['mode'] == 'train') and \
        (args.db_search == None or args.ref_fasta == None):
        parser.error(
            'Test, train and optimize modes mode require the arguments db_search and ref_fasta.'
        )

    if args.fixed_mods != config.default_fixed_mods:
        check_mods(parser, args.fixed_mods, 'fixed')
    if args.variable_mods != config.default_variable_mods:
        check_mods(parser, args.variable_mods, 'variable')

    config.globals['fixed_mods'] = []
    for fixed_mod in args.fixed_mods:
        config.globals['fixed_mods'].append(fixed_mod)
    config.globals['variable_mods'] = []
    for variable_mod in args.variable_mods:
        config.globals['variable_mods'].append(variable_mod)

    if (args.denovogui != None) ^ (args.mgf != None):
        parser.error('Both DeNovoGUI and mgf arguments must be specified.')
    if args.denovogui == None and args.filename == None:
        parser.error(
            'Run DeNovoGUI via Postnovo to generate Novor and PepNovo+ files, '
            'or supply these files.'
        )
    if args.denovogui:
        check_path(args.denovogui, args.iodir)
        config.globals['denovogui_fp'] = args.denovogui
        check_path(args.mgf, args.iodir)
        if args.iodir:
            config.globals['mgf_fp'] = os.path.join(args.iodir, args.mgf)
        else:
            config.globals['mgf_fp'] = os.path.join(args.mgf)

    if args.cpus > cpu_count() or args.cpus < 1:
        parser.error(str(cpu_count()) + ' cores are available')
    config.globals['cpus'] = args.cpus

    if args.quiet:
        config.globals['verbose'] = False
    else:
        config.globals['verbose'] = True

    if args.min_len:
        if args.min_len < config.train_consensus_len:
            parser.error(
                'min_len must be >= {0} aa'.format(config.train_consensus_len)
            )
        config.globals['min_len'] = args.min_len

    if args.min_prob:
        if not 0 < args.min_prob < 1:
            parser.error('min_prob must be between 0 and 1')
        config.globals['min_prob'] = args.min_prob

    if args.db_search:
        ref_name = os.path.basename(args.ref_fasta.split('.fasta')[0])
        if os.path.basename(args.db_search) != args.filename + '.' + ref_name + '.tsv':
            raise AssertionError(
                'db_search argument must be consistent with the format, <dataset>.<ref>.tsv'
            )
        check_path(args.db_search, args.iodir)
        check_path(args.ref_fasta, args.iodir)
        if args.iodir:
            config.globals['db_search_fp'] = os.path.join(args.iodir, args.db_search)
            config.globals['ref_fasta_fp'] = os.path.join(args.iodir, args.ref_fasta)
        else:
            config.globals['db_search_fp'] = args.db_search
            config.globals['ref_fasta_fp'] = args.ref_fasta

    if args.max_total_sacrifice:
        if not 0 < args.max_total_sacrifice < 1:
            parser.error('max_sacrifice must be between 0 and 1')
        if not 0 < args.sacrifice_floor < 1:
            parser.error('sacrifice_floor must be between 0 and 1')
        if not 0 < args.max_sacrifice_per_percent_extension < 1:
            parser.error('max_sacrifice_per_percent_extension must be between 0 and 1')
        config.globals['max_total_sacrifice'] = args.max_total_sacrifice
        config.globals['sacrific_floor'] = args.sacrifice_floor
        config.globals['max_sacrifice_per_percent_extension'] = \
            args.max_sacrifice_per_percent_extension

    return

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

    return

def check_path(path, iodir=None, return_str=False):
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

def check_mgf(mgf_fp):

    error_msg = (
        'For correct mgf input file formatting, '
        'see https://github.com/semiller10/postnovo/wiki/1.-Input-File-Setup'
    )
    with open(mgf_fp) as f:
        if f.readline().rstrip() != 'BEGIN IONS':
            raise AssertionError(error_msg)
        title_line = f.readline()
        if 'TITLE=File: "' not in title_line or \
            "; SpectrumID: " not in title_line or \
            "; scans: " not in title_line:
            raise AssertionError(error_msg)
        if 'RTINSECONDS=' not in f.readline():
            raise AssertionError(error_msg)
        pepmass_line = f.readline()
        if 'PEPMASS=' not in pepmass_line:
            raise AssertionError(error_msg)
        charge_line = f.readline()
        #Only one charge can be assigned per spectrum.
        if 'CHARGE=' not in charge_line:
            raise AssertionError(error_msg)
        peak_line = ''
        while peak_line != 'END IONS\n':
            peak_line = f.readline()
        if peak_line != 'END IONS\n':
            raise AssertionError(error_msg)
        #BEGIN IONS immediately follows END IONS in mgf file made by msconvert process.
        if 'BEGIN IONS' not in f.readline():
            raise AssertionError(error_msg)

    return

def format_mgf(mgf_fp, for_deepnovo):

    if for_deepnovo:
        new_mgf_fp = os.path.splitext(mgf_fp)[0] + '.deepnovo.mgf'
    else:
        new_mgf_fp = os.path.splitext(mgf_fp)[0] + 'temp.mgf'

    with open(mgf_fp) as in_f, open(new_mgf_fp, mode='w') as out_f:
        for line in open(mgf_fp):
            if line[:7] == 'TITLE=':
                title_line = line
                scans_line = line[line.index('; scans: "') + 10: -2] + '\n'
            elif line[:12] == 'RTINSECONDS=':
                rt_line = line
            elif line[:8] == 'PEPMASS=':
                #Remove intensity information (number after space).
                pepmass_line = line.split(' ')[0] + '\n'
            elif line[:7] == 'CHARGE=':
                #Remove ambiguous charge states (value after ' and ').
                charge_line = line.split(' ')[0] + '\n'
                #Charge is the last header line, so write the new header.
                out_f.write(title_line)
                out_f.write(pepmass_line)
                out_f.write(charge_line)
                out_f.write(scans_line)
                out_f.write(rt_line)
                if for_deepnovo:
                    #DeepNovo requires a nominal (nonsense) peptide sequence assignment.
                    out_f.write('SEQ=A\n')
            else:
                out_f.write(line)

    if for_deepnovo:
        os.rename(new_mgf_fp, mgf_fp)

    return

def run_denovogui():

    fixed_mods = '"' + ', '.join(config.globals['fixed_mods']) + '"'
    variable_mods = '"' + ', '.join(config.globals['variable_mods']) + '"'
    if config.globals['frag_resolution'] == 'low':
        frag_analyzer = 'Trap'
    elif config.globals['frag_resolution'] == 'high':
        frag_analyzer = 'FT'

    for frag_mass_tol in config.globals['frag_mass_tols']:
        param_fp = (
            '"' + 
            os.path.join(
                config.globals.iodir, config.globals['filename'] + '.' + frag_mass_tol + '.par'
            ) + 
            '"'
        )

        #Create the parameters file.
        subprocess.call(
            [
                'java', '-cp', 
                '"' + config.globals['denovogui_fp'] + '"', 
                'com.compomics.denovogui.cmd.IdentificationParametersCLI', 
                '-out', param_fp, 
                '-prec_tol', str(config.globals['pre_mass_tol']), 
                '-frag_tol', frag_mass_tol, 
                '-fixed_mods', fixed_mods, 
                '-variable_mods', variable_mods, 
                '-pepnovo_hitlist_length', str(config.seqs_reported_per_alg_dict['pn']), 
                '-novor_fragmentation', config.globals['frag_method'], 
                '-novor_mass_analyzer', frag_analyzer
            ]
        )

        #Run Novor and PepNovo+.
        with open('denovogui.out', 'w') as handle:
            subprocess.call(
                [
                    'java', '-cp', 
                    '"' + config.globals['denovogui_fp'] + '"', 
                    ' com.compomics.denovogui.cmd.DeNovoCLI ', 
                    '-spectrum_files', '"' + config.globals['mgf_fp'] + '"', 
                    '-output_folder', '"' + config.globals['iodir'] + '"', 
                    '-id_params', param_fp, 
                    '-pepnovo', '1', 
                    '-novor', '1', 
                    '-directag', '0', 
                    '-threads', str(config.globals['cpus'])
                ], 
                stdout=handle
            )

        subprocess.call(
            [
                'mv', 
                os.path.join(config.globals['iodir'], config.globals['filename'] + '.novor.csv'), 
                os.path.join(
                    config.globals['iodir'], 
                    config.globals['filename'] + '.' + frag_mass_tol + '.novor.csv'
                ), 
            ]
        )
        subprocess.call(
            [
                'mv', 
                os.path.join(config.globals['iodir'], config.globals['filename'] + '.mgf.out'), 
                os.path.join(
                    config.globals['iodir'], 
                    config.globals['filename'] + '.' + frag_mass_tol + '.mgf.out'
                ), 
            ]
        )

    return

def make_other_globals():

    ## Example: 
    ## alg_combo_list = [
    ## ('novor', 'pn'), 
    ## ('novor', 'deepnovo'), 
    ## ('pn', 'deepnovo'), 
    ## ('novor', 'pn', 'deepnovo')
    ## ]
    config.globals['alg_combos'] = []
    for combo_level in range(2, len(config.globals['algs']) + 1):
        config.globals['alg_combos'] += [
            combo for combo in combinations(config.globals['algs'], combo_level)
        ]

    #Columns in prediction table recording the algorithmic origin of de novo sequences
    config.globals['is_alg_names'] = []
    for alg in config.globals['algs']:
        config.globals['is_alg_names'].append('is ' + alg + ' seq')
    #Numeric keys for each type of Postnovo prediction: 
    #With 2 algs, Novor and PN => (0, 1) for PN only, (1, 0) for Novor only, (1, 1) for Novor + PN
    config.globals['is_alg_keys'] = []
    for key in list(product((0, 1), repeat=len(config.globals['alg_list'])))[1:]:
        config.globals['is_alg_keys'].append(key)

    return