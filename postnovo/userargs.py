''' Command line script: this module processes user input '''

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

    args = parse_args(test_argv)
    if args.denovogui_fp != None:
        run_denovogui(args)
    set_global_vars(args)

    return args
    
def parse_args(test_argv=None):
    
    parser = argparse.ArgumentParser(
        description='postnovo post-processes peptide de novo sequences to improve their accuracy'
        )

    parser.add_argument(
        '--filename',
        help=('If DeNovoGUI {0}-{1} Da output files are provided, '
              'provide the filename prefix rather than an mgf file: '
              'e.g., <filename>.0.2.novor.csv, <filename>.0.2.mgf.out', '<filename>.0.2.deepnovo.tab'
              .format(config.frag_mass_tols[0], config.frag_mass_tols[1]))
        )
    parser.add_argument(
        '--deepnovo',
        default=False,
        action='store_true',
        help=(
            'flag for use of deepnovo: '
            'deepnovo output files, e.g., <filename>.0.2.deepnovo.tab, '
            'should be in iodir'
            )
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
        '--db_name_list',
        nargs='+',
        help=(
            'provide the names of databases associated with MSGF+ searches: '
            'each name should be UNIQUE to a PAIR of .tsv PSM and .fasta db files in iodir, '
            'e.g., input of "db1, db2" would find files such as '
            '"proteome1.db1.tsv, proteome1.db1.fasta, proteome1.db2.tsv, proteome1.db2.fasta"'
            )
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

    if test_argv:
        raw_args = parser.parse_args(test_argv)
    else:
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

    # If DeNovoGUI is run by postnovo
    if args.filename == None:
        args.filename = os.path.splitext(os.path.basename(args.mgf_fp))[0]
    # Else DeNovoGUI output is provided
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

        for tol in config.frag_mass_tols:
            novor_fp = os.path.join(config.iodir[0], args.filename + '.' + tol + '.novor.csv')
            file_mass_tol_line = pd.read_csv(novor_fp, nrows = 12).iloc[11][0]
            file_mass_tol = round(float(
                file_mass_tol_line.strip('# fragmentIonErrorTol = ').strip('Da')) * 10) / 10
            if float(tol) != file_mass_tol:
                raise AssertionError(
                    'Novor files do not have the asserted order')

    if args.deepnovo:
        missing_files = []
        for mass_tol in config.frag_mass_tols:
            try:
                missing_files.append(
                    check_path(
                        args.filename + '.' + mass_tol + '.deepnovo.tab',
                        config.iodir[0],
                        return_str=True
                        )
                    )
            except TypeError:
                pass

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
        check_mgf(args.mgf_fp, args.iodir)

    if args.db_name_list:
        iodir_files = [f for f in os.listdir(args.iodir) if os.path.isfile(os.path.join(args.iodir, f))]
        for name in args.db_name_list:
            matching_tsv_files = []
            matching_db_files = []
            for f in iodir_files:
                if os.path.splitext(f)[1] == '.tsv' and name in f:
                    matching_tsv_files.append(f)
                elif os.path.splitext(f)[1] == '.fasta' and name in f:
                    matching_db_files.append(f)
            if matching_tsv_files and matching_db_files:
                if len(matching_tsv_files) > 1 or len(matching_db_files) > 1:
                    parser.error(
                        'for each name in db_name_list, '
                        'there must be exactly one corresponding .tsv PSM file containing this string, '
                        'and exactly one .fasta database file containing this string in iodir'
                        )
            else:
                parser.error(
                    'for each name in db_name_list, '
                    'there must be two corresponding .tsv PSM and .fasta db files in iodir'
                    )

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

def check_mgf(mgf_fp, iodir):
    '''
    Check to see if mgf is formatted in standard (default Proteome Discoverer raw -> mgf) fashion
    '''
    
    # Default Proteome Discoverer raw -> mgf format, as I have seen it
    # MASS=Monoisotopic [at very top of file]
    # BEGIN IONS
    # TITLE=File: "<path to raw file>"; [continuing...]
    # SpectrumID: "<spectrum count>"; [continuing...]
    # scans: "<scan>"
    # PEPMASS=<m/z to the hundred-thousandth separated by one space from intensity to the hundred-thousandth>
    # CHARGE=<charge state, e.g., 2+>
    # RTINSECONDS=<retention time to the zeroth, e.g., 0>
    # SCANS=<scan>
    # <peak list: m/z to the thousandth separated by one space from intensity to six sig figs>
    
    # The only thing I ignore when reformatting thusly is rounding

    if iodir is not None:
        mgf_fp = os.path.join(iodir, mgf_fp)
    with open(mgf_fp) as f:
        for line in f:
            if 'TITLE=' in line:
                try:
                    # Proteome Discoverer raw -> mgf output for Orbitrap Elite
                    # works in all the programs of the pipeline
                    if (
                        line.index('File: \"') <
                        line.index('SpectrumID: \"') <
                        line.index('scans: \"')
                        ):
                        break
                except:
                    # Default ProteoWizard msconvert output for Orbitrap Elite
                    if (
                        line.index('File:\"') <
                        line.index('NativeID:\"') <
                        line.index('scan=')
                        ):
                        reformat_msconvert_mgf(mgf_fp)
                        break

def reformat_msconvert_mgf(mgf_fp):
    '''
    Reformat default ProteoWizard msconvert raw -> mgf output to Proteome Discoverer format
    '''

    # Default ProteoWizard msconvert raw -> mgf format, as I have seen it
    # BEGIN IONS
    # TITLE=<file basename w/out extension>.<scan>.<scan>.2 [continuing...]
    # File:"<file basename w/ extension>", [continuing...]
    # NativeID:"controllerType=0 controllerNumber=1 scan=<scan>"
    # RTINSECONDS=<retention time to the ten-thousandth>
    # PEPMASS=<m/z to the trillionth separated by one space from intensity to the trillionth>
    # CHARGE=<charge state, e.g., 2+>
    # <peak list: m/z to the ten-millionth separated by one space from intensity to the ten-billionth>

    original_mgf_path = os.path.splitext(mgf_path)[0] + '.original.mgf'
    subprocess.call(['cp', mgf_path, original_mgf_path])
    with open(mgf_path) as handle:
        original_mgf = handle.readlines()

    raw_fp_for_title = '\"' + os.path.splitext(mgf_path)[0] + '.raw' + '\"'
    spectrum_id_count = 0
    title_lines = []
    scan_lines = []
    rt_lines = []
    pepmass_lines = []
    charge_lines = []
    for line in original_mgf:
        if 'BEGIN IONS' in line:
            spectrum_id_count += 1
        elif 'TITLE' in line:
            scan = line[line.index('scan=') + 5: line.index('\"\n')]
            scan_lines.append('SCANS={0}\n'.format(scan))
            title_lines.append(
                'TITLE=File: ' + raw_fp_for_title + '; '
                'SpectrumID: \"{0}\"; '
                'scans: \"{1}\"\n'.format(str(spectrum_id_count), scan)
                )
        elif 'RTINSECONDS' in line:
            rt_lines.append(line)
        elif 'PEPMASS' in line:
            pepmass_lines.append(line)
        elif 'CHARGE' in line:
            charge_lines.append(line)
                
    new_mgf = ['MASS=Monoisotopic\n']
    spectrum_index = 0
    for line in original_mgf:
        if 'TITLE' in line:
            new_mgf.append(title_lines[spectrum_index])
            new_mgf.append(pepmass_lines[spectrum_index])
            new_mgf.append(charge_lines[spectrum_index])
            new_mgf.append(rt_lines[spectrum_index])
            new_mgf.append(scan_lines[spectrum_index])
            spectrum_index += 1
        elif 'RTINSECONDS' in line:
            pass
        elif 'PEPMASS' in line:
            pass
        elif 'CHARGE' in line:
            pass
        else:
            new_mgf.append(line)

    with open(mgf_path, 'w') as handle:
        for line in new_mgf:
            handle.write(line)

def run_denovogui(args):

    denovogui_param_args = \
        OrderedDict().fromkeys(
            ['-out',
             '-frag_tol',
             '-fixed_mods',
             '-variable_mods',
             '-pepnovo_hitlist_length',
             '-novor_fragmentation',
             '-novor_mass_analyzer']
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

    config.filename.append(args.filename)

    if args.deepnovo:
        config.alg_list.append('deepnovo')

    for combo_level in range(2, len(config.alg_list) + 1):
        combo_level_combo_list = [combo for combo in combinations(config.alg_list, combo_level)]
        for alg_combo in combo_level_combo_list:
            config.alg_combo_list.append(alg_combo)
    # MultiIndex cols for prediction_df
    for alg in config.alg_list:
        is_alg_col_name = 'is ' + alg + ' seq'
        config.is_alg_col_names.append(is_alg_col_name)
    is_alg_col_multiindex_list = list(product((0, 1), repeat=len(config.alg_list)))
    for multiindex_key in is_alg_col_multiindex_list[1:]:
        config.is_alg_col_multiindex_keys.append(multiindex_key)
    
    precursor_mass_tol_info_str = pd.read_csv(
        os.path.join(config.iodir[0], 
                     config.filename[0] + '.' + config.frag_mass_tols[0] + '.novor.csv'),
        nrows = 13).iloc[12][0]
    try:
        config.precursor_mass_tol[0] = float(
            re.search('(?<=# precursorErrorTol = )(.*)(?=ppm)', 
                      precursor_mass_tol_info_str).group(0))
    except ValueError:
        pass

    for tol in config.frag_mass_tols:
        config.novor_files.append(
            os.path.join(config.iodir[0], args.filename + '.' + tol + '.novor.csv'))
        config.pn_files.append(
            os.path.join(config.iodir[0], args.filename + '.' + tol + '.mgf.out'))
        if 'deepnovo' in config.alg_list:
            config.deepnovo_files.append(
                os.path.join(config.iodir[0], args.filename + '.' + tol + '.deepnovo.tab')
                )
    alg_fp_lists = {'novor': config.novor_files,
                    'pn': config.pn_files}
    if 'deepnovo' in config.alg_list:
        alg_fp_lists['deepnovo'] = config.deepnovo_files

    config.mode[0] = args.mode

    for fixed_mod in args.fixed_mods:
        config.fixed_mods.append(fixed_mod)
    for variable_mod in args.variable_mods:
        config.variable_mods.append(variable_mod)

    if args.db_name_list:
        iodir_files = [f for f in os.listdir(args.iodir) if os.path.isfile(os.path.join(args.iodir, f))]
        for name in args.db_name_list:
            config.db_name_list.append(name)
            for f in iodir_files:
                if os.path.splitext(f)[1] == '.tsv' and name in f:
                    config.psm_fp_list.append(os.path.join(config.iodir[0], f))
                if os.path.splitext(f)[1] == '.fasta' and name in f:
                    config.db_fp_list.append(os.path.join(config.iodir[0], f))

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