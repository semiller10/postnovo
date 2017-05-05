''' Command line script: this module processes user input '''

import getopt
import sys
import subprocess
import datetime
import os

import postnovo.config as config
import postnovo.utils as utils

#import config
#import utils

from multiprocessing import cpu_count
from itertools import combinations, product
from collections import OrderedDict


def setup():
    args = run_denovogui(
        order_by_tol(
            arg_cross_check(
                parse_args())))
    
    download_forest_dict(args)
    set_global_vars(args)
    return args
    
def parse_args():
    ''' Return command line args as dict '''
    
    args = {}

    ## FOR DEBUGGING PURPOSES: UNCOMMENT
    ## Take in args
    #if len(sys.argv) == 1:
    #     print(config.help_str)
    #     sys.exit(0)

    ## FOR DEBUGGING PURPOSES: REMOVE
    #test_str = ['--param_file', 'param.json',
    #            '--iodir', 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\test']
    test_str = ['--param_file', 'param.json',
                '--iodir', '/home/samuelmiller/5-2-17/postnovo/io']

    try:
        ## FOR DEBUGGING PURPOSES: REMOVE AND UNCOMMENT
        opts, leftover = getopt.getopt(test_str, shortopts = '', longopts = config.getopt_opts)
        #opts, leftover = getopt.getopt(sys.argv[1:], shortopts = '', longopts = config.getopt_opts)
    except getopt.GetoptError:
        print(config.help_str)
        sys.exit(1)

    if '--param_file' in [opt[0] for opt in opts]:
        return parse_param_file(opts)

    try:
        iodir = [opt[1] for opt in opts][[opt[0] for opt in opts].index('--iodir')]
        check_path(iodir)
    except:
        print('No iodir')
        sys.exit(1)

    # Parse each option
    for opt, arg in opts:
        if opt == '--param_file':
            return 
        elif opt == '--help':
            print(help_str)
            sys.exit(0)
        elif opt == '--quiet':
            args['quiet'] = True
        elif opt == '--train':
            args['train'] = True
        elif opt == '--test':
            args['test'] = True
        elif opt == '--optimize':
            args['optimize'] = True
        elif opt == '--iodir':
            args['iodir'] = iodir
        elif opt == '--denovogui_path':
            denovogui_path = arg
            check_path(denovogui_path)
            args['denovogui_path'] = denovogui_path
        elif opt == '--denovogui_mgf_path':
            denovogui_mgf_path = arg
            check_path(denovogui_mgf_path)
            args['denovogui_mgf_path'] = denovogui_mgf_path
        #elif opt == '--frag_mass_tols':
        #    frag_mass_tols = arg.split(',')
        #    frag_mass_tols = [os.path.join(iodir, tol.strip()) for tol in frag_mass_tols]
        #    check_frag_mass_tols(frag_mass_tols)
        #    args['frag_mass_tols'] = frag_mass_tols
        elif opt == '--novor_files':
            novor_files = arg.split(',')
            novor_files = [os.path.join(iodir, f.strip()) for f in novor_files]
            check_novor_files(novor_files)
            args['novor_files'] = novor_files
        elif opt == '--peaks_files':
            peaks_files = arg.split(',')
            peaks_files = [os.path.join(iodir, f.strip()) for f in peaks_files]
            check_peaks_files(peaks_files)
            args['peaks_files'] = peaks_files
        elif opt == '--pn_files':
            pn_files = arg.split(',')
            pn_files = [os.path.join(iodir, f.strip()) for f in pn_files]
            check_pn_files(pn_files)
            args['pn_files'] = pn_files
        elif opt == '--min_len':
            min_len = check_min_len(arg)
            args['min_len'] = min_len
        elif opt == '--min_prob':
            min_prob = check_min_prob(arg)
            args['min_prob'] = min_prob
        elif opt == '--db_search_ref_file':
            check_path(arg, iodir)
            args['db_search_ref_file'] = os.path.join(iodir, arg)
        elif opt == '--fasta_ref_file':
            check_path(arg, iodir)
            args['fasta_ref_file'] = os.path.join(iodir, arg)
        elif opt == '--cores':
            check_cores(arg)
            args['cores'] = int(arg)
        else:
            print('Unrecognized option ' + opt)
            sys.exit(1)

    return args

def arg_cross_check(args):

    if 'train' in args or 'test' in args or 'optimize' in args:
        train_test_optimize_cross_check(args)

    #if 'frag_mass_tols' not in args:
    #    print('Fragment mass tolerance(s) of input must be specified')
    #    sys.exit(1)

    if 'denovogui_path' in args:
        if 'denovogui_mgf_path' not in args:
            print('denovogui_mgf_path argument also needed')
            sys.exit(1)

    if 'denovogui_mgf_path' in args:
        if 'denovogui_path' not in args:
            print('denovogui_path argument also needed')
            sys.exit(1)

    if 'denovogui_path' not in args:
        for file_set in ['novor_files', 'pn_files']:
            if file_set in args:
                if len(config.frag_mass_tols) != len(args[file_set]):
                #if len(args['frag_mass_tols']) != len(args[file_set]):
                    print('List of fragment mass tolerances must align with list of file inputs')
                    sys.exit(1)
            else:
                if file_set == 'novor_files':
                    print('Novor input is required')
                elif file_set == 'pn_files':
                    print('PepNovo+ input is required')
                sys.exit(1)

    return args

def train_test_optimize_cross_check(args):

    if 'db_search_ref_file' not in args:
        print('A database search PSM table reference file is required')
        sys.exit(1)
    if 'fasta_ref_file' not in args:
        print('A fasta reference file is required')
        sys.exit(1)
    if 'train' in args:
        if 'test' in args or 'optimize' in args:
            print('Train, test and optimize options are exclusive')
            sys.exit(1)
    if 'test' in args:
        if 'optimize' in args:
            print('Train, test and optimize options are exclusive')
            sys.exit(1)

def parse_param_file(opts):
    
    args = {}

    for opt, arg in opts:
        if opt == '--param_file':
            param_file = arg

        elif opt == '--iodir':
            iodir = arg
            if os.path.exists(iodir) is False:
                print(iodir + ' does not exist')
                sys.exit(1)

        else:
            print('When using a json param file to pass arguments to postnovo, other command line arguments beside iodir are not accepted')
            sys.exit(1)

    check_path(param_file, iodir)
    args = utils.load_json_objects(iodir, param_file.strip('.json'))

    for opt, arg in args.items():
        if opt == 'denovogui_path':
            check_path(arg)
        elif opt == 'denovogui_mgf_path':
            check_path(arg)
        #elif opt == 'frag_mass_tols':
        #    frag_mass_tols = [str(frag_mass_tol) for frag_mass_tol in arg]
        #    check_frag_mass_tols(frag_mass_tols)
        #    args['frag_mass_tols'] = frag_mass_tols
        elif opt == 'novor_files':
            novor_files = [os.path.join(iodir, f.strip()) for f in arg]
            check_novor_files(novor_files)
            args['novor_files'] = novor_files
        elif opt == 'peaks_files':
            peaks_files = [os.path.join(iodir, f.strip()) for f in arg]
            check_peaks_files(peaks_files)
            args['peaks_files'] = peaks_files
        elif opt == 'pn_files':
            pn_files = [os.path.join(iodir, f.strip()) for f in arg]
            check_pn_files(pn_files)
            args['pn_files'] = pn_files
        elif opt == 'min_len':
            min_len = check_min_len(arg)
            args['min_len'] = min_len
        elif opt == 'min_prob':
            min_prob = check_min_prob(arg)
            args['min_prob'] = min_prob
        elif opt == 'db_search_ref_file':
            db_search_ref_file = os.path.join(iodir, arg)
            check_path(db_search_ref_file)
            args['db_search_ref_file'] = db_search_ref_file
        elif opt == 'fasta_ref_file':
            fasta_ref_file = os.path.join(iodir, arg)
            check_path(fasta_ref_file)
            args['fasta_ref_file'] = fasta_ref_file
        elif opt == 'cores':
            check_cores(arg)
        else:
            if opt in config.getopt_opts:
                pass
            else:
                print('Unrecognized option ' + opt)
                sys.exit(1)

    args['iodir'] = iodir
    return args

def check_path(path, iodir = None):
    if iodir is None:
        if os.path.exists(path) is False:
            print(path + ' does not exist')
            sys.exit(1)
    else:
        full_path = os.path.join(iodir, path)
        if os.path.exists(full_path) is False:
            print(full_path + ' does not exist')

#def check_frag_mass_tols(frag_mass_tols):
#    for tol in frag_mass_tols:
#        if tol not in config.accepted_mass_tols:
#            print(tol + ' must be in list of accepted fragment mass tolerances: ' +\
#                ', '.join(config.accepted_mass_tols))
#            sys.exit(1)

def check_novor_files(novor_files):
    for i, file_name in enumerate(novor_files):
        if '.novor.csv' not in file_name:
            print(file_name + ' must have novor.csv file extension')
            sys.exit(1)
        check_path(file_name)

def check_peaks_files(peaks_files):
    for i, file_name in peaks_files:
        if '.csv' not in file_name:
            print(file_name + ' must have csv file extension')
            sys.exit(1)
        check_path(file_name)

def check_pn_files(pn_files):
    for i, file_name in enumerate(pn_files):
        if '.mgf.out' not in file_name:
            print(file_name + ' must have mgf.out file extension')
            sys.exit(1)
        check_path(file_name)

def check_min_len(arg):
    try:
        min_len = int(arg)
    except ValueError:
        print('Minimum reported sequence length must be an integer >0')
        sys.exit(1)
    if min_len < config.train_consensus_len:
        print('Sequences shorter than length ' + str(config.train_consensus_len) + ' are not supported')
        sys.exit(1)
    return min_len

def check_min_prob(arg):
    try:
        min_prob = float(arg)
    except ValueError:
        print('Minimum reported sequence probability must be a number between 0 and 1')
        sys.exit(1)
    if min_prob <= 0 or min_prob >= 1:
        print('Minimum reported sequence probability must be a number between 0 and 1')
        sys.exit(1)
    return min_prob

def check_cores(arg):
    if not float(arg).is_integer():
        print('Specify an integer number of cores')
        sys.exit(1)
    if int(arg) > cpu_count() or int(arg) < 1:
        print(str(cpu_count()) + ' cores are available')
        sys.exit(1)

def download_forest_dict(args):
    if os.path.exists(config.training_dir) is False:
        os.makedirs(config.training_dir)

    if os.path.exists(os.path.join(config.training_dir, 'forest_dict.pkl')) is False\
        and ('predict' in args or 'test' in args):
        with urlopen(forest_dict_url) as response,\
            open(os.path.join(config.training_dir, 'forest_dict.pkl'), 'wb') as out_file:
            copyfileobj(response, out_file)

def run_denovogui(args):

    if 'denovogui_path' not in args:
        return args

    args['novor_files'] = []
    args['pn_files'] = []

    denovogui_param_args = OrderedDict().fromkeys(['-out', '-frag_tol',
                                                   '-fixed_mods', '-variable_mods',
                                                   '-pepnovo_hitlist_length', '-novor_fragmentation', '-novor_mass_analyzer'])
    denovogui_param_args['-fixed_mods'] = '\"' + config.fixed_mod + '\"'
    denovogui_param_args['-variable_mods'] = '\"' + config.variable_mod + '\"'
    denovogui_param_args['-pepnovo_hitlist_length'] = str(config.seqs_reported_per_alg_dict['pn'])
    denovogui_param_args['-novor_fragmentation'] = '\"' + config.frag_method + '\"'
    denovogui_param_args['-novor_mass_analyzer'] = '\"' + config.frag_mass_analyzer + '\"'

    denovogui_args = OrderedDict().fromkeys(['-spectrum_files', '-output_folder', '-id_params',
                                             '-pepnovo', '-novor', '-directag', '-threads'])
    mgf_input_name = os.path.splitext(os.path.basename(args['denovogui_mgf_path']))[0]
    denovogui_args['-spectrum_files'] = '\"' + args['denovogui_mgf_path'] + '\"'
    denovogui_args['-output_folder'] = '\"' + args['iodir'] + '\"'
    denovogui_args['-pepnovo'] = '1'
    denovogui_args['-novor'] = '1'
    denovogui_args['-directag'] = '0'
    denovogui_args['-threads'] = str(args['cores'])

    for tol in config.frag_mass_tols:
    #for tol in args['frag_mass_tols']:

        denovogui_param_file_cmd = 'java -cp ' +\
            '\"' + args['denovogui_path'] + '\"' +\
            ' com.compomics.denovogui.cmd.IdentificationParametersCLI '
        denovogui_param_args['-frag_tol'] = tol
        denovogui_param_args['-out'] = '\"' +\
            os.path.join(args['iodir'],
                 mgf_input_name + '_' + tol + '.par') + '\"'
        for opt, arg in denovogui_param_args.items():
            denovogui_param_file_cmd += opt + ' ' + arg + ' '
        subprocess.call(denovogui_param_file_cmd, shell = True)

        denovogui_cmd = 'java -cp ' +\
            '\"' + args['denovogui_path'] + '\"' +\
            ' com.compomics.denovogui.cmd.DeNovoCLI '
        denovogui_args['-id_params'] = denovogui_param_args['-out']
        for opt, arg in denovogui_args.items():
            denovogui_cmd += opt + ' ' + arg + ' '
        subprocess.call(denovogui_cmd, shell = True)

        args['novor_files'].append(os.path.join(args['iodir'], mgf_input_name + '_' + tol + '.novor.csv'))
        set_novor_output_filename_cmd = 'mv ' +\
            '\"' + os.path.join(args['iodir'], mgf_input_name + '.novor.csv') + '\" ' +\
            '\"' + os.path.join(args['iodir'], args['novor_files'][-1]) + '\"'
        subprocess.call(set_novor_output_filename_cmd, shell = True)

        args['pn_files'].append(os.path.join(args['iodir'], mgf_input_name + '_' + tol + '.mgf.out'))
        set_pn_output_filename_cmd = 'mv ' +\
            '\"' + os.path.join(args['iodir'], mgf_input_name + '.mgf.out') + '\" ' +\
            '\"' + os.path.join(args['iodir'], args['novor_files'][-1]) + '\"'
        subprocess.call(set_pn_output_filename_cmd, shell = True)

    sys.exit(0)

    return args

def set_global_vars(args):

    config.iodir.append(args['iodir'])

    if 'quiet' in args:
        config.verbose[0] = False

    if 'train' in args:
        config.run_type[0] = 'train'
    elif 'optimize' in args:
        config.run_type[0] = 'optimize'
    elif 'test' in args:
        config.run_type[0] = 'test'

    #for tol in sorted(args['frag_mass_tols']):
    #    config.frag_mass_tols.append(tol)

    if 'novor_files' in args:
        novor_files_local, _ = order_inputs(
            args['novor_files'], args['frag_mass_tols'])
        #novor_files_local, _ = order_inputs(
        #    args['novor_files'], args['frag_mass_tols'])
        for novor_file in novor_files_local:
            config.novor_files.append(novor_file)

        config.alg_list.append('novor')
        config.alg_tols_dict['novor'] = OrderedDict(
            zip(config.frag_mass_tols,
                [os.path.basename(novor_file) for novor_file in config.novor_files]))

    if 'peaks_files' in args:
        peaks_files_local, _ = order_inputs(
            args['peaks_files'], config.frag_mass_tols)
        #peaks_files_local, _ = order_inputs(
        #    args['peaks_files'], args['frag_mass_tols'])
        for peaks_file in peaks_files_local:
            config.peaks_files.append(peaks_file)

        config.alg_list.append('peaks')
        config.alg_tols_dict['peaks'] = OrderedDict(
            zip(config.frag_mass_tols,
                [os.path.basename(peaks_file) for peaks_file in config.peaks_files]))

    if 'pn_files' in args:
        pn_files_local, _ = order_inputs(
            args['pn_files'], config.frag_mass_tols)
        #pn_files_local, _ = order_inputs(
        #    args['pn_files'], args['frag_mass_tols'])
        for pn_file in pn_files_local:
            config.pn_files.append(pn_file)

        config.alg_list.append('pn')
        config.alg_tols_dict['pn'] = OrderedDict(
            zip(config.frag_mass_tols,
                [os.path.basename(pn_file) for pn_file in config.pn_files]))

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

    tol_alg_dict_local = invert_dict_of_lists(config.alg_tols_dict)
    for k, v in tol_alg_dict_local.items():
        config.tol_alg_dict[k] = v

    for tol in config.frag_mass_tols:
        config.tol_basenames_dict[tol] = []
    for alg in config.alg_tols_dict:
        for tol in config.alg_tols_dict[alg]:
            config.tol_basenames_dict[tol] += [config.alg_tols_dict[alg][tol]]

    if 'min_len' in args:
        config.min_len[0] = args['min_len']
    if 'min_prob' in args:
        config.min_prob[0] = args['min_prob']

    if 'db_search_ref_file' in args:
        config.db_search_ref_file[0] = args['db_search_ref_file']
    if 'fasta_ref_file' in args:
        config.fasta_ref_file[0] = args['fasta_ref_file']
        
    if 'cores' in args:
        config.cores[0] = args['cores']

def order_by_tol(args):
    for alg in config.accepted_algs:
        for arg in args:
            if alg in arg:
                args[alg + '_files'] =\
                    list(list(
                        zip(*(sorted(zip(args[alg + '_files'],
                                         config.frag_mass_tols),
                                     key = lambda x: x[1])))
                        )[0])
                #args[alg + '_files'] =\
                #    list(list(
                #        zip(*(sorted(zip(args[alg + '_files'],
                #                         args['frag_mass_tols']),
                #                     key = lambda x: x[1])))
                #        )[0])
    return args

def order_inputs(file_names, tols):
    tol_index = [i for i in range(len(tols))]
    ordered_index, ordered_tols = zip(*sorted(
        zip(tol_index, tols), key = lambda x: x[1]))
    ordered_file_names = list(zip(*sorted(
        zip(ordered_index, file_names), key = lambda x: x[0])))[1]
    return list(ordered_file_names), list(ordered_tols)

def invert_dict_of_lists(d):
    values = set(a for b in d.values() for a in b)
    values = sorted(list(values))
    invert_d = OrderedDict((new_k, [k for k, v in d.items() if new_k in v]) for new_k in values)
    return invert_d