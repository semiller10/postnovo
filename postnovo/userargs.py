''' Command line script: this module processes user input '''

import getopt
import sys
import subprocess
import datetime
import os

from postnovo import config
from postnovo import utils

from multiprocessing import cpu_count
from itertools import combinations, product


def setup():
    args = run_denovogui(
        order_by_tols(
            arg_cross_check(
                parse_args())))
    
    download_forest_dict(args)
    set_global_vars(args)
    
def parse_args():
    ''' Return command line args as dict '''

    help_str = ('postnovo.py\n\
    --iodir <"/home/postnovo_io">\n\
    --frag_mass_tols <"0.3, 0.5">\n\
    --novor_files <"novor_output_0.3.novor.csv, novor_output_0.5.novor.csv">\n\
    --peaks_files <"peaks_output_0.3.csv, peaks_output_0.5.csv">\n\
    --pn_files <"pn_output_0.3.mgf.out, pn_output_0.5.mgf.out">\n\
    --denovogui_path <"/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar">\n\
    --denovogui_mgf_path <"/home/ms_files/spectra.mgf">\n\
    --train\n\
    --test\n\
    --optimize\n\
    --min_len <9>\n\
    --min_prob <0.75>\n\
    --ref_file <reffile>\n\
    --cores <3>\n\
    --quiet')
    
    args = {}

    # Take in args
    if not argv:
        print(help_str)

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   ['help', 'quiet', 'train', 'test', 'optimize',
                                    'iodir=',
                                    'denovogui_path=', 'denovogui_mgf_path=',
                                    'frag_mass_tols=',
                                    'novor_files=', 'peaks_files=', 'pn_files=',
                                    'min_len=', 'min_prob=',
                                    'ref_file=',
                                    'cores='])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(1)

    # Parse each option
    for opt, arg in opts:

        if opt in ('--help'):
            print(help_str)
            sys.exit(0)

        elif opt in ('--quiet'):
            args['quiet'] = True

        elif opt in ('--train'):
            args['train'] = True

        elif opt in ('--test'):
            args['test'] = True

        elif opt in ('--optimize'):
            args['optimize'] = True

        elif opt in ('--iodir'):
            iodir = arg
            if os.path.exists(iodir) is False:
                print(iodir + ' does not exist')
                sys.exit(1)
            args['iodir'] = iodir

        elif opt in ('--denovogui_path'):
            denovogui_path = arg
            if os.path.exists(denovogui_path) is False:
                print(denovogui_path + ' does not exist')
                sys.exit(1)
            args['denovogui_path'] = denovogui_path

        elif opt in ('--denovogui_mgf_path'):
            denovogui_mgf_path = arg
            if os.path.exists(denovogui_mgf_path) is False:
                print(denovogui_mgf_path + ' does not exist')
                sys.exit(1)
            args['denovogui_mgf_path'] = denovogui_mgf_path

        elif opt in ('--frag_mass_tols'):
            frag_mass_tols = arg.split(',')
            frag_mass_tols = [tol.strip() for tol in frag_mass_tols]
            for tol in frag_mass_tols:
                if tol not in config.accepted_mass_tols:
                    print(tol + ' must be in list of accepted fragment mass tolerances: ' +\
                        ', '.join(config.accepted_mass_tols))
                    sys.exit(1)
            args['frag_mass_tols'] = frag_mass_tols

        elif opt in ('--novor_files'):
            novor_files = arg.split(',')
            novor_files = [f.strip() for f in novor_files]
            for i, file_name in enumerate(novor_files):
                if '.novor.csv' not in file_name:
                    print(file_name + ' must have novor.csv file extension')
                    sys.exit(1)
            args['novor_files'] = novor_files

        elif opt in ('--peaks_files'):
            peaks_files = arg.split(',')
            peaks_files = [f.strip() for f in peaks_files]
            for i, file_name in peaks_files:
                if '.csv' not in file_name:
                    print(file_name + ' must have csv file extension')
                    sys.exit(1)
            args['peaks_files'] = peaks_files

        elif opt in ('--pn_files'):
            pn_files = arg.split(',')
            pn_files = [f.strip() for f in pn_files]
            for i, file_name in enumerate(pn_files):
                if '.mgf.out' not in file_name:
                    print(file_name + ' must have mgf.out file extension')
                    sys.exit(1)
            args['pn_files'] = pn_files

        elif opt in ('--min_len'):
            try:
                min_len = int(arg)
            except ValueError:
                print('Minimum reported sequence length must be an integer >0')
                sys.exit(1)
            if min_len < config.train_consensus_len:
                print('Sequences shorter than length ' + str(config.train_consensus_len) + ' are not supported')
                sys.exit(1)
            args['min_len'] = min_len

        elif opt in ('--min_prob'):
            try:
                min_prob = float(arg)
            except ValueError:
                print('Minimum reported sequence probability must be a number between 0 and 1')
                sys.exit(1)
            if min_prob <= 0 or min_prob >= 1:
                print('Minimum reported sequence probability must be a number between 0 and 1')
                sys.exit(1)
            args['min_prob'] = min_prob

        elif opt in ('--ref_file'):
            args['ref_file'] = arg

        elif opt in ('--cores'):
            if not float(arg).is_integer():
                print('Specify an integer number of cores')
                sys.exit(1)
            if int(arg) > cpu_count() or int(arg) < 1:
                print(str(cpu_count()) + ' cores are available')
                sys.exit(1)
            args['cores'] = int(arg)

        else:
            print('Unrecognized option ' + opt)
            sys.exit(1)

    return args

def arg_cross_check(args):

    updated_args = {}

    for file_set in ['novor_files', 'peaks_files', 'pn_files']:
        if file_set in args:
            updated_args[file_set] = []
            for file_name in args[file_set]:
                file_path = os.path.join(args['iodir'], file_name)
                if os.path.exists(file_path) is False:
                    print(file_name + ' must be in ' + args['iodir'])
                    sys.exit(1)
                updated_args[file_set].append(file_path)

    if 'ref_file' in args:
        ref_path = os.path.join(args['iodir'], args['ref_file'])
        if os.path.exists(ref_path) is False:
            print(arg + ' must be in postnovo/userfiles')
            sys.exit(1)
        updated_args['ref_file'] = ref_path

    for option, arg in updated_args.items():
        args[option] = arg

    config.iodir.append(args['iodir'])

    if 'train' in args:
        if 'ref_file' not in args:
            print('Training requires a PSM reference file')
            sys.exit(1)
        if 'test' in args or 'optimize' in args:
            print('Train, test and optimize options are exclusive')

    if 'test' in args:
        if 'ref_file' not in args:
            print('Testing requires a PSM reference file')
            sys.exit(1)
        if 'train' in args or 'optimize' in args:
            print('Train, test and optimize options are mutually exclusive')
            sys.exit(1)

    if 'optimize' in args:
        if 'ref_file' not in args:
            print('Model optimization requires a PSM reference file')
            sys.exit(1)
        if 'train' in args or 'test' in args:
            print('Train, test and optimize options are exclusive')

    if 'frag_mass_tols' not in args:
        print('Fragment mass tolerance(s) of input must be specified')
        sys.exit(1)

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
                if len(args['frag_mass_tols']) != len(args[file_set]):
                    print('List of fragment mass tolerances must align with list of file inputs')
                    sys.exit(1)
            else:
                if file_set == 'novor_files':
                    print('Novor input is required')
                elif file_set == 'pn_files':
                    print('PepNovo+ input is required')
                sys.exit(1)

    return args

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

    for tol in args['frag_mass_tols']:

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

    if 'quiet' in args:
        config.verbose[0] = False

    if 'train' in args:
        config.run_type[0] = 'train'
    elif 'optimize' in args:
        config.run_type[0] = 'optimize'
    elif 'test' in args:
        config.run_type[0] = 'test'

    for tol in sorted(args['frag_mass_tols']):
        config.frag_mass_tols.append(tol)

    if 'novor_files' in args:
        novor_files_local, _ = order_inputs(
            args['novor_files'], args['frag_mass_tols'])
        for novor_file in novor_files_local:
            config.novor_files.append(novor_file)

        config.alg_list.append('novor')
        config.alg_tols_dict['novor'] = OrderedDict(
            zip(config.frag_mass_tols,
                [os.path.basename(novor_file) for novor_file in config.novor_files]))

    if 'peaks_files' in args:
        peaks_files_local, _ = order_inputs(
            args['peaks_files'], args['frag_mass_tols'])
        for peaks_file in peaks_files_local:
            config.peaks_files.append(peaks_file)

        config.alg_list.append('peaks')
        config.alg_tols_dict['peaks'] = OrderedDict(
            zip(config.frag_mass_tols,
                [os.path.basename(peaks_file) for peaks_file in config.peaks_files]))

    if 'pn_files' in args:
        pn_files_local, _ = order_inputs(
            args['pn_files'], args['frag_mass_tols'])
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

    if 'ref_file' in args:
        config.ref_file[0] = args['ref_file']
        
    if 'cores' in args:
        config.cores[0] = args['cores']

def order_by_tol(args):
    for alg in config.accepted_algs:
        for arg in args:
            if alg in arg:
                args[alg + '_files'] =\
                    list(list(
                        zip(*(sorted(zip(args[alg + '_files'],
                                         args['frag_mass_tols']),
                                     key = lambda x: x[1])))
                        )[0])
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

if __name__ == '__main__':
    main()