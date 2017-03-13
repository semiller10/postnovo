''' Command line script: this module processes user input '''

import input
import consensus
import masstol
import interspec
import classifier

import getopt
import sys
import subprocess
import datetime
import time

from config import *
from utils import *

from os.path import join, exists, basename, abspath, splitext, dirname
from multiprocessing import cpu_count
from itertools import combinations, product


def main(argv):
    start_time = time.time()

    #user_args = parse_user_args(argv)
    #save_json_objects(test_dir, **{'user_args': user_args})
    user_args = load_json_objects(test_dir, 'user_args')

    #user_args = run_denovogui(user_args)
    set_global_vars(user_args)

    alg_basename_dfs_dict = input.load_files()
    save_pkl_objects(test_dir, **{'alg_basename_dfs_dict': alg_basename_dfs_dict})
    #alg_basename_dfs_dict = load_pkl_objects(test_dir, 'alg_basename_dfs_dict')
    ## example:
    ## alg_basename_dfs_dict = odict('novor': novor input df, 'pn': pn input df)

    prediction_df = consensus.make_prediction_df(alg_basename_dfs_dict)
    save_pkl_objects(test_dir, **{'consensus_prediction_df': prediction_df})
    #prediction_df = load_pkl_objects(test_dir, 'consensus_prediction_df')

    prediction_df = masstol.update_prediction_df(prediction_df)
    save_pkl_objects(test_dir, **{'mass_tol_prediction_df': prediction_df})
    #prediction_df = load_pkl_objects(test_dir, 'mass_tol_prediction_df')

    prediction_df = interspec.update_prediction_df(prediction_df)
    save_pkl_objects(test_dir, **{'interspec_prediction_df': prediction_df})
    #prediction_df = load_pkl_objects(test_dir, 'interspec_prediction_df')

    classifier.classify(prediction_df = prediction_df)
    #classifier.classify()

    verbose_print('total time elapsed:', time.time() - start_time)

def parse_user_args(argv):
    ''' Return command line args as dict '''

    help_str = ('postnovo.py\n\
    --quiet\n\
    --denovogui_path <"C:\Program Files (x86)\DeNovoGUI-1.15.5-windows\DeNovoGUI-1.15.5\DeNovoGUI-1.15.5.jar">\n\
    --denovogui_mgf_path <"spectra.mgf">\n\
    --train\n\
    --test\n\
    --optimize\n\
    --frag_mass_tols <"0.3, 0.5">\n\
    --novor_files <"novor_output_0.3.novor.csv, novor_output_0.5.novor.csv">\n\
    --peaks_files <"peaks_output_0.3.csv peaks_output_0.5.csv">\n\
    --pn_files <"pn_output_0.3.mgf.out, pn_output_0.5.mgf.out">\n\
    --min_len <9>\n\
    --min_prob <0.75>\n\
    --ref_file <reffile>\n\
    --cores <3>')
    
    user_args = {}

    # Take in args
    if not argv:
        print(help_str)

    try:
        opts, args = getopt.getopt(argv,
                                   'hqtsod:m:f:n:p:e:l:b:r:c:',
                                   ['help', 'quiet', 'train', 'test', 'optimize',
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

        if opt in ('-h', '--help'):
            print(help_str)
            sys.exit(0)

        elif opt in ('-q', '--quiet'):
            user_args['quiet'] = True

        elif opt in ('-t', '--train'):
            user_args['train'] = True

        elif opt in ('-s', '--test'):
            user_args['test'] = True

        elif opt in ('-o', '--optimize'):
            user_args['optimize'] = True

        elif opt in ('-d', '--denovogui_path'):
            denovogui_path = abspath(arg)
            if exists(denovogui_path) is False:
                print(arg + ' does not exist')
                sys.exit(1)
            user_args['denovogui_path'] = denovogui_path

        elif opt in ('-m', '--denovogui_mgf_path'):
            denovogui_mgf_path = abspath(arg)
            if exists(denovogui_mgf_path) is False:
                print(denovogui_mgf_path + ' does not exist')
                sys.exit(1)
            user_args['denovogui_mgf_path'] = denovogui_mgf_path

        elif opt in ('-f', '--fragment_mass_tols'):
            frag_mass_tols = arg.split(',')
            frag_mass_tols = [tol.strip() for tol in frag_mass_tols]
            for tol in frag_mass_tols:
                if tol not in accepted_mass_tols:
                    print(tol + ' must be in list of accepted fragment mass tolerances: ' +\
                        ', '.join(accepted_mass_tols))
                    sys.exit(1)
            user_args['frag_mass_tols'] = frag_mass_tols

        elif opt in ('-n', '--novor_files'):
            novor_files = arg.split(',')
            novor_files = [f.strip() for f in novor_files]
            for i, file_name in enumerate(novor_files):
                if '.novor.csv' not in file_name:
                    print(file_name + ' must have novor.csv file extension')
                    sys.exit(1)
                elif exists(join(userfiles_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                novor_files[i] = join(userfiles_dir, file_name)
            user_args['novor_files'] = novor_files

        elif opt in ('-p', '--peaks_files'):
            peaks_files = arg.split(',')
            peaks_files = [f.strip() for f in peaks_files]
            for i, file_name in peaks_files:
                if '.csv' not in file_name:
                    print(file_name + ' must have csv file extension')
                    sys.exit(1)
                elif exists(join(userfiles_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                peaks_files[i] = join(userfiles_dir, file_name)
            user_args['peaks_files'] = peaks_files

        elif opt in ('-e', '--pn_files'):
            pn_files = arg.split(',')
            pn_files = [f.strip() for f in pn_files]
            for i, file_name in enumerate(pn_files):
                if '.mgf.out' not in file_name:
                    print(file_name + ' must have mgf.out file extension')
                    sys.exit(1)
                elif exists(join(userfiles_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                pn_files[i] = join(userfiles_dir, file_name)
            user_args['pn_files'] = pn_files

        elif opt in ('-l', '--min_len'):
            try:
                min_len = int(arg)
            except ValueError:
                print('Minimum reported sequence length must be an integer >0')
                sys.exit(1)
            if user_args['min_len'] < train_consensus_len:
                print('Sequences shorter than length ' + str(train_consensus_len) + ' are not supported')
                sys.exit(1)
            user_args['min_len'] = min_len

        elif opt in ('-b', '--min_prob'):
            try:
                min_prob = float(arg)
            except ValueError:
                print('Minimum reported sequence probability must be a number between 0 and 1')
                sys.exit(1)
            if min_prob <= 0 or min_prob >= 1:
                print('Minimum reported sequence probability must be a number between 0 and 1')
                sys.exit(1)
            user_args['min_prob'] = min_prob

        elif opt in ('-r', '--ref_file'):
            if exists(join(userfiles_dir, arg)) is False:
                print(arg + ' must be in postnovo/userfiles')
                sys.exit(1)
            user_args['ref_file'] = join(userfiles_dir, arg)

        elif opt in ('-c', '--cores'):
            if not float(arg).is_integer():
                print('Specify an integer number of cores')
                sys.exit(1)
            if int(arg) > cpu_count() or int(arg) < 1:
                print(str(cpu_count()) + ' cores are available')
                sys.exit(1)
            user_args['cores'] = int(arg)

        else:
            print('Unrecognized option ' + opt)
            sys.exit(1)

    if user_args['train']:
        if 'ref_file' not in user_args:
            print('Training requires a protein reffile')
            sys.exit(1)
        if 'test' in user_args or 'optimize' in user_args:
            print('Train, test and optimize options are exclusive')

    if user_args['test']:
        if 'ref_file' not in user_args:
            print('Testing requires a protein reffile')
            sys.exit(1)
        if 'train' in user_args or 'optimize' in user_args:
            print('Train, test and optimize options are mutually exclusive')
            sys.exit(1)

    if user_args['optimize']:
        if 'ref_file' not in user_args:
            print('Model optimization requires a protein reffile')
            sys.exit(1)
        if 'train' in user_args or 'test' in user_args:
            print('Train, test and optimize options are exclusive')

    if 'frag_mass_tols' not in user_args:
        print('Fragment mass tolerances must be specified')
        sys.exit(1)

    if 'denovogui_path' in user_args:
        if 'denovogui_mgf_path' not in user_args:
            print('denovogui_mgf_path command line argument also needed:\
            place mgf file in postnovo/userfiles to run DeNovoGUI')
            sys.exit(1)
    if 'denovogui_mgf_path' in user_args:
        if 'denovogui_path' in user_args:
            print('denovogui_path command line argument also needed')
            sys.exit(1)

    if 'denovogui_path' in user_args:
        if 'frag_mass_tols' not in user_args:
            print('Specify the fragment mass tolerances used in DeNovoGUI.')
            sys.exit(1)
    else:
        if 'novor_files' in user_args:
            if len(user_args['frag_mass_tols']) != len(user_args['novor_files']):
                print('Fragment mass tolerances must align and be of same length as file inputs')
                sys.exit(1)
        elif 'novor_files' not in user_args:
            print('Novor input is required')
            sys.exit(1)

        if 'pn_files' in user_args:
            if len(user_args['frag_mass_tols']) != len(user_args['pn_files']):
                print('Fragment mass tolerances must align and be of same length as file inputs')
                sys.exit(1)
        elif 'pn_files' not in user_args:
            print('PepNovo+ input is required')
            sys.exit(1)

    user_args = order_by_tol(user_args)
    return user_args

def run_denovogui(user_args):

    user_args['novor_files'] = []
    user_args['pn_files'] = []

    denovogui_param_args = OrderedDict().fromkeys(['-out', '-frag_tol',
                                                   '-fixed_mods', '-variable_mods',
                                                   '-pepnovo_hitlist_length', '-novor_fragmentation', '-novor_mass_analyzer'])
    denovogui_param_args['-fixed_mods'] = '\"' + fixed_mod + '\"'
    denovogui_param_args['-variable_mods'] = '\"' + variable_mod + '\"'
    denovogui_param_args['-pepnovo_hitlist_length'] = str(seqs_reported_per_alg_dict['pn'])
    denovogui_param_args['-novor_fragmentation'] = '\"' + frag_method + '\"'
    denovogui_param_args['-novor_mass_analyzer'] = '\"' + frag_mass_analyzer + '\"'

    denovogui_args = OrderedDict().fromkeys(['-spectrum_files', '-output_folder', '-id_params',
                                             '-pepnovo', '-novor', '-directag', '-threads'])
    mgf_input_name = splitext(basename(user_args['denovogui_mgf_path']))[0]
    denovogui_args['-spectrum_files'] = '\"' + user_args['denovogui_mgf_path'] + '\"'
    denovogui_args['-output_folder'] = '\"' + userfiles_dir + '\"'
    denovogui_args['-pepnovo'] = '1'
    denovogui_args['-novor'] = '1'
    denovogui_args['-directag'] = '0'
    denovogui_args['-threads'] = str(user_args['cores'])

    for tol in user_args['frag_mass_tols']:

        denovogui_param_file_cmd = 'java -cp ' +\
            '\"' + user_args['denovogui_path'] + '\"' +\
            ' com.compomics.denovogui.cmd.IdentificationParametersCLI '
        denovogui_param_args['-frag_tol'] = tol
        denovogui_param_args['-out'] = '\"' +\
            join(userfiles_dir,
                 mgf_input_name + '_' + tol + '.par') + '\"'
        for opt, arg in denovogui_param_args.items():
            denovogui_param_file_cmd += opt + ' ' + arg + ' '
        subprocess.call(denovogui_param_file_cmd, shell = True)

        denovogui_cmd = 'java -cp ' +\
            '\"' + user_args['denovogui_path'] + '\"' +\
            ' com.compomics.denovogui.cmd.DeNovoCLI '
        denovogui_args['-id_params'] = denovogui_param_args['-out']
        for opt, arg in denovogui_args.items():
            denovogui_cmd += opt + ' ' + arg + ' '
        subprocess.call(denovogui_cmd, shell = True)

        user_args['novor_files'].append(mgf_input_name + '_' + tol + '.novor.csv')
        set_novor_output_filename_cmd = 'mv ' +\
            '\"' + join(userfiles_dir, mgf_input_name + '.novor.csv') + '\" ' +\
            '\"' + join(userfiles_dir, user_args['novor_files'][-1]) + '\"'
        subprocess.call(set_novor_output_filename_cmd, shell = True)

        user_args['pn_files'].append(mgf_input_name + '_' + tol + '.mgf.out')
        set_pn_output_filename_cmd = 'mv ' +\
            '\"' + join(userfiles_dir, mgf_input_name + '.mgf.out') + '\" ' +\
            '\"' + join(userfiles_dir, user_args['novor_files'][-1]) + '\"'
        subprocess.call(set_pn_output_filename_cmd, shell = True)

    return user_args

def set_global_vars(user_args):

    if 'quiet' in user_args:
        verbose[0] = False

    if 'train' in user_args:
        run_type[0] = 'train'
    elif 'optimize' in user_args:
        run_type[0] = 'optimize'
    elif 'test' in user_args:
        run_type[0] = 'test'

    for tol in sorted(user_args['frag_mass_tols']):
        frag_mass_tols.append(tol)

    if 'novor_files' in user_args:
        novor_files_local, _ = order_inputs(
            user_args['novor_files'], user_args['frag_mass_tols'])
        for novor_file in novor_files_local:
            novor_files.append(novor_file)

        alg_list.append('novor')
        alg_tols_dict['novor'] = OrderedDict(
            zip(frag_mass_tols,
                [basename(novor_file) for novor_file in novor_files]))

    if 'peaks_files' in user_args:
        peaks_files_local, _ = order_inputs(
            user_args['peaks_files'], user_args['frag_mass_tols'])
        for peaks_file in peaks_files_local:
            peaks_files.append(peaks_file)

        alg_list.append('peaks')
        alg_tols_dict['peaks'] = OrderedDict(
            zip(frag_mass_tols,
                [basename(peaks_file) for peaks_file in peaks_files]))

    if 'pn_files' in user_args:
        pn_files_local, _ = order_inputs(
            user_args['pn_files'], user_args['frag_mass_tols'])
        for pn_file in pn_files_local:
            pn_files.append(pn_file)

        alg_list.append('pn')
        alg_tols_dict['pn'] = OrderedDict(
            zip(frag_mass_tols,
                [basename(pn_file) for pn_file in pn_files]))

    for combo_level in range(2, len(alg_list) + 1):
        combo_level_combo_list = [combo for combo in combinations(alg_list, combo_level)]
        for alg_combo in combo_level_combo_list:
            alg_combo_list.append(alg_combo)

    # MultiIndex cols for prediction_df
    for alg in alg_list:
        is_alg_col_name = 'is ' + alg + ' seq'
        is_alg_col_names.append(is_alg_col_name)

    is_alg_col_multiindex_list = list(product((0, 1), repeat = len(alg_list)))
    for multiindex_key in is_alg_col_multiindex_list[1:]:
        is_alg_col_multiindex_keys.append(multiindex_key)

    tol_alg_dict_local = invert_dict_of_lists(alg_tols_dict)
    for k, v in tol_alg_dict_local.items():
        tol_alg_dict[k] = v

    for tol in frag_mass_tols:
        tol_basenames_dict[tol] = []
    for alg in alg_tols_dict:
        for tol in alg_tols_dict[alg]:
            tol_basenames_dict[tol] += [alg_tols_dict[alg][tol]]

    if 'min_len' in user_args:
        min_len[0] = user_args['min_len']
    if 'min_prob' in user_args:
        min_prob[0] = user_args['min_prob']

    if 'ref_file' in user_args:
        ref_file[0] = user_args['ref_file']

    if 'cores' in user_args:
        cores[0] = user_args['cores']

def order_by_tol(user_args):
    for alg in accepted_algs:
        for arg in user_args:
            if alg in arg:
                user_args[alg + '_files'] =\
                    zip(*(sorted(zip(user_args[alg + '_files'],
                                     user_args['frag_mass_tols']),
                                 key = lambda x: x[1])))
    return user_args

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
    main(sys.argv[1:])