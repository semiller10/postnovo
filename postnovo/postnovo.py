import input
import consensus
import classifier

import getopt
import sys
import time

from config import *
from utils import (save_pkl_objects, load_pkl_objects,
                   save_json_objects, load_json_objects,
                   verbose_print)

from os.path import join, exists
from multiprocessing import cpu_count


def main(argv):
    start_time = time.time()

    #user_args = parse_user_args(argv)

    #save_json_objects(_test_dir, **{'user_args': user_args})
    user_args, = load_json_objects(_test_dir, 'user_args')

    set_global_vars(user_args)

    alg_df_name_dict, tol_df_name_dict, alg_tol_dict = input.load_files()

    save_pkl_objects(_test_dir, **{'alg_df_name_dict': alg_df_name_dict,
                                  'tol_df_name_dict': tol_df_name_dict,
                                  'alg_tol_dict': alg_tol_dict})
    #alg_df_name_dict, tol_df_name_dict, alg_tol_dict =\
    #    load_pkl_objects(_test_dir, 'alg_df_name_dict',
    #                     'tol_df_name_dict',
    #                     'alg_tol_dict')

    ## Object schema:
    ## alg_df_name_dict = odict('novor': novor input df, 'pn': pn input df)
    ## tol_df_name_dict = odict('0.4': ['proteome-0.4.novor.csv', 'proteome-0.4.mgf.out'], '0.5': ['proteome-0.5.novor.csv', 'proteome-0.5.mgf.out'])
    ## alg_tol_dict = odict('novor': odict('0.4': 'proteome-0.4.novor.csv', '0.5': 'proteome-0.5.novor.csv'),
    ##                     'pn': odict('0.4': 'proteome-0.4.mgf.out', '0.5': 'proteome-0.5.mgf.out'))

    prediction_df = consensus.make_prediction_df(alg_df_name_dict, tol_df_name_dict, alg_tol_dict)

    save_pkl_objects(_test_dir, **{'consensus_prediction_df': prediction_df})
    #prediction_df, = load_pkl_objects(_test_dir, 'consensus_prediction_df')

    classifier.classify(prediction_df = prediction_df)
    #classifier.classify()

    verbose_print('total time elapsed:', time.time() - start_time)

def set_global_vars(user_args):

    if 'train' in user_args:
        _run_type[0] = 'train'
    elif 'optimize' in user_args:
        _run_type[0] = 'optimize'
    elif 'test' in user_args:
        _run_type[0] = 'test'

    if 'quiet' in user_args:
        _verbose[0] = False

    if 'novor_files' in user_args:
        _novor_files = user_args['novor_files']
        _novor_tols = user_args['novor_tols']
        _alg_list.append('novor')

    if 'peaks_files' in user_args:
        _peaks_files = user_args['peaks_files'] 
        _peaks_tols = user_args['peaks_tols']
        _alg_list.append('peaks')

    if 'pn_files' in user_args:
        _pn_files = user_args['pn_files']
        _pn_tols = user_args['pn_tols']
        _alg_list.append('pn')

    if 'min_len' in user_args:
        _min_len[0] = user_args['min_len']
    if 'min_prob' in user_args:
        _min_prob[0] = user_args['min_prob']

    if 'ref_file' in user_args:
        _ref_file[0] = user_args['ref_file']

    if 'cores' in user_args:
        _cores[0] = user_args['cores']

def parse_user_args(argv):
    ''' Return command line args as dict '''

    help_str = ('postnovo.py\n\
    --train\n\
    --novorfiles <"novorfile0.3, novorfile0.5, ...">\n\
    --novortols <"0.3, 0.5, ...">\n\
    --peaksfiles <"peaksfile0.3 peaksfile0.5, ...">\n\
    --peakstols <"0.3, 0.5, ...">\n\
    --pnfiles <"pnfile0.3, pnfile0.5, ...">\n\
    --pntols <"0.3, 0.5, ...">\n\
    --minlen <9>\n\
    --minprob <0.75>\n\
    --reffile <reffile>\n\
    --cores <3>')
    
    user_args = {}

    # Take in args
    if not argv:
        print(help_str)

    try:
        opts, args = getopt.getopt(argv,
                                   'htqson:u:p:v:e:w:l:b:r:c:',
                                   ['help', 'train', 'quiet', 'test', 'optimize',
                                    'novorfiles=', 'novortols=',
                                    'peaksfiles=', 'peakstols=',
                                    'pnfiles=', 'pntols=',
                                    'minlen=', 'minprob=',
                                    'reffile=',
                                    'cores='])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(1)

    # Parse each option
    for opt, arg in opts:

        if opt in ('-h', '--help'):
            print(help_str)
            sys.exit(0)

        elif opt in ('-t', '--train'):
            user_args['train'] = True

        elif opt in ('-q', '--quiet'):
            user_args['quiet'] = True

        elif opt in ('-s', '--test'):
            user_args['test'] = True

        elif opt in ('-o', '--optimize'):
            user_args['optimize'] = True

        elif opt in ('-n', '--novorfiles'):
            novor_files = arg.split(',')
            novor_files = [f.strip() for f in novor_files]
            for i, file_name in enumerate(novor_files):
                if '.novor.csv' not in file_name:
                    print(file_name + ' must have novor.csv file extension')
                    sys.exit(1)
                elif exists(join(_userfiles_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                novor_files[i] = join(_userfiles_dir, file_name)
            user_args['novor_files'] = novor_files

        elif opt in ('u', '--novortols'):
            novor_tols = arg.split(',')
            novor_tols = [tol.strip() for tol in novor_tols]
            for tol in novor_tols:
                if tol not in _accepted_mass_tols:
                    print(tol + ' must be in list of accepted fragment mass tolerances: ' + \
                        ' '.join(novor_tols))
                    sys.exit(1)
            user_args['novor_tols'] = novor_tols

        elif opt in ('p', '--peaksfiles'):
            peaks_files = arg.split(',')
            peaks_files = [f.strip() for f in peaks_files]
            for i, file_name in peaks_files:
                if '.csv' not in file_name:
                    print(file_name + ' must have csv file extension')
                    sys.exit(1)
                elif exists(join(_userfiles_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                peaks_files[i] = join(_userfiles_dir, file_name)
            user_args['peaks_files'] = peaks_files

        elif opt in ('v', '--peakstols'):
            peaks_tols = arg.split(',')
            peaks_tols = [tol.strip() for tol in peaks_tols]
            for tol in peaks_tols:
                if tol not in _accepted_mass_tols:
                    print(tol + ' must be in list of accepted fragment mass tolerances: ' + \
                        ' '.join(novor_tols))
                    sys.exit(1)
            user_args['novor_tols'] = novor_tols

        elif opt in ('e', '--pnfiles'):
            pn_files = arg.split(',')
            pn_files = [f.strip() for f in pn_files]
            for i, file_name in enumerate(pn_files):
                if '.mgf.out' not in file_name:
                    print(file_name + ' must have mgf.out file extension')
                    sys.exit(1)
                elif exists(join(_userfiles_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                pn_files[i] = join(_userfiles_dir, file_name)
            user_args['pn_files'] = pn_files

        elif opt in ('w', '--pntols'):
            pn_tols = arg.split(',')
            pn_tols = [tol.strip() for tol in pn_tols]
            for tol in pn_tols:
                if tol not in _accepted_mass_tols:
                    print(tol + ' must be in list of accepted fragment mass tolerances: ' + \
                        ' '.join(pn_tols))
                    sys.exit(1)
            user_args['pn_tols'] = pn_tols

        elif opt in ('l', '--minlen'):
            try:
                min_len = int(arg)
            except ValueError:
                print('Minimum reported sequence length must be an integer >0')
                sys.exit(1)
            if user_args['min_len'] < _train_consensus_len:
                print('Sequences shorter than length ' + str(_train_consensus_len) + ' are not supported')
                sys.exit(1)
            user_args['min_len'] = min_len

        elif opt in ('b', '--minprob'):
            try:
                min_prob = float(arg)
            except ValueError:
                print('Minimum reported sequence probability must be a number between 0 and 1')
                sys.exit(1)
            if min_prob <= 0 or min_prob >= 1:
                print('Minimum reported sequence probability must be a number between 0 and 1')
                sys.exit(1)
            user_args['min_prob'] = min_prob

        elif opt in ('r', '--reffile'):
            if exists(join(_userfiles_dir, arg)) is False:
                print(arg + ' must be in postnovo/userfiles')
                sys.exit(1)
            user_args['ref_file'] = join(_userfiles_dir, arg)

        elif opt in ('c', '--cores'):
            if not float(arg).is_integer():
                print('Specify an integer number of cores')
                sys.exit(1)
            if int(arg) > cpu_count() or int(arg) < 1:
                print(str(cpu_count()) + ' cores are available')
                sys.exit(1)
            user_args['cores'] = int(arg)

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

    if 'novor_files' not in user_args:
        print('Novor input is required')
        sys.exit(1)

    if 'pn_files' not in user_args:
        print('PepNovo+ input is required')
        sys.exit(1)

    user_args = order_by_mass_tol(user_args)

    return user_args

def order_by_mass_tol(user_args):

    for alg in _accepted_algs:
        for arg in user_args:
            if alg in arg:
                user_args[alg + '_files'], user_args[alg + '_tols'] =\
                    zip(*(sorted(zip(user_args[alg + '_files'],
                                     user_args[alg + '_tols']),
                                 key = lambda x: x[1])))
        
    return user_args

if __name__ == '__main__':
    main(sys.argv[1:])