import input
import consensus
import classifier

import getopt
import sys
import time

from config import (test_dir, user_files_dir,
                    accepted_algs, accepted_mass_tols,
                    run_type, train_consensus_len,
                    default_min_prob)
from utils import (save_pkl_objects, load_pkl_objects,
                   save_json_objects, load_json_objects,
                   verbose_print)

from os.path import join, exists
from multiprocessing import cpu_count

def main(argv):
    start_time = time.time()

    #user_args = parse_user_args(argv)

    #save_json_objects(test_dir, **{'user_args': user_args})
    user_args, = load_json_objects(test_dir, 'user_args')

    alg_list, alg_df_name_dict, tol_df_name_dict, alg_tol_dict = input.load_files(user_args)

    save_pkl_objects(test_dir, **{'alg_df_name_dict': alg_df_name_dict,
                                  'tol_df_name_dict': tol_df_name_dict,
                                  'alg_tol_dict': alg_tol_dict,
                                  'alg_list': alg_list})
    #save_pkl_objects(test_dir, **{'alg_list_test': alg_list})
    #alg_df_name_dict, tol_df_name_dict, alg_tol_dict, alg_list =\
    #    load_pkl_objects(test_dir, 'alg_df_name_dict',
    #                     'tol_df_name_dict',
    #                     'alg_tol_dict',
    #                     'alg_list')
    #alg_list, = load_pkl_objects(test_dir, 'alg_list')

    ## Object schema:
    ## alg_df_name_dict = odict('novor': novor input df, 'pn': pn input df)
    ## tol_df_name_dict = odict('0.4': ['proteome-0.4.novor.csv', 'proteome-0.4.mgf.out'], '0.5': ['proteome-0.5.novor.csv', 'proteome-0.5.mgf.out'])
    ##alg_tol_dict = odict('novor': odict('0.4': 'proteome-0.4.novor.csv', '0.5': 'proteome-0.5.novor.csv'),
    ##                     'pn': odict('0.4': 'proteome-0.4.mgf.out', '0.5': 'proteome-0.5.mgf.out'))

    prediction_df = consensus.make_prediction_df(alg_df_name_dict, tol_df_name_dict, alg_tol_dict,
                                                 user_args['min_len'], user_args['cores'], alg_list)

    save_pkl_objects(test_dir, **{'consensus_prediction_df': prediction_df})
    #prediction_df, = load_pkl_objects(test_dir, 'consensus_prediction_df')

    classifier.classify(user_args['ref_file'], user_args['cores'], alg_list, min_prob = user_args['min_prob'], prediction_df = prediction_df)
    #classifier.classify(user_args['ref_file'], user_args['cores'], alg_list)

    verbose_print('total time elapsed:', time.time() - start_time)

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
    user_args['train'] = False
    user_args['test'] = False
    user_args['optimize'] = False
    user_args['min_len'] = train_consensus_len
    user_args['min_prob'] = default_min_prob
    user_args['cores'] = 1

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
            run_type[0] = 'train'

        elif opt in ('-q', '--quiet'):
            user_args['quiet'] = True
            utils.verbose[0] = False

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
                elif exists(join(user_files_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                novor_files[i] = join(user_files_dir, file_name)
            user_args['novor_files'] = novor_files

        elif opt in ('u', '--novortols'):
            novor_tols = arg.split(',')
            novor_tols = [tol.strip() for tol in novor_tols]
            for tol in novor_tols:
                if tol not in accepted_mass_tols:
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
                elif exists(join(user_files_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                peaks_files[i] = join(user_files_dir, file_name)
            user_args['peaks_files'] = peaks_files

        elif opt in ('v', '--peakstols'):
            peaks_tols = arg.split(',')
            peaks_tols = [tol.strip() for tol in peaks_tols]
            for tol in peaks_tols:
                if tol not in accepted_mass_tols:
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
                elif exists(join(user_files_dir, file_name)) is False:
                    print(file_name + ' must be in postnovo/userfiles')
                    sys.exit(1)
                pn_files[i] = join(user_files_dir, file_name)
            user_args['pn_files'] = pn_files

        elif opt in ('w', '--pntols'):
            pn_tols = arg.split(',')
            pn_tols = [tol.strip() for tol in pn_tols]
            for tol in pn_tols:
                if tol not in accepted_mass_tols:
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
            if exists(join(user_files_dir, arg)) is False:
                print(arg + ' must be in postnovo/userfiles')
                sys.exit(1)
            user_args['ref_file'] = join(user_files_dir, arg)

        elif opt in ('c', '--cores'):
            if not float(arg).is_integer():
                print('Specify an integer number of cores')
                sys.exit(1)
            if int(arg) > cpu_count() or int(arg) < 1:
                print(str(cpu_count()) + ' cores are available')
                sys.exit(1)
            user_args['cores'] = int(arg)

    if user_args['test']:
        if 'ref_file' not in user_args:
            print('Testing requires a protein reffile')
            sys.exit(1)
        if user_args['optimize']:
            print('Testing and optimization are exclusive')
            sys.exit(1)
        # --test overrides --train
        run_type[0] = 'test'

    if user_args['optimize']:
        if 'ref_file' not in user_args:
            print('Model optimization requires a protein reffile')
            sys.exit(1)
        # --optimize overrides --train
        run_type[0] = 'optimize'

    if user_args['train']:
        if 'ref_file' not in user_args:
            print('Training requires a protein reffile')
            sys.exit(1)

    if 'novor_files' not in user_args:
        print('Novor input is required')
        sys.exit(1)

    if 'pn_files' not in user_args:
        print('PepNovo+ input is required')
        sys.exit(1)

    if user_args['min_len'] < train_consensus_len:
        print('Sequences shorter than length ' + str(train_consensus_len) + ' are not supported')
        sys.exit(1)

    user_args = order_by_mass_tol(user_args)

    return user_args

def order_by_mass_tol(user_args):

    for alg in accepted_algs:
        for arg in user_args:
            if alg in arg:
                user_args[alg + '_files'], user_args[alg + '_tols'] =\
                    zip(*(sorted(zip(user_args[alg + '_files'],
                                     user_args[alg + '_tols']),
                                 key = lambda x: x[1])))
        
    return user_args

if __name__ == '__main__':
    main(sys.argv[1:])