import postnovo.config as config
import postnovo.utils as utils
import postnovo.userargs as userargs
import postnovo.input as input
import postnovo.consensus as consensus
import postnovo.masstol as masstol
import postnovo.interspec as interspec
import postnovo.classifier as classifier

#import config
#import utils
#import userargs
#import input
#import consensus
#import masstol
#import interspec
#import classifier

from time import time


def main():
    start_time = time()

    args = userargs.setup()
    utils.save_json_objects(config.iodir[0], **{'args': args})
    #args = utils.load_json_objects(r'C:\Users\Samuel\Documents\Visual Studio 2015\Projects\postnovo\test', 'args')
    #userargs.set_global_vars(args)

    alg_basename_dfs_dict = input.load_files()
    utils.save_pkl_objects(config.iodir[0], **{'alg_basename_dfs_dict': alg_basename_dfs_dict})
    #alg_basename_dfs_dict = utils.load_pkl_objects(config.iodir[0], 'alg_basename_dfs_dict')
    ## example:
    ## alg_basename_dfs_dict = odict('novor': novor input df, 'pn': pn input df)

    prediction_df = consensus.make_prediction_df(alg_basename_dfs_dict)
    utils.save_pkl_objects(config.iodir[0], **{'consensus_prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'consensus_prediction_df')

    prediction_df = masstol.update_prediction_df(prediction_df)
    utils.save_pkl_objects(config.iodir[0], **{'mass_tol_prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'mass_tol_prediction_df')

    prediction_df = interspec.update_prediction_df(prediction_df)
    utils.save_pkl_objects(config.iodir[0], **{'interspec_prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'interspec_prediction_df')

    classifier.classify(prediction_df = prediction_df)
    #classifier.classify()

    utils.verbose_print('total time elapsed:', time() - start_time)

if __name__ == '__main__':
    main()