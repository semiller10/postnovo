from postnovo import config
from postnovo import utils
from postnovo import userargs
from postnovo import input
from postnovo import consensus
from postnovo import masstol
from postnovo import interspec
from postnovo import classifier
from time import time


def main():
    start_time = time()

    args = userargs.setup()
    utils.save_json_objects(config.iodir[0], **{'args': args})
    #args = utils.load_json_objects(config.iodir[0], 'args')

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