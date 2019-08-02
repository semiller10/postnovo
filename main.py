''' Postnovo entry point. '''

#Copyright 2018, Samuel E. Miller. All rights reserved.
#Postnovo is publicly available for non-commercial uses.
#Licensed under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
#See postnovo/LICENSE.txt.

import classifier
import config
import consensus
import input
import interspec
import masstol
import singlealg
import userargs
import utils

import os
import pandas as pd
import sys

from collections import OrderedDict
from time import time

def main():
    '''
    Entry point of Postnovo app.
    '''

    start_time = time()

    test_argv = None

    userargs.setup(test_argv)

    if config.globals['Retrain']:
        classifier.train_models()
    else:
        input.parse()

        single_alg_prediction_df = singlealg.do_single_alg_procedure()
        ##REMOVE: for debugging
        #utils.save_pkl_objects(
        #    config.globals['Output Directory'], 
        #    **{'single_alg_prediction_df.pkl': single_alg_prediction_df})
        ##REMOVE: for debugging
        #single_alg_prediction_df = utils.load_pkl_objects(
        #    config.globals['Output Directory'], 
        #    'single_alg_prediction_df.pkl')

        consensus_prediction_df = consensus.do_consensus_procedure()
        ##REMOVE: for debugging
        #utils.save_pkl_objects(
        #    config.globals['Output Directory'], 
        #    **{'consensus_prediction_df.pkl': consensus_prediction_df})
        ##REMOVE: for debugging
        #consensus_prediction_df = utils.load_pkl_objects(
        #    config.globals['Output Directory'], 
        #    'consensus_prediction_df.pkl')

        prediction_df = pd.concat(
            [single_alg_prediction_df.reset_index(), consensus_prediction_df.reset_index()], 
            ignore_index=True)
        if 'index' in prediction_df.columns:
            prediction_df.drop('index', axis=1, inplace=True)

        prediction_df = masstol.do_mass_tol_procedure(prediction_df)
    
        prediction_df = interspec.do_interspec_procedure(prediction_df)

        classifier.classify(prediction_df)
        ##REMOVE: for debugging
        #classifier.classify(None)

    print('Postnovo successfully completed')
    utils.verbose_print('Total time elapsed:', time() - start_time)

    return

if __name__ == '__main__':
    main()