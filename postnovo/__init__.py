import sys

from time import time

if 'postnovo' in sys.modules:
    import postnovo.config as config
    import postnovo.utils as utils
    import postnovo.userargs as userargs
    import postnovo.input as input
    import postnovo.consensus as consensus
    import postnovo.masstol as masstol
    import postnovo.interspec as interspec
    import postnovo.classifier as classifier
else:
    import config
    import utils
    import userargs
    import input
    import consensus
    import masstol
    import interspec
    import classifier

def main():
    start_time = time()

    #test_argv = [
    #    '--deepnovo',
    #    '--iodir',
    #    'C:\\Users\\Samuel\\Downloads\\postnovo_test_121917',
    #    '--denovogui_fp',
    #    'C:\\Program Files (x86)\\DeNovoGUI-1.4.12-windows\\DeNovoGUI-1.15.11\\DeNovoGUI-1.15.11.jar',
    #    '--mgf_fp',
    #    '042017_toolik_core_27_2_1_1_sem.test.mgf',
    #    '--cores',
    #    '3'
    #    ]
    #test_argv = [
    #    '--filename',
    #    '042017_toolik_core_27_2_1_1_sem.test',
    #    '--deepnovo',
    #    '--iodir',
    #    'c:\\users\\samuel\\downloads\\postnovo_test_121917',
    #    '--cores',
    #    '3'
    #    ]

    #test_argv = ['--iodir',
    #             'C:\\Users\\Samuel\\Downloads',
    #             '--filename',
    #             'toolik_8_2_1_1',
    #             #'--denovogui_fp',
    #             #'C:\\Program Files (x86)\\DeNovoGUI-1.4.12-windows\\DeNovoGUI-1.15.11\\DeNovoGUI-1.15.11.jar',
    #             #'--mgf_fp',
    #             #'toolik_8_2_1_1.mgf',
    #             '--cores',
    #             '3']

    #test_argv = ['--iodir',
    #             '/scratch/samuelmiller/12-7-17',
    #             '--filename',
    #             'toolik_8_2_2_1',
    #             #'--denovogui_fp',
    #             #'/home/samuelmiller/DeNovoGUI/DeNovoGUI-1.15.10/DeNovoGUI-1.15.10.jar',
    #             #'--mgf_fp',
    #             #'/scratch/samuelmiller/12-7-17/toolik_8_2_2_1.mgf',
    #             '--cores',
    #             '32']

    #test_argv = ['--iodir',
    #             '/scratch/samuelmiller/11-11-17/082917_toolik_core_9_2_1_1_sem_results',
    #             '--filename',
    #             '082917_toolik_core_9_2_1_1_sem',
    #             '--db_name_list',
    #             'ERR1017187',
    #             'ERR1019366',
    #             'ERR1022687',
    #             'ERR1022692',
    #             'ERR1034454',
    #             'ERR1035437',
    #             'ERR1035438',
    #             'ERR1035441',
    #             'ERR1039457',
    #             'ERR1039458',
    #             'SRR5208451.transcript',
    #             'SRR5208454.transcript',
    #             'SRR5208455.transcript',
    #             'SRR5208541.transcript',
    #             'SRR5208544.transcript',
    #             'SRR5208545.transcript',
    #             'SRR5450431',
    #             'SRR5450432',
    #             'SRR5450434',
    #             'SRR5450438',
    #             'SRR5450631',
    #             'SRR5450755',
    #             'SRR5471030',
    #             'SRR5471031',
    #             'SRR5471032',
    #             'SRR5471221',
    #             'SRR5476649',
    #             'SRR5476651',
    #             '--cores',
    #             '32']

    #test_argv = ['--iodir',
    #             '/scratch/samuelmiller/11-11-17/082917_toolik_core_10_2_1_1_sem_results',
    #             '--filename',
    #             '082917_toolik_core_10_2_1_1_sem',
    #             '--db_name_list',
    #             'ERR1017187',
    #             'ERR1019366',
    #             'ERR1022687',
    #             'ERR1022692',
    #             'ERR1034454',
    #             'ERR1035437',
    #             'ERR1035438',
    #             'ERR1035441',
    #             'ERR1039457',
    #             'ERR1039458',
    #             'mgm4477874.3',
    #             'SRR825158',
    #             'SRR825188',
    #             'SRR5450431.Graph2Pro',
    #             'SRR5450432.Graph2Pro',
    #             'SRR5450434.Graph2Pro',
    #             'SRR5450438.Graph2Pro',
    #             'SRR5208451.transcript',
    #             'SRR5208454.transcript',
    #             'SRR5208455.transcript',
    #             'SRR5208541.transcript',
    #             'SRR5208544.transcript',
    #             'SRR5208545.transcript',
    #             'SRR5450631',
    #             'SRR5450755',
    #             'SRR5471030',
    #             'SRR5471031',
    #             'SRR5471032',
    #             'SRR5471221',
    #             'SRR5476649',
    #             'SRR5476651',
    #             '--cores',
    #             '32']

    #test_argv = ['--denovogui_fp',
    #             'C:\\Program Files (x86)\\DeNovoGUI-1.4.12-windows\\DeNovoGUI-1.15.11\\DeNovoGUI-1.15.11.jar',
    #             '--mgf_fp',
    #             'C:\\Users\\Samuel\\Downloads\\082917_toolik_core_16_2_2_1_sem.mgf',
    #             '--psm_fp_list',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1017187.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1019366.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1022687.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1022692.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1034454.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1035437.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1035438.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1035441.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1039457.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.ERR1039458.DBGraphPep2Pro.fixedKR.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.mgm4477874.3.fastq.fgs.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.SRR825158.fastq.fgs.0.01.tsv',
    #             '082917_toolik_core_10_2_1_1_sem.SRR825188.fastq.fgs.0.01.tsv',
    #             '--db_name_list',
    #             'ERR1017187.Graph2Pro',
    #             'ERR1019366.Graph2Pro',
    #             'ERR1022687.Graph2Pro',
    #             'ERR1022692.Graph2Pro',
    #             'ERR1034454.Graph2Pro',
    #             'ERR1035437.Graph2Pro',
    #             'ERR1035438.Graph2Pro',
    #             'ERR1035441.Graph2Pro',
    #             'ERR1039457.Graph2Pro',
    #             'ERR1039458.Graph2Pro',
    #             'mgm4477874.3.fgs',
    #             'SRR825158.fgs',
    #             'SRR825188.fgs',
    #             '--cores',
    #             '3']

    #test_argv = ['--filename', '042017_toolik_core_2_2_1_1_sem', '--iodir', 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\8-16-17',
    #             '--psm_fp_list', '042017_toolik_core_2_2_1_1_sem.ERR1017187.DBGraphPep2Pro.fixedKR.0.01.tsv', '042017_toolik_core_2_2_1_1_sem.mgm4477874.3.fastq.fgs.tsv', '042017_toolik_core_2_2_1_1_sem.SRR825158.fastq.fgs.tsv', '042017_toolik_core_2_2_1_1_sem.SRR825188.fastq.fgs.tsv',
    #             '--db_name_list', 'ERR1017187.Graph2Pro', 'mgm4477874.3.fgs', 'SRR825158.fgs', 'SRR825188.fgs',
    #             '--cores', '4']
    if 'test_argv' in locals():
        args = userargs.setup(test_argv)
    else:
        args = userargs.setup()

    input_df_dict = input.load_files()
    utils.save_pkl_objects(config.iodir[0], **{'input_df_dict': input_df_dict})
    #utils.save_dfs(config.iodir[0], 
    #               **{alg + '.' + config.filename[0] + '.' + tol: input_df_dict[alg][tol]
    #                  for alg in config.alg_list for tol in config.frag_mass_tols})
    #input_df_dict = utils.load_pkl_objects(config.iodir[0], 'input_df_dict')

    prediction_df = consensus.make_prediction_df(input_df_dict)
    utils.save_pkl_objects(config.iodir[0], **{'consensus_prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'consensus_prediction_df')

    prediction_df = masstol.update_prediction_df(prediction_df)
    utils.save_pkl_objects(config.iodir[0], **{'mass_tol_prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'mass_tol_prediction_df')

    prediction_df = interspec.update_prediction_df(prediction_df)
    utils.save_pkl_objects(config.iodir[0], **{'interspec_prediction_df': prediction_df})
    #prediction_df = utils.load_pkl_objects(config.iodir[0], 'interspec_prediction_df')

    classifier.classify(prediction_df = prediction_df)
    # COMMENT
    #classifier.classify()

    utils.verbose_print('total time elapsed:', time() - start_time)

if __name__ == '__main__':
    main()