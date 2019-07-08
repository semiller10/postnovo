# from __future__ import print_function

import argparse
import numpy as np
import os.path
import pandas as pd
import random

def main():
    # test_argv = [
    #     'make_test_mgf',
    #     'C:\\Users\\Samuel\\Downloads\\Ecoli-AE_sample.mgf',
    #     'C:\\Users\\Samuel\\Downloads\\EcoliAE_sequest_percolator_fdr.txt'
    # ]
    if 'test_argv' in locals():
        args = parse_args(test_argv)
    else:
        args = parse_args()
        
    if args.command == 'make_test_mgf':
        mgf_list = make_mgf(args.input_mgf, args.accuracy_file)
        # write test mgf file
        mgf = os.path.join(
            os.path.dirname(args.input_mgf),
            os.path.splitext(os.path.basename(args.input_mgf))[0] + '-full.test.mgf'
            )
        with open(mgf, 'w') as handle:
            for line in mgf_list:
                handle.write(line)
    elif args.command == 'make_train_files':
        mgf_list = make_mgf(args.input_mgf, args.accuracy_file)
        train_mgf_list, valid_mgf_list, test_mgf_list = split_mgf(mgf_list)
        target_list, dbseq_list = make_target_dbseq(test_mgf_list)
        # write files
        dir_basename = os.path.join(
            os.path.dirname(args.input_mgf),
            os.path.splitext(os.path.basename(args.input_mgf))[0]
            )
        train_mgf = dir_basename + '.train.mgf'
        valid_mgf = dir_basename +'.valid.mgf'
        test_mgf = dir_basename + '.test.mgf'
        target = dir_basename + '.target'
        dbseq = dir_basename + '.dbseq'
        lists_to_write = [train_mgf_list, valid_mgf_list, test_mgf_list, target_list, dbseq_list]
        files_to_write = [train_mgf, valid_mgf, test_mgf, target, dbseq]
        for i in range(len(files_to_write)):
            with open(files_to_write[i], 'w') as handle:
                for line in lists_to_write[i]:
                    handle.write(line)
    elif args.command == 'merge_train_files':
        merge_train_files(args.dir, args.output_basename, args.input_basenames)

def parse_args(test_argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    make_test_mgf_subparser = subparsers.add_parser('make_test_mgf')
    make_test_mgf_subparser.add_argument('input_mgf')
    make_test_mgf_subparser.add_argument('accuracy_file')

    train_subparser = subparsers.add_parser('make_train_files')
    train_subparser.add_argument('input_mgf')
    train_subparser.add_argument('accuracy_file')

    merge_train_files_subparser = subparsers.add_parser('merge_train_files')
    merge_train_files_subparser.add_argument('dir')
    merge_train_files_subparser.add_argument('output_basename')
    merge_train_files_subparser.add_argument('input_basenames', nargs=argparse.REMAINDER)

    if test_argv:
        args = parser.parse_args(test_argv)
    else:
        args = parser.parse_args()

    return args

def make_mgf(input_mgf, accuracy_file=None):

    mod_symbols = {
        'c': 'C(+57.02)',
        'm': 'M(+15.99)'
    }

    with open(input_mgf) as handle:
        input_mgf_list = handle.readlines()

    if accuracy_file:
        accuracy_file_df = pd.read_csv(
            accuracy_file, 
            sep='\t', 
            header=0, 
            dtype={'Annotated Sequence': str, 'First Scan': str, 'Percolator q-Value': np.float64}
            )
        # retain confident PSMs
        spectra_ids = accuracy_file_df[accuracy_file_df['Percolator q-Value'] <= 0.01]
        # when multiple high-confidence PSMs are assigned to a spectrum,
        # ignore the spectrum
        spectra_ids = spectra_ids[
            spectra_ids.groupby('First Scan')['Percolator q-Value'].transform('count') == 1
            ]
        spectra_ids.set_index('First Scan', inplace=True)
        seq_list = spectra_ids['Annotated Sequence'].tolist()

        new_seq_list = []
        for seq in seq_list:
            new_seq = seq
            for mod_aa in mod_symbols:
                if mod_aa in seq:
                    new_seq = new_seq.replace(mod_aa, mod_symbols[mod_aa])
            new_seq_list.append(new_seq)
        spectra_ids['Annotated Sequence'] = new_seq_list
    
    new_mgf_list = []
    new_mgf_block = []
    for line in input_mgf_list:
        line_text = line.rstrip()
        if 'MASS=' == line_text[:len('MASS=')]:
            pass
        elif 'BEGIN IONS' == line_text:
            new_mgf_block.append(line)
        elif 'TITLE=' == line_text[:len('TITLE=')]:
            new_mgf_block.append(line)
        elif 'PEPMASS=' == line_text[:len('PEPMASS=')]:
            # deepnovo considers precursor mass, but not intensity
            new_mgf_block.append(line_text[:line_text.index(' ')] + '\n')
        elif 'CHARGE=' == line_text[:len('CHARGE=')]:
            new_mgf_block.append(line)
        elif 'RTINSECONDS=' == line_text[:len('RTINSECONDS=')]:
            rt_line = line
        elif 'SCANS=' == line_text[:len('SCANS=')]:
            scan = line_text.replace('SCANS=', '')
            # if considering spectra with db search PSMs, not unknown spectra,
            # only retain spectra that were given a high-confidence seq assignment by db search
            if accuracy_file:
                try:
                    seq = spectra_ids.get_value(scan, 'Annotated Sequence')
                    seq_found = True
                except KeyError:
                    seq_found = False
            # if considering truly unknown spectra for de novo sequencing,
            # assign dud sequences to spectra to allow deepnovo test module to function
            else:
                seq = 'A'
                seq_found = True
            new_mgf_block.append(line)
            new_mgf_block.append(rt_line)
            if seq_found:
                new_mgf_block.append('SEQ=' + seq + '\n')
        elif 'END IONS' == line_text:
            new_mgf_block.append(line)
            if seq_found:
                new_mgf_list += new_mgf_block
            new_mgf_block = []
        elif '' == line_text:
            # blank line between spectra
            new_mgf_block.append(line)
        else:
            # fragmentation spectrum mass, intensity data
            new_mgf_block.append(line)
    # blank line at beginning of new_mgf_list is an artifact of this method's list construction        
    if new_mgf_list[0] == '\n':
        new_mgf_list = new_mgf_list[1:]

    return new_mgf_list

def split_mgf(mgf_list):

    # loop through the training spectra,
    # record the lines reading BEGIN IONS
    spectrum_starts = []
    for i, line in enumerate(mgf_list):
        if 'BEGIN IONS\n' == line:
            spectrum_starts.append(i)
    spectrum_count = len(spectrum_starts)

    valid_count = int(0.05 * spectrum_count)
    test_count = int(0.005 * spectrum_count)
    train_count = spectrum_count - valid_count - test_count
    sample = random.sample(spectrum_starts, valid_count + test_count)
    valid_sample = sample[:valid_count]
    test_sample = sample[valid_count:]

    new_mgf_lists = {'train': [], 'valid': [], 'test': []}
    for i, line in enumerate(mgf_list):
        if 'BEGIN IONS\n' == line:
            if i in valid_sample:
                new_mgf_list = new_mgf_lists['valid']
            elif i in test_sample:
                new_mgf_list = new_mgf_lists['test']
            else:
                new_mgf_list = new_mgf_lists['train']
        new_mgf_list.append(line)

    return new_mgf_lists['train'], new_mgf_lists['valid'], new_mgf_lists['test']

def make_target_dbseq(test_mgf_list):

    mod_symbols = {
        'C(+57.02)': 'Cmod',
        'M(+15.99)': 'Mmod'
    }

    target_list = []
    dbseq_list = ['scan' + '\t' + 'target_seq' + '\n']
    for line in test_mgf_list:
        if 'SCANS=' == line[:len('SCANS=')]:
            target_list.append(line)
            scan = line.rstrip()[len('SCANS='):]
        elif 'SEQ=' == line[:len('SEQ=')]:
            target_list.append(line)
            seq = line.rstrip()[len('SEQ='):]
            new_seq = seq
            for mod_symbol in mod_symbols:
                if mod_symbol in seq:
                    new_seq = new_seq.replace(mod_symbol, mod_symbols[mod_symbol])
            dbseq_list.append(scan + '\t' + new_seq + '\n')

    return target_list, dbseq_list

def merge_train_files(dir, output_basename, input_basenames):

    output_dir_basename = os.path.join(dir, output_basename)
    with open(output_dir_basename + '.train.mgf', 'w') as new_train_mgf, \
        open(output_dir_basename + '.valid.mgf', 'w') as new_valid_mgf, \
        open(output_dir_basename + '.test.mgf', 'w') as new_test_mgf, \
        open(output_dir_basename + '.target', 'w') as new_target:
        for i, input_basename in enumerate(input_basenames):
            fraction = str(i+1) + ':'
            input_dir_basename = os.path.join(dir, input_basename)
            with open(input_dir_basename + '.train.mgf') as handle:
                train_mgf_list = handle.readlines()
                for line in train_mgf_list:
                    if 'SCANS=' == line[:len('SCANS=')]:
                        new_train_mgf.write('SCANS=F' + fraction + line[len('SCANS:'):])
                    else:
                        new_train_mgf.write(line)
            with open(input_dir_basename + '.valid.mgf') as handle:
                valid_mgf_list = handle.readlines()
                for line in valid_mgf_list:
                    if 'SCANS=' == line[:len('SCANS=')]:
                        new_valid_mgf.write('SCANS=F' + fraction + line[len('SCANS:'):])
                    else:
                        new_valid_mgf.write(line)
            with open(input_dir_basename + '.test.mgf') as handle:
                test_mgf_list = handle.readlines()
                for line in test_mgf_list:
                    if 'SCANS=' == line[:len('SCANS=')]:
                        new_test_mgf.write('SCANS=F' + fraction + line[len('SCANS:'):])
                    else:
                        new_test_mgf.write(line)
            with open(input_dir_basename + '.target') as handle:
                target_list = handle.readlines()
                for line in target_list:
                    if 'SCANS=' == line[:len('SCANS=')]:
                        new_target.write('SCANS=F' + fraction + line[len('SCANS:'):])
                    else:
                        new_target.write(line)

if __name__ == '__main__':
    main()