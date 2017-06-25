import argparse
import os
import random
import multiprocessing
import time
import sys
import subprocess
import pandas as pd
import math
import matplotlib
# Set backend to make image files on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import subprocess

#from attrdict import AttrDict
from collections import OrderedDict
from Bio import SeqIO
from attrdict import AttrDict

blast_table_headers = ['query_id',
                       'subject_accession',
                       'query_start',
                       'query_end',
                       'subject_start',
                       'subject_end',
                       'e_value',
                       'bit_score',
                       'percent_identity',
                       'gaps',
                       'taxon_id']

def main():

    #args = get_args()
    #test_args(args)
    # TEST CODE: REPLACE CMD LINE ARGS WITH ANALOGOUS ATTRDICT
    args = AttrDict()
    args.out_dir = 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\test2'
    #args.out_dir = '/home/samuelmiller/FragGeneScanPlus/tmp'
    args.blast_batch_path = '/home/samuelmiller/FragGeneScanPlus/tmp/blaststats_blast_batch.sh'
    args.cores = 24
    #args.max_seqs_per_process = 1
    args.max_seqs_per_process = 2000
    args.max_nonident = 4
    args.translated_seqs_path = '/home/samuelmiller/FragGeneScanPlus/tmp/ERR1019366_1.trimmed.fgsp.pep_list.faa'
    #args.min_peptide_length = 10
    args.min_peptide_length = 7
    #args.max_peptide_length = 11
    args.max_peptide_length = 17
    args.peptide_length_interval = 1
    #args.peptide_sample_size = 8
    args.peptide_sample_size = 5000
    args.dna_reads_path = None
    args.min_dna_length = None
    args.max_dna_length = None
    args.dna_length_interval = None
    args.dna_sample_size = None
    args.blastn_path = None
    args.nt_db_dir = None
    args.blastp_path = '/home/samuelmiller/ncbi-blast-2.6.0+/bin/blastp'
    args.aa_db_dir = '/home/samuelmiller/refseq_protein/refseq_protein'

    if args.dna_sample_size is not None:
        raise NotImplementedError('DNA searches are not implemented.')
    if args.peptide_sample_size is not None:
        #aa_seq_filepath = peptide_setup(args)
        #seq_sample_dict = draw_seqs('aa', args, aa_seq_filepath)
        #split_fasta_pathname_list = make_fasta_files('aa', args, seq_sample_dict)
        #blast_seqs('aa', args, split_fasta_pathname_list)
        split_fasta_pathname_list = ['C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\test2\\peptide_sample_' + str(i) + '.faa'
                                     for i in range(1, 29)]
        analyze_blast_output('aa', args, split_fasta_pathname_list)

    #seq_sample_dict = draw_seqs(args)
    #split_fasta_pathname_list = make_fasta_files(args, seq_sample_dict)
    #blast_seqs(args, split_fasta_pathname_list)
    #analyze_blast_output(args, split_fasta_pathname_list)

    #args.out_dir = 'C:\\Users\\Samuel\\Documents\\blast_contig_seqs'
    #split_fasta_pathname_list = ['C:\\Users\\Samuel\\Documents\\blast_contig_seqs\\contig_sample_' + str(i) for i in range(1, 31)]
    #analyze_blast_output(args, split_fasta_pathname_list)

def test_args(args):

    if args.min_peptide_length is not None:
        if (args.max_peptide_length - args.min_peptide_length) % args.peptide_length_interval > 0:
            raise ValueError('max peptide length must be a multiple of peptide length interval greater than min peptide length')

    if args.min_dna_length is not None:
        if (args.max_dna_length - args.min_dna_length) % args.dna_length_interval > 0:
            raise ValueError('max dna length must be a multiple of dna length interval greater than min dna length')

    if isinstance(args.max_nonindent, int):
        if args.max_nonindent > args.min_peptide_length:
            raise ValueError('max number of non-identical residues in alignment must be <= minimum query seq length')
    else:
        raise ValueError('max number of non-identical residues in alignment must be a whole number')

def peptide_setup(args):
    
    aa_seq_filepath = os.path.join(
        args.out_dir,
        os.path.splitext(os.path.basename(args.translated_seqs_path))[0] + '.sample.txt'
        )

    # Discard the fasta header lines
    # Preserve lines at least the maximum sample length
    cmd = (
        'sed \'2~2n; d\' {0} | ' +\
        'grep \'^.\{{{1}\}}\' ' +\
        '> {2}'
        ).format(args.translated_seqs_path,
                args.max_peptide_length,
                aa_seq_filepath)
    subprocess.call(cmd, shell = True)

    return aa_seq_filepath

def analyze_blast_output(seq_type, args, split_fasta_pathname_list):

    if seq_type == 'nt':
        min_length = args.min_dna_length
        max_length = args.max_dna_length
        sample_size = args.dna_sample_size
    elif seq_type == 'aa':
        min_length = args.min_peptide_length
        max_length = args.max_peptide_length
        sample_size = args.peptide_sample_size
    nonident_bins = [i for i in range(args.max_nonident)]

    length_match_df = pd.DataFrame(columns=['query_length', 'nonident'])
    for fasta_pathname in split_fasta_pathname_list:
        blast_table_pathname = os.path.splitext(
            os.path.basename(fasta_pathname)
            )[0] + '.out'
        blast_df = pd.read_csv(os.path.join(args.out_dir, blast_table_pathname), sep='\t', header=None)
        blast_df.columns = blast_table_headers

        # SAMUEL: Remove when done testing
        blast_df = blast_df.groupby('query_id').first().reset_index()

        blast_df['query_length'] = blast_df['query_id']\
            .str.split('_').str[0].str.split('L').str[1]
        blast_df['query_length'] = pd.to_numeric(blast_df['query_length'])
        # deletions occur where query end-start < subject end-start
        blast_df['insertions'] = blast_df['query_end'] - blast_df['query_start']\
            - blast_df['subject_end'] + blast_df['subject_start']
        blast_df['insertions'] = blast_df['insertions'].apply(lambda x: x if x > 0 else 0)
        blast_df['deletions'] = blast_df['subject_end'] - blast_df['subject_start']\
            - blast_df['query_end'] + blast_df['query_start']
        blast_df['deletions'] = blast_df['deletions'].apply(lambda x: x if x > 0 else 0)
        blast_df['nonident'] =\
            round(
                blast_df['query_length']\
                    - (blast_df['query_end'] - blast_df['query_start'] + 1 + blast_df['deletions'])\
                    * (blast_df['percent_identity'] / 100),
                0)

        nonident_test_series = blast_df['nonident'].apply(
            lambda x: x - int(x)
            )
        if (nonident_test_series != 0).any():
            raise ArithmeticError('Nonident calculation is faulty')

        length_match_df = pd.concat(
            [length_match_df,
            pd.concat(
                [blast_df['query_length'], blast_df['nonident']],
                axis=1)],
            axis=0)

    length_match_df.set_index('query_length', inplace=True)
    length_groups = length_match_df.groupby(length_match_df.index)
    # Record the proportion of hits with N non-identical residues at each length
    length_nonident_dict = OrderedDict(
        [(l, []) for l in range(min_length, max_length + 1)]
        )
    # Loop through each peptide length
    for length in range(min_length, max_length + 1):
        # Count the number of hits with N non-identical residues
        nonident_counts = length_groups.get_group(length)['nonident'].value_counts()
        # Loop through the non-identical residue range specified by user (0 to max_nonident)
        for nonident in nonident_bins:
            # The sample of peptides may not have N non-identical residues
            if nonident in nonident_counts.index:
                # Record the proportion of hits with N non-identical residues
                length_nonident_dict[length].append(
                    nonident_counts[nonident] / sample_size)
            else:
                length_nonident_dict[length].append(0)
        # If the max number of non-identical residues < max seq length
        # add another bar for the proportion of hits with non-identical residues > max_nonident
        if args.max_nonident < max_length:
            length_nonident_dict[length].append(
                (1 - sum(length_nonident_dict[length])))
    plot_nonident(seq_type, args, length_nonident_dict, nonident_bins)

def plot_nonident(seq_type, args, length_nonident_dict, nonident_bins):

    if seq_type == 'nt':
        min_length = args.min_dna_length
        max_length = args.max_dna_length
        results_type = 'dna'
        title_type = 'DNA'
        db_name = os.path.basename(args.nt_db_dir)
    elif seq_type == 'aa':
        min_length = args.min_peptide_length
        max_length = args.max_peptide_length
        results_type = 'peptide'
        title_type = 'Peptide'
        db_name = os.path.basename(args.aa_db_dir)

    fig, ax = plt.subplots()
    ax.set_title(
        title_type + ' sequence hits to ' + db_name + ':\n'
        + 'Proportion of query seqs with N residues non-identical to the subject\n'
        + 'for different query seq lengths',
        horizontalalignment='center',
        multialignment='center',
        fontsize=12
        )

    x_positions = list(range(len(length_nonident_dict)))
    bar_gap = 0.2
    bar_width = 1 / (len(length_nonident_dict[min_length]))
    
    for i, nonident_bin in enumerate(range(len(length_nonident_dict[min_length]))):
        x_values = [position + bar_width * (i + 0.5) + bar_gap * j for (j, position) in enumerate(x_positions)]
        y_values = [length_nonident_dict[length][nonident_bin]
                    for length in length_nonident_dict]
        ax.bar(x_values,
               y_values,
               bar_width,
               alpha=0.6,
               color=cm.jet(i / len(length_nonident_dict[min_length]))
               )

    ax.set_xticks(
        [position * (1 + bar_gap) + len(length_nonident_dict[min_length]) / 2 * bar_width
         for position in x_positions]
        )
    ax.set_xticklabels(length_nonident_dict.keys())
    ax.set_xlabel('Query seq length')
    ax.set_xlim(-bar_gap, len(x_positions) * (1 + bar_gap))
    ax.set_ylabel('Proportion of query seqs with N non-identical residues')
    ax.set_ylim(0, 1)
    if args.max_nonident < max_length:
        ax.legend(
            [str(nonident) for nonident in nonident_bins + ['> ' + str(args.max_nonident)]],
             loc = 'upper right'
             )
    else:
        ax.legend(
            [str(nonident) for nonident in nonident_bins],
            loc = 'upper right'
            )
    ax.grid()
    fig.tight_layout()
    save_path = os.path.join(args.out_dir, results_type + '_results.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')
    plt.close()

# Convert fastq to fastn format
# Remove header lines
# Iterate through file
# If line length is greater than the maximum subsequence length under consideration,
# randomly choose one of the subseq lengths (e.g., 27, 30, 33) to sample
# Sample the subseq if the quota for that length has not been met

def make_fasta_files(seq_type, args, seq_sample_dict):

    # Make a single fasta list
    # Headers have the format: L<length>_<N>
    # Split the fasta list for multithreading
    # BLAST with only 1 hit retained per query
    # Merge into dict for each length
    # Compute proportion with exact matches

    if seq_type == 'nt':
        fasta_basename = 'dna_sample'
    elif seq_type == 'aa':
        fasta_basename = 'peptide_sample'

    # Place all sampled seqs into single fasta list
    fasta_list = []
    for length in seq_sample_dict:
        seqs = seq_sample_dict[length]
        str_length = str(length)
        for i, seq in enumerate(seqs):
            fasta_list.append('>L' + str_length + '_' + str(i) + '\n')
            fasta_list.append(seq + '\n')
    
    cores = args.cores
    max_seqs_per_process = args.max_seqs_per_process
    out_dir = args.out_dir
    parent_fasta_size = len(fasta_list) / 2
    child_fasta_size = int(parent_fasta_size / cores)
    remainder = parent_fasta_size % cores
    split_fasta_pathname_list = []

    # When there are a small number of seqs,
    # divide into a number of files equal to the number of cores
    if child_fasta_size + remainder < max_seqs_per_process:
        for core in range(cores):
            child_fasta_list = fasta_list[core * child_fasta_size * 2: (core + 1) * child_fasta_size * 2]
            child_fasta_path = os.path.join(out_dir, fasta_basename + '_' + str(core + 1) + '.faa')
            with open(child_fasta_path, 'w') as child_fasta_file:
                for line in child_fasta_list:
                    child_fasta_file.write(line)
            split_fasta_pathname_list.append(child_fasta_path)
        # Tack on the remainder to the last fasta file
        with open(child_fasta_path, 'a') as child_fasta_file:
            child_fasta_list = fasta_list[cores * child_fasta_size * 2:]
            for line in child_fasta_list:
                child_fasta_file.write(line)
    # If there is a large number of seqs,
    # cap the number of seqs in each file
    else:
        fasta_line = 0
        child_fasta_count = 1
        while fasta_line < len(fasta_list):
            child_fasta_list = fasta_list[fasta_line: fasta_line + max_seqs_per_process * 2]
            child_fasta_filename = os.path.join(out_dir, fasta_basename + '_' + str(child_fasta_count) + '.faa')
            with open(child_fasta_filename, 'w') as child_fasta_file:
                for line in child_fasta_list:
                    child_fasta_file.write(line)
            split_fasta_pathname_list.append(child_fasta_filename)
            fasta_line += max_seqs_per_process * 2
            child_fasta_count += 1

    return split_fasta_pathname_list
    
def blast_seqs(seq_type, args, fasta_pathname_list):

    if seq_type == 'nt':
        blast_process = 'blastn'
        blast_path = args.blastn_path
        db_dir = args.nt_db_dir
    if seq_type == 'aa':
        blast_process = 'blastp'
        blast_path = args.blastp_path
        db_dir = args.aa_db_dir

    with open(args.blast_batch_path, 'r') as blast_batch_template_file:
        blast_batch_template = blast_batch_template_file.read()
    temp_blast_batch_script = blast_batch_template
    temp_blast_batch_script = temp_blast_batch_script.replace('FASTA_FILES=', 'FASTA_FILES=({})'.format(' '.join(fasta_pathname_list)))
    temp_blast_batch_script = temp_blast_batch_script.replace('MAX_PROCESSES=', 'MAX_PROCESSES={}'.format(args.cores - 1))
    temp_blast_batch_script = temp_blast_batch_script.replace('BLAST_PROCESS=', 'BLAST_PROCESS={}'.format(blast_process))
    temp_blast_batch_script = temp_blast_batch_script.replace('BLAST_PATH=', 'BLAST_PATH={}'.format(blast_path))
    temp_blast_batch_script = temp_blast_batch_script.replace('DB_DIR=', 'DB_DIR={}'.format(db_dir))
    temp_blast_batch_path = os.path.join(os.path.dirname(args.blast_batch_path), 'blaststats_blast_batch~.sh')
    with open(temp_blast_batch_path, 'w') as temp_blast_batch_file:
        temp_blast_batch_file.write(temp_blast_batch_script)
    os.chmod(temp_blast_batch_path, 0o777)
    subprocess.call([temp_blast_batch_path])

def get_args():

    parser = argparse.ArgumentParser(
        description='Draw sequences covering range of lengths from translated metagenomic contigs \
        and BLAST to determine minimum length for unique match to database.'
        )

    parser.add_argument(
        '--out_dir',
        default='/home/samuelmiller/arctic_metagenomes/',
        help='output directory')

    blast_group = parser.add_argument_group('blast_group')
    blast_group.add_argument(
        '--blast_batch_path',
        default='/home/samuelmiller/arctic_metagenomes/blaststats_blast_batch.sh',
        help='blaststats_blast_batch.sh path')
    blast_group.add_argument(
        '--cores',
        default=16,
        type=int,
        help='number of cores to use for BLAST+: \
        {} are available'.format(multiprocessing.cpu_count()))
    blast_group.add_argument(
        '--max_seqs_per_process',
        default=2000,
        type=int,
        help='maximum number of query seqs per BLAST+ instance')

    analysis_group.add_argument(
        '--max_nonident',
        default=4,
        type=int,
        help='maximum number of nonidentical residues that should be considered in alignments')

    aa_group = parser.add_argument_group('aa_group')
    #aa_group.add_argument(
    #    '--translated_seqs_path',
    #    default='/home/samuelmiller/arctic_metagenomes/megahit_out/intermediate_contigs/k87.contig-pep-seqs.txt',
    #    help='translated metagenomic seqs filepath: no headers allowed\n\
    #    produce from fasta file with command ' + r'`sed \'2~2n; d\' infile > outfile`')
    aa_group.add_argument(
        '--translated_seqs_path',
        default='/home/samuelmiller/arctic_metagenomes/ERR1019366_1.k87',
        help='translated metagenomic sequences filepath')
    aa_group.add_argument(
        '--min_peptide_length',
        default=7,
        type=int,
        help='minimum sequence length to consider')
    aa_group.add_argument(
        '--max_peptide_length',
        default=17,
        type=int,
        help='maximum sequence length to consider')
    aa_group.add_argument(
        '--peptide_length_interval',
        default=1,
        type=int,
        help='sampling interval between min and max lengths (e.g., 1 = every length between min and max)\n\
        guaranteed to sample min, but not necessarily max if max does not align with sampling interval')
    aa_group.add_argument(
        '--peptide_sample_size',
        default=10000,
        type=int,
        help='number of sequences to draw at each length')

    nt_group = parser.add_argument_group('nt_group')
    nt_group.add_argument(
        '--dna_reads_path',
        default=None,
        help='translated metagenomic contigs filepath: no headers allowed\n\
        produce with command ' + r'`sed \'1d; n; d\' infile > outfile`')
    nt_group.add_argument(
        '--min_dna_length',
        default=None,
        type=int,
        help='minimum sequence length to consider')
    nt_group.add_argument(
        '--max_dna_length',
        default=None,
        type=int,
        help='maximum sequence length to consider')
    nt_group.add_argument(
        '--dna_length_interval',
        default=3,
        type=int,
        help='sampling interval between min and max lengths (e.g., 3 = every 3rd length between min and max)\n\
        guaranteed to sample min, but not necessarily max if max does not align with sampling interval')
    nt_group.add_argument(
        '--dna_sample_size',
        default=None,
        type=int,
        help='number of sequences to draw at each length')

    blastn_group = parser.add_argument_group('blastn')
    blastn_group.add_argument(
        '--blastn_path',
        default=None,
        help='blastn filepath')
    blastn_group.add_argument(
        '--nt_db_dir',
        default=None,
        help='BLAST+ nt database directory path')

    blastp_group = parser.add_argument_group('blastp')
    blastp_group.add_argument(
        '--blastp_path',
        default='/home/samuelmiller/ncbi-blast-2.6.0+/bin/blastp',
        help='blastp filepath')
    blastp_group.add_argument(
        '--aa_db_dir',
        default='/home/samuelmiller/refseq_protein/refseq_protein',
        help='BLAST+ aa database directory path')

    return parser.parse_args()

def draw_seqs(seq_type, args, seq_filepath):

    if seq_type == 'aa':
        min_length = args.min_peptide_length
        max_length = args.max_peptide_length
        interval = args.peptide_length_interval
        sample_size = args.peptide_sample_size

    # Count the number of lines in the file
    line_count = 0
    with open(seq_filepath) as f:
        for line in f:
            line_count += 1

    ## For each seq length, find the first line of seqs of the length
    #length_blocks = OrderedDict()
    #with open(seq_filepath) as f:
    #    current_length = first_line_length = len(f.readline())
    #    length_blocks[first_line_length - 1] = 0
    #    for i, line in enumerate(f):
    #        if len(line) > current_length:
    #            current_length = len(line)
    #            if current_length - 1 > max_length:
    #                break
    #            length_blocks[current_length - 1] = i

    ## Randomly select the lines from which seqs will be sampled
    #sample_line_dict = OrderedDict()
    #for subseq_length in range(min_length, max_length + 1):
    #    if subseq_length < first_line_length - 1:
    #        sample_lower_line_bound = 0
    #    else:
    #        sample_lower_line_bound = length_blocks[subseq_length]
    #    sample_line_dict[subseq_length] = sorted(
    #        random.sample(
    #            range(sample_lower_line_bound, line_count + 1),
    #            sample_size
    #            )
    #        )

    # Randomly select the lines from which seqs will be sampled
    sample_line_dict = OrderedDict()
    for subseq_length in range(min_length, max_length + 1):
        sample_line_dict[subseq_length] = sorted(
            random.sample(
                range(line_count + 1),
                sample_size
                )
            )

    ## Subsample seqs from the chosen lines
    #seq_sample_dict = OrderedDict(
    #    [(subseq_length, []) for subseq_length in sample_line_dict]
    #    )
    #if min_length < first_line_length - 1:
    #    first_line_of_interest = 0
    #else:
    #    first_line_of_interest = length_blocks[min_length]
    #with open(seq_filepath) as f:
    #    for i, line in enumerate(f):
    #        if i >= first_line_of_interest:
    #            for subseq_length in range(min_length, max_length + 1):
    #                try:
    #                    if i == sample_line_dict[subseq_length][0]:
    #                        parent_seq = line.strip('\n')
    #                        subseq_start = random.randint(0, len(parent_seq) - subseq_length)
    #                        subseq = parent_seq[subseq_start: subseq_start + subseq_length]
    #                        seq_sample_dict[subseq_length].append(subseq)
    #                        del(sample_line_dict[subseq_length][0])
    #                except IndexError:
    #                    pass

    # Subsample seqs from the chosen lines
    seq_sample_dict = OrderedDict(
        [(subseq_length, []) for subseq_length in sample_line_dict]
        )
    with open(seq_filepath) as f:
        for i, line in enumerate(f):
            for subseq_length in range(min_length, max_length + 1):
                try:
                    if i == sample_line_dict[subseq_length][0]:
                        parent_seq = line.strip('\n')
                        subseq_start = random.randint(0, len(parent_seq) - subseq_length)
                        subseq = parent_seq[subseq_start: subseq_start + subseq_length]
                        seq_sample_dict[subseq_length].append(subseq)
                        del(sample_line_dict[subseq_length][0])
                except IndexError:
                    pass

    return seq_sample_dict


if __name__ == '__main__':
    main()