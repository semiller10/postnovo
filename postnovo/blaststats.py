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

#from attrdict import AttrDict
from collections import OrderedDict
from Bio import SeqIO

# SAMUEL: remove subject_gi after testing
blast_table_headers = ['query_id',
                       'subject_gi',
                       'subject_accession',
                       'query_start',
                       'query_end',
                       'subject_start',
                       'subject_end',
                       'e_value',
                       'bit_score',
                       'percent_identity',
                       'taxon_id'
                       ]
proportion_bins = [i / 10 for i in range(11)]

def main():

    args = get_args()
    #args = AttrDict()
    #args.out_dir = '/home/samuelmiller/arctic_metagenomes/'
    #args.contig_path = '/home/samuelmiller/arctic_metagenomes/megahit_out/intermediate_contigs/k87.contig-pep.txt'
    #args.min_length = 7
    #args.max_length = 12
    #args.sample_size = 10
    #args.blast_batch_path = '/home/samuelmiller/arctic_metagenomes/blast_batch.sh'
    #args.blastp_path = '/home/samuelmiller/ncbi-blast-2.6.0+/bin/blastp'
    #args.db_dir = '/home/samuelmiller/refseq_protein/refseq_protein'
    #args.cores = 16
    #args.max_seqs_per_process = 1000

    #seq_sample_dict = draw_seqs(args)
    #split_fasta_pathname_list = make_fasta_files(args, seq_sample_dict)
    #blast_seqs(args, split_fasta_pathname_list)
    #analyze_blast_output(args, split_fasta_pathname_list)

    args.out_dir = 'C:\\Users\\Samuel\\Documents\\blast_contig_seqs'
    split_fasta_pathname_list = ['C:\\Users\\Samuel\\Documents\\blast_contig_seqs\\contig_sample_' + str(i) for i in range(1, 31)]
    analyze_blast_output(args, split_fasta_pathname_list)

def analyze_blast_output(args, split_fasta_pathname_list):

    length_match_df = pd.DataFrame(columns=['query_length', 'query_match_proportion'])
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
        blast_df['query_match_proportion'] =\
            blast_df['percent_identity'] *\
            (blast_df['query_end'] - blast_df['query_start'] + 1) / blast_df['query_length']
        blast_df['query_match_proportion'] = blast_df['query_match_proportion'].apply(
            lambda x: math.floor(x / 10) / 10
            )
        length_match_df = pd.concat(
            [length_match_df,
            pd.concat([blast_df['query_length'], blast_df['query_match_proportion']], axis=1)],
            axis=0)
    length_match_df.set_index('query_length', inplace=True)
    length_groups = length_match_df.groupby(length_match_df.index)
    length_match_proportion_dict = OrderedDict(
        [(l, []) for l in range(args.min_length, args.max_length + 1)]
        )
    for length in range(args.min_length, args.max_length + 1):
        proportion_counts = length_groups.get_group(length)['query_match_proportion'].value_counts()
        for p in proportion_bins:
            if p in proportion_counts.index:
                length_match_proportion_dict[length].append(
                    proportion_counts[p] / args.sample_size
                )
            else:
                length_match_proportion_dict[length].append(0)
    plot_match_proportion(args, length_match_proportion_dict)

def plot_match_proportion(args, length_match_proportion_dict):

    # SAMUEL: Modify to only plot those proportions represented in the dict

    fig, ax = plt.subplots()
    plt.title(
        'Peptide sequence hits to RefSeq:\n\
        Proportion of query sequence residues matching the subject sequence\n\
        for different query sequence lengths'
        )

    x_positions = list(range(len(length_match_proportion_dict)))
    bar_gap = 0.2
    bar_width = 1 / len(proportion_bins)
    max_height = 0
    for i, length in enumerate(length_match_proportion_dict):
        plt.bar(x_positions,
                length_match_proportion_dict[length],
                bar_width,
                alpha=0.5,
                color=cm.jet(i / len(x_positions)),
                label=length)
        if max(length_match_proportion_dict[length]) > max_height:
            max_height = max(length_match_proportion_dict[length])
    ax.set_xticks([position * (1 + bar_gap) + len(proportion_bins) / 2 * bar_width
                   for position in x_positions])
    ax.set_xticklabels(proportion_bins)
    plt.xlim(0, (len(x_positions) + 1) * (1 + bar_gap))
    plt.ylim(0, max_height)
    plt.legend(['Length ' + str(length) for length in length_match_proportion_dict],
               loc = 'upper right')
    plt.grid()
    plt.tight_layout()
    save_path = join(args.out_dir + 'blastp_results.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')
    plt.close()

# Convert fastq to fastn format
# Remove header lines
# Iterate through file
# If line length is greater than the maximum subsequence length under consideration,
# randomly choose one of the subseq lengths (e.g., 27, 30, 33) to sample
# Sample the subseq if the quota for that length has not been met

def make_fasta_files(args, seq_sample_dict):

    # Make a single fasta list
    # Headers have the format: L<length>_<N>
    # Split the fasta list for multithreading
    # BLAST with only 1 hit retained per query
    # Merge into dict for each length
    # Compute proportion with exact matches

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
    fasta_basename = 'contig_sample'
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
    # If there are a large number of seqs,
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
    
def blast_seqs(args, fasta_pathname_list):

    with open(args.blast_batch_path, 'r') as blast_batch_template_file:
        blast_batch_template = blast_batch_template_file.read()
    temp_blast_batch_script = blast_batch_template
    temp_blast_batch_script = temp_blast_batch_script.replace('FASTA_FILES=', 'FASTA_FILES=({})'.format(' '.join(fasta_pathname_list)))
    temp_blast_batch_script = temp_blast_batch_script.replace('MAX_PROCESSES=', 'MAX_PROCESSES={}'.format(args.cores - 1))
    temp_blast_batch_script = temp_blast_batch_script.replace('BLASTP_PATH=', 'BLASTP_PATH={}'.format(args.blastp_path))
    temp_blast_batch_script = temp_blast_batch_script.replace('DB_DIR=', 'DB_DIR={}'.format(args.db_dir))
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

    parser.add_argument('-o', '--out_dir',
                        default='/home/samuelmiller/arctic_metagenomes/',
                        help='output directory')

    metagenome_group = parser.add_argument_group('metagenome')
    metagenome_group.add_argument('-c', '--contig_path',
                                  default='/home/samuelmiller/arctic_metagenomes/megahit_out/intermediate_contigs/k87.contig-pep-seqs.txt',
                                  help='translated metagenomic contigs filepath: no headers allowed\n\
                                  produce with command ' + r'`sed \'1d; n; d\' infile > outfile`')
    metagenome_group.add_argument('-i', '--min_length', default=7, type=int,
                                  help='minimum sequence length to consider')
    metagenome_group.add_argument('-x', '--max_length', default=12, type=int,
                                  help='maximum sequence length to consider')
    metagenome_group.add_argument('-z', '--sample_size', default=10000, type=int,
                                  help='number of sequences to draw at each length')

    blast_group = parser.add_argument_group('blast')
    blast_group.add_argument('-a', '--blast_batch_path',
                             default='/home/samuelmiller/arctic_metagenomes/blaststats_blast_batch.sh',
                             help='blaststats_blast_batch.sh path'
                             )
    blast_group.add_argument('-b', '--blastp_path',
                             default='/home/samuelmiller/ncbi-blast-2.6.0+/bin/blastp',
                             help='blastp filepath'
                             )
    blast_group.add_argument('-d', '--db_dir',
                             default='/home/samuelmiller/refseq_protein/refseq_protein',
                             help='BLAST+ database directory path')
    blast_group.add_argument('-r', '--cores', default=16, type=int,
                             help='number of cores to use for BLAST+: \
                             {} are available'.format(multiprocessing.cpu_count()))
    blast_group.add_argument('-p', '--max_seqs_per_process', default=2000, type=int,
                             help='maximum number of query seqs per BLAST+ instance')

    return parser.parse_args()

def draw_seqs(args):

    min_length = args.min_length
    max_length = args.max_length

    # Count the number of lines in the file
    line_count = 0
    with open(args.contig_path) as f:
        for line in f:
            line_count += 1

    # For each seq length, find the first line of seqs of the length
    length_blocks = OrderedDict()
    with open(args.contig_path) as f:
        current_length = first_line_length = len(f.readline())
        length_blocks[first_line_length - 1] = 0
        for i, line in enumerate(f):
            if len(line) > current_length:
                current_length = len(line)
                if current_length - 1 > max_length:
                    break
                length_blocks[current_length - 1] = i

    # Randomly select the lines from which seqs will be sampled
    sample_line_dict = OrderedDict()
    for subseq_length in range(min_length, max_length + 1):
        if subseq_length < first_line_length - 1:
            sample_lower_line_bound = 0
        else:
            sample_lower_line_bound = length_blocks[subseq_length]
        sample_line_dict[subseq_length] = sorted(
            random.sample(
                range(sample_lower_line_bound, line_count + 1),
                args.sample_size
                )
            )

    # Subsample seqs from the chosen lines
    seq_sample_dict = OrderedDict(
        [(subseq_length, []) for subseq_length in sample_line_dict]
        )
    if min_length < first_line_length - 1:
        first_line_of_interest = 0
    else:
        first_line_of_interest = length_blocks[min_length]
    with open(args.contig_path) as f:
        for i, line in enumerate(f):
            if i >= first_line_of_interest:
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