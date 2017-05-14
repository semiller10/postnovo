import argparse
import os
import stat
import pkg_resources
import subprocess
import pandas as pd

## FOR DEBUGGING PURPOSES: REMOVE
blast_batch_pathname = '/home/samuelmiller/5-9-17/postnovo/blast/blast_batch.sh'

raw_blast_table_headers = ['qseqid', 'sgi', 'sacc', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'pident', 'staxids']
merged_blast_table_headers = ['scan', 'xle permutation', 'postnovo score'] + raw_blast_table_headers[1:]
seq_table_headers = ['scan', 'xle permutation', 'seq']

def split_fasta(fasta_path, cores, max_seqs_per_process):

    fasta_list = open(fasta_path, 'r').readlines()
    split_fasta_pathname_list = []
    fasta_filename = os.path.basename(fasta_path).strip('.faa')
    fasta_dirname = os.path.dirname(fasta_path)
    parent_fasta_size = len(fasta_list) / 2
    if parent_fasta_size % int(parent_fasta_size) > 0:
        raise ValueError('The fasta input must have an even number of lines.')
    child_fasta_size = int(parent_fasta_size / cores)
    remainder = parent_fasta_size % cores

    if child_fasta_size + remainder < max_seqs_per_process:
        for core in range(cores):
            child_fasta_list = fasta_list[core * child_fasta_size * 2: (core + 1) * child_fasta_size * 2]
            child_fasta_path = os.path.join(fasta_dirname, fasta_filename + '_' + str(core + 1) + '.faa')
            with open(child_fasta_path, 'w') as child_fasta_file:
                for line in child_fasta_list:
                    child_fasta_file.write(line)
            split_fasta_pathname_list.append(child_fasta_path)
        with open(child_fasta_path, 'a') as child_fasta_file:
            child_fasta_list = fasta_list[cores * child_fasta_size * 2:]
            for line in child_fasta_list:
                child_fasta_file.write(line)
    else:
        fasta_line = 0
        child_fasta_count = 1
        while fasta_line < len(fasta_list):
            child_fasta_list = fasta_list[fasta_line: fasta_line + max_seqs_per_process * 2]
            child_fasta_filename = os.path.join(fasta_dirname, fasta_filename + '_' + str(child_fasta_count) + '.faa')
            with open(child_fasta_filename, 'w') as child_fasta_file:
                for line in child_fasta_list:
                    child_fasta_file.write(line)
            split_fasta_pathname_list.append(child_fasta_filename)
            fasta_line += max_seqs_per_process * 2
            child_fasta_count += 1

    return split_fasta_pathname_list

def run_blast(fasta_pathname_list, blastp_path, db_dir, cores):
    
    #blast_batch_pathname = pkg_resources.resource_filename('postnovo', 'blast_batch.sh')

    with open(blast_batch_pathname, 'r') as blast_batch_template_file:
        blast_batch_template = blast_batch_template_file.read()
    temp_blast_batch_script = blast_batch_template
    temp_blast_batch_script = temp_blast_batch_script.replace('FASTA_FILES=', 'FASTA_FILES=({})'.format(' '.join(fasta_pathname_list)))
    temp_blast_batch_script = temp_blast_batch_script.replace('MAX_PROCESSES=', 'MAX_PROCESSES={}'.format(cores - 1))
    temp_blast_batch_script = temp_blast_batch_script.replace('BLASTP_PATH=', 'BLASTP_PATH={}'.format(blastp_path))
    temp_blast_batch_script = temp_blast_batch_script.replace('DB_DIR=', 'DB_DIR={}'.format(db_dir))
    temp_blast_batch_pathname = os.path.join(os.path.dirname(blast_batch_pathname), 'blast_batch~.sh')
    with open(temp_blast_batch_pathname, 'w') as temp_blast_batch_file:
        temp_blast_batch_file.write(temp_blast_batch_script)
    os.chmod(temp_blast_batch_pathname, 0o777)
    subprocess.call([temp_blast_batch_pathname])

def merge_blast_tables(fasta_path, split_fasta_pathname_list):

    merged_blast_table = pd.DataFrame(columns = merged_blast_table_headers)
    for fasta_pathname in split_fasta_pathname_list:
        out_pathname = os.path.join(os.path.dirname(fasta_pathname),
                                    os.path.basename(fasta_pathname).strip('.faa') + '.out')
        split_blast_table_raw = pd.read_table(out_pathname, names = raw_blast_table_headers)
        scan_col_plus_two_more = split_blast_table_raw['qseqid'].apply(lambda x: pd.Series(x.split(':')))
        # If considering sequences without Xle permutations
        if len(scan_col_plus_two_more.columns) == 1:
            scan_col_plus_two_more = pd.concat([scan_col_plus_two_more, pd.Series(index = scan_col_plus_two_more)], axis = 1)
        permut_col_score_col = scan_col_plus_two_more[1].apply(lambda x: pd.Series(x.split(';')))
        # If considering data without postnovo scores
        if len(permut_col_score_col.columns) == 1:
            permut_col_score_col = pd.concat([permut_col_score_col, pd.Series(index = permut_col_score_col)], axis = 1)
        split_blast_table = pd.concat([scan_col_plus_two_more[0], permut_col_score_col, split_blast_table_raw[split_blast_table_raw.columns[1:]]], axis = 1)
        split_blast_table.columns = merged_blast_table_headers
        merged_blast_table = pd.concat([merged_blast_table, split_blast_table], axis = 0)
    merged_blast_table.set_index(['scan', 'xle permutation'], inplace = True)

    seq_table = tabulate_fasta(fasta_path)
    merged_blast_table = seq_table.join(merged_blast_table)
    return merged_blast_table

def tabulate_fasta(fasta_path):

    raw_fasta_table = pd.read_table(fasta_path, header = None)
    fasta_headers = raw_fasta_table.ix[::2, 0]
    scan_col_plus_two_more = fasta_headers.apply(lambda x: pd.Series(x.split(':')))
    # If considering sequences without Xle permutations
    if len(scan_col_plus_two_more.columns) == 1:
        scan_col_plus_two_more = pd.concat([scan_col_plus_two_more, pd.Series(index = scan_col_plus_two_more)], axis = 1)
    scan_col_plus_two_more.index = range(len(scan_col_plus_two_more))
    scan_col_plus_two_more[0] = scan_col_plus_two_more[0].apply(lambda x: x.strip('>'))
    permut_col_score_col = scan_col_plus_two_more[1].apply(lambda x: pd.Series(x.split(';')))
    # If considering data without postnovo scores
    if len(permut_col_score_col.columns) == 1:
        permut_col_score_col = pd.concat([permut_col_score_col, pd.Series(index = permut_col_score_col)], axis = 1)
    permut_col_score_col.index = range(len(permut_col_score_col))
    seq_col = raw_fasta_table.ix[1::2, 0]
    seq_col.index = range(len(seq_col))
    seq_table = pd.concat([scan_col_plus_two_more[0], permut_col_score_col[0], seq_col], axis = 1)
    seq_table.columns = seq_table_headers
    seq_table.set_index(['scan', 'xle permutation'], inplace = True)
    seq_table['len'] = seq_table['seq'].apply(lambda seq: len(seq))

    return seq_table

def filter_blast_table(blast_table):

    blast_table['pseq'] = (blast_table['qend'] - blast_table['qstart'] + 1) / blast_table['len']
    filtered_blast_table = blast_table[blast_table['pseq'] == 1]
    filtered_blast_table = blast_table[blast_table['pident'] == 1]
    filtered_blast_table.index = filtered_blast_table.index.droplevel('xle permutation')
    return filtered_blast_table

if __name__ == '__main__':

    #parser = argparse.ArgumentParser(description = 'Set up BLAST+ search for de novo sequences')
    #parser.add_argument('--fasta_path',
    #                    help = 'fasta input filepath')
    #parser.add_argument('--blastp_path',
    #                    help = 'blastp filepath')
    #parser.add_argument('--db_dir',
    #                    help = 'directory containing BLAST database')
    #parser.add_argument('--cores', default = 1, type = int,
    #                    help = 'number of cores to use')
    #parser.add_argument('--max_seqs_per_process', default = 1000, type = int,
    #                    help = 'maximum number of query seqs per BLAST+ instance')
    #args = parser.parse_args()

    #split_fasta_pathname_list = split_fasta(args.fasta_path, args.cores, args.max_seqs_per_process)
    #run_blast(split_fasta_pathname_list, args.blastp_path, args.db_dir, args.cores)
    #merged_blast_table = merge_blast_tables(args.fasta_path, split_fasta_pathname_list)
    #filter_blast_table(merged_blast_table)

    fasta_path = '/home/samuelmiller/5-9-17/postnovo/io/predict_042017_toolik_core_2_2_1_1_sem_on_Syn7803_EcoliAE_TIE1Tryp_DvT1NC/postnovo_seqs.faa'
    cores = 12
    max_seqs_per_process = 1000
    blastp_path = '/home/samuelmiller/ncbi-blast-2.6.0+/bin/blastp'
    db_dir = '/home/samuelmiller/refseq_protein/refseq_protein'

    split_fasta_pathname_list = split_fasta(fasta_path, cores, max_seqs_per_process)
    run_blast(split_fasta_pathname_list, blastp_path, db_dir, cores)
    merged_blast_table = merge_blast_tables(fasta_path, split_fasta_pathname_list)
    filtered_blast_table = filter_blast_table(merged_blast_table)
    filtered_blast_table.to_csv(os.path.join(os.path.dirname(fasta_path), 'filtered_blast_table.csv'), sep = '\t', header = True)