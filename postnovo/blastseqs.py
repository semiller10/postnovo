import argparse
import os
import stat
import pkg_resources
import subprocess
import pandas as pd
import re

from pkg_resources import resource_filename
from Bio import Entrez
Entrez.email = 'samuelmiller@uchicago.edu'
from collections import OrderedDict, Counter
from functools import partial
from multiprocessing import Pool, current_process, cpu_count
multiprocessing_taxid_count = 0

raw_blast_table_headers = ['qseqid', 'sgi', 'sacc', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'pident', 'staxids']
merged_blast_table_headers = ['scan', 'xle permutation', 'postnovo score'] + raw_blast_table_headers[1:]
seq_table_headers = ['scan', 'xle permutation', 'seq']

search_ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
taxon_assignment_threshold = 0.9
taxa_assignment_table_headers = ['scan', 'seq', 'postnovo score'] + search_ranks

def main():

    args = parse_args()
    args = make_fasta(args)

    split_fasta_pathname_list = split_fasta(args.faa_fp, args.cores, args.max_seqs_per_process)
    run_blast(split_fasta_pathname_list, args.blastp_fp, args.db_fp, args.cores)
    merged_blast_table = merge_blast_tables(args.faa_fp, split_fasta_pathname_list)
    filtered_blast_table = filter_blast_table(merged_blast_table)
    filtered_blast_table = retrieve_taxonomy(filtered_blast_table, args.cores)
    taxa_assignment_table, taxa_count_table = find_parsimonious_taxonomy(filtered_blast_table)

    filtered_blast_table.to_csv(
        os.path.join(os.path.dirname(args.faa_fp), 'filtered_blast_table.txt'),
        sep='\t', header=True)
    taxa_assignment_table.to_csv(
        os.path.join(os.path.dirname(args.faa_fp), 'taxa_assignment_table.txt'),
        sep='\t', header=True)
    taxa_count_table.to_csv(
        os.path.join(os.path.dirname(args.faa_fp), 'taxa_count_table.txt'),
        sep='\t', header=True)

def parse_args():

    parser = argparse.ArgumentParser(
        description = 'BLAST+ and taxonomic identification of peptide sequences'
        )
    seq_input_group = parser.add_mutually_exclusive_group()
    seq_input_group.add_argument('--faa_fp',
                                 help=('faa input filepath: '
                                       'should have faa extension')
                                 )
    seq_input_group.add_argument('--seq_table_fp',
                                 help=('csv table with two columns of headers and seqs: '
                                       'any PTM symbols are removed from seqs, '
                                       'columns should not have headers')
                                 )
    parser.add_argument('--blastp_fp',
                        help='blastp filepath')
    parser.add_argument('--db_fp',
                        help=('BLAST database filepath, '
                              'as it would be specified in a BLAST+ search: '
                              'e.g., /home/samuelmiller/blast_db/refseq_protein/refseq_protein')
                        )
    parser.add_argument('--cores',
                        type=int,
                        default=1,
                        help = 'number of cores to use')
    parser.add_argument('--max_seqs_per_process',
                        type=int,
                        default=1000,
                        help='maximum number of query seqs per BLAST+ instance')

    args = parser.parse_args()
    check_args(parser, args)
    return args

def check_args(parser, args):

    if args.faa_fp == None and args.seq_table_fp == None:
        parser.error('fasta or seq table input must be provided')
    if args.faa_fp != None:
        if not os.path.exists(args.faa_fp):
            parser.error(args.faa_fp + ' does not exist')
    if args.seq_table_fp != None:
        if not os.path.exists(args.seq_table_fp):
            parser.error(args.seq_table_fp + ' does not exist')
        if os.path.splitext(args.seq_table_fp)[1] != '.csv':
            parser.error(args.seq_table_fp + ' must have a csv extension')

    if args.blastp_fp == None:
        parser.error('blastp filepath must be provided')
    if not os.path.exists(args.blastp_fp):
        parser.error(args.blastp_fp + ' does not exist')

    if args.db_fp == None:
        parser.error('BLAST+ database filepath must be provided')
    db_dir = os.path.dirname(args.db_fp)
    db = os.path.basename(args.db_fp)
    if not os.path.exists(db_dir):
        parser.error(db_dir + ' does not exist')
    if not os.path.exists(
        os.path.join(db_dir, db + '.pal')):
        parser.error(db + ' does not exist in ' + db_dir)

    if args.cores < 1 or args.cores > cpu_count():
        parser.error(str(cpu_count()) + ' cores are available')

    if args.max_seqs_per_process < 1:
        parser.error(str(args.max_seqs_per_process) + ' must be a positive number')

def make_fasta(args):

    if args.faa_fp != None:
        return args

    args.faa_fp = os.path.join(os.path.dirname(args.seq_table_fp),
                                 os.path.splitext(args.seq_table_fp)[0] + '.faa')
    seq_table = pd.read_csv(args.seq_table_fp, header=None)
    seq_table.columns = ['headers', 'seqs']
    seq_table['headers'] = seq_table['headers'].apply(lambda header: '>' + header)
    remove_psm_fn = partial(re.sub, pattern='\(.*\)|\[.*\]|\||\^|\+|\-|\.|[0-9]', repl='')
    seq_table['seqs'] = seq_table['seqs'].apply(lambda seq: remove_psm_fn(string=seq))
    seq_table = seq_table.stack()
    print(seq_table)
    seq_table.to_csv(args.faa_fp, index=False, header=False)

    return args

def split_fasta(faa_fp, cores, max_seqs_per_process):

    fasta_list = open(faa_fp, 'r').readlines()
    split_fasta_pathname_list = []
    fasta_filename = os.path.basename(faa_fp).strip('.faa')
    fasta_dirname = os.path.dirname(faa_fp)
    parent_fasta_size = len(fasta_list) / 2
    if parent_fasta_size % int(parent_fasta_size) > 0:
        raise ValueError('The fasta input must have an even number of lines.')
    child_fasta_size = int(parent_fasta_size / cores)
    remainder = parent_fasta_size % cores

    if child_fasta_size + remainder < max_seqs_per_process:
        for core in range(cores):
            child_fasta_list = fasta_list[core * child_fasta_size * 2: (core + 1) * child_fasta_size * 2]
            child_fasta_fp = os.path.join(fasta_dirname, fasta_filename + '_' + str(core + 1) + '.faa')
            with open(child_fasta_fp, 'w') as child_fasta_file:
                for line in child_fasta_list:
                    child_fasta_file.write(line)
            split_fasta_pathname_list.append(child_fasta_fp)
        with open(child_fasta_fp, 'a') as child_fasta_file:
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

def run_blast(fasta_pathname_list, blastp_fp, db_fp, cores):
    
    blast_batch_fp = resource_filename('postnovo', 'bashscripts/blast_batch.sh')
    with open(blast_batch_fp) as blast_batch_template_file:
        blast_batch_template = blast_batch_template_file.read()
    temp_blast_batch_script = blast_batch_template
    temp_blast_batch_script = temp_blast_batch_script.replace('FASTA_FILES=', 'FASTA_FILES=({})'.format(' '.join(fasta_pathname_list)))
    temp_blast_batch_script = temp_blast_batch_script.replace('MAX_PROCESSES=', 'MAX_PROCESSES={}'.format(cores - 1))
    temp_blast_batch_script = temp_blast_batch_script.replace('BLASTP_PATH=', 'BLASTP_PATH={}'.format(blastp_fp))
    temp_blast_batch_script = temp_blast_batch_script.replace('DB_DIR=', 'DB_DIR={}'.format(db_fp))
    temp_blast_batch_fp = os.path.join(os.path.dirname(blast_batch_fp), 'blast_batch~.sh')
    with open(temp_blast_batch_fp, 'w') as temp_blast_batch_file:
        temp_blast_batch_file.write(temp_blast_batch_script)
    os.chmod(temp_blast_batch_fp, 0o777)
    subprocess.call([temp_blast_batch_fp])

def merge_blast_tables(faa_fp, split_fasta_pathname_list):

    merged_blast_table = pd.DataFrame(columns = merged_blast_table_headers)
    for fasta_pathname in split_fasta_pathname_list:
        out_pathname = os.path.join(os.path.dirname(fasta_pathname),
                                    os.path.basename(fasta_pathname).strip('.faa') + '.out')
        split_blast_table_raw = pd.read_table(out_pathname, names = raw_blast_table_headers)
        qseqid_list = split_blast_table_raw['qseqid'].tolist()
        scan_col_plus_another = [qseqid.split(':') for qseqid in qseqid_list]
        scan_col = pd.Series([split_qseqid[0] for split_qseqid in scan_col_plus_another])
        permut_col_score_col = pd.DataFrame([split_qseqid[1].split(';') for split_qseqid in scan_col_plus_another])
        split_blast_table = pd.concat([scan_col, permut_col_score_col, split_blast_table_raw[split_blast_table_raw.columns[1:]]], axis = 1)
        split_blast_table.columns = merged_blast_table_headers
        merged_blast_table = pd.concat([merged_blast_table, split_blast_table], axis = 0)

        #scan_col_plus_another = split_blast_table_raw['qseqid'].apply(lambda x: pd.Series(x.split(':')))
        # If considering sequences without Xle permutations
        #if len(scan_col_plus_another.columns) == 1:
        #    scan_col_plus_another = pd.concat([scan_col_plus_another, pd.Series(index = scan_col_plus_another)], axis = 1)
        #permut_col_score_col = scan_col_plus_another[1].apply(lambda x: pd.Series(x.split(';')))
        ## If considering data without postnovo scores
        #if len(permut_col_score_col.columns) == 1:
        #    permut_col_score_col = pd.concat([permut_col_score_col, pd.Series(index = permut_col_score_col)], axis = 1)
    merged_blast_table.set_index(['scan', 'xle permutation'], inplace = True)

    seq_table = tabulate_fasta(faa_fp)
    merged_blast_table = seq_table.join(merged_blast_table)
    return merged_blast_table

def tabulate_fasta(faa_fp):

    raw_fasta_table = pd.read_table(faa_fp, header = None)
    fasta_headers_list = raw_fasta_table.ix[::2, 0].tolist()
    scan_col_plus_another = [fasta_header.split(':') for fasta_header in fasta_headers_list]
    scan_col = pd.Series([fasta_header[0].strip('>') for fasta_header in scan_col_plus_another])
    permut_col = pd.Series([split_header[1].split(';')[0] for split_header in scan_col_plus_another])
    seq_col = raw_fasta_table.ix[1::2, 0]
    seq_col.index = range(len(seq_col))
    seq_table = pd.concat([scan_col, permut_col, seq_col], axis = 1)
    seq_table.columns = seq_table_headers
    seq_table.set_index(['scan', 'xle permutation'], inplace = True)
    seq_table['len'] = seq_table['seq'].apply(lambda seq: len(seq))

    #scan_col_plus_another = fasta_headers.apply(lambda x: pd.Series(x.split(':')))
    # If considering sequences without Xle permutations
    #if len(scan_col_plus_another.columns) == 1:
    #    scan_col_plus_another = pd.concat([scan_col_plus_another, pd.Series(index = scan_col_plus_another)], axis = 1)
    #scan_col_plus_another.index = range(len(scan_col_plus_another))
    #scan_col_plus_another[0] = scan_col_plus_another[0].apply(lambda x: x.strip('>'))
    #permut_col_score_col = scan_col_plus_another[1].apply(lambda x: pd.Series(x.split(';')))
    ## If considering data without postnovo scores
    #if len(permut_col_score_col.columns) == 1:
    #    permut_col_score_col = pd.concat([permut_col_score_col, pd.Series(index = permut_col_score_col)], axis = 1)
    #permut_col_score_col.index = range(len(permut_col_score_col))
    #seq_col = raw_fasta_table.ix[1::2, 0]
    #seq_col.index = range(len(seq_col))

    return seq_table

def filter_blast_table(blast_table):

    blast_table['pseq'] = (blast_table['qend'] - blast_table['qstart'] + 1) / blast_table['len']
    filtered_blast_table = blast_table[blast_table['pseq'] == 1]
    filtered_blast_table = filtered_blast_table[filtered_blast_table['pident'] == 100]
    filtered_blast_table.index = filtered_blast_table.index.droplevel('xle permutation')
    return filtered_blast_table

def retrieve_taxonomy(filtered_blast_table, cores):

    #taxid_list = filtered_blast_table['staxids'].tolist()
    unique_taxid_list = filtered_blast_table['staxids'].unique().tolist()
    unique_taxid_dict = {}.fromkeys(unique_taxid_list)
    one_percent_number_taxids = len(unique_taxid_list) / 100 / cores
    rank_dict = OrderedDict().fromkeys(search_ranks)
    #rank_dict = OrderedDict([(rank, []) for rank in search_ranks])
    search_ranks_set = set(search_ranks)
    
    #taxid_taxa_lists = []
    #for taxid in unique_taxid_list:
    #    print(taxid)
    #    unique_taxid_dict[taxid] = query_entrez_taxonomy_db(taxid, rank_dict, search_ranks_set)

    single_var_query_entrez_taxonomy_db = partial(query_entrez_taxonomy_db,
            rank_dict = rank_dict, search_ranks_set = search_ranks_set,
            one_percent_number_taxids = one_percent_number_taxids, cores = cores)
    multiprocessing_pool = Pool(cores)
    taxid_taxa_lists = multiprocessing_pool.map(single_var_query_entrez_taxonomy_db, unique_taxid_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()
    for i, taxid in enumerate(unique_taxid_list):
        unique_taxid_dict[taxid] = taxid_taxa_lists[i]

    list_of_rank_taxa_table_rows = []
    for taxid in filtered_blast_table['staxids']:
        list_of_rank_taxa_table_rows.append(unique_taxid_dict[taxid])
    rank_taxa_table = pd.DataFrame(list_of_rank_taxa_table_rows, columns = search_ranks)
    filtered_blast_table.reset_index(inplace = True)
    filtered_blast_table = pd.concat([filtered_blast_table, rank_taxa_table], axis = 1)
    filtered_blast_table.set_index('scan', inplace = True)

    #for taxid in taxid_dict:
    #    tax_ranks_set = set()
    #    tax_info = Entrez.read(Entrez.efetch(db = 'Taxonomy', id = taxid, retmode = 'xml'))[0]['LineageEx']
    #    for entry in tax_info:
    #        entry_rank = entry['Rank']
    #        if entry_rank in search_ranks_set:
    #            rank_dict[entry_rank].append(entry['ScientificName'])
    #            tax_ranks_set.add(entry_rank)
    #    for rank in search_ranks_set.difference(tax_ranks_set):
    #        rank_dict[rank].append('')
    #for rank in rank_dict:
    #    filtered_blast_table[rank] = rank_dict[rank]

    return filtered_blast_table

def query_entrez_taxonomy_db(taxid, rank_dict, search_ranks_set, one_percent_number_taxids, cores):

    #if current_process()._identity[0] % cores == 1:
    #    global multiprocessing_taxid_count
    #    multiprocessing_taxid_count += 1
    #    if int(multiprocessing_taxid_count % one_percent_number_taxids) == 0:
    #        percent_complete = int(multiprocessing_taxid_count / one_percent_number_taxids)
    #        if percent_complete <= 100:
    #            utils.verbose_print_over_same_line('Entrez taxonomy search progress: ' + str(percent_complete) + '%')

    tax_ranks_set = set()
    tax_info = Entrez.read(Entrez.efetch(db = 'Taxonomy', id = taxid, retmode = 'xml'))[0]['LineageEx']
    for entry in tax_info:
        entry_rank = entry['Rank']
        if entry_rank in search_ranks_set:
            rank_dict[entry_rank] = entry['ScientificName']
            tax_ranks_set.add(entry_rank)
    for rank in search_ranks_set.difference(tax_ranks_set):
        rank_dict[rank] = ''

    return [taxon for taxon in rank_dict.values()]

def find_parsimonious_taxonomy(filtered_blast_table):

    list_of_taxa_assignment_rows = []
    for scan in filtered_blast_table.index.get_level_values('scan').unique():
        scan_table = filtered_blast_table.loc[[scan]]
        scan_table = scan_table.drop_duplicates(subset = ['staxids'])
        for rank_index, rank in enumerate(search_ranks):
            most_common_taxon_count = Counter(scan_table[rank]).most_common(1)[0]
            if most_common_taxon_count[0] != '':
                if most_common_taxon_count[1] >= taxon_assignment_threshold * len(scan_table):
                    scan_table.reset_index(inplace = True)
                    representative_row = scan_table.ix[
                        scan_table[scan_table[rank] == most_common_taxon_count[0]][rank].first_valid_index()
                        ]
                    list_of_taxa_assignment_rows.append([scan] + [representative_row['seq']] + [representative_row['postnovo score']] +\
                        rank_index * ['N/A'] + representative_row[rank:].tolist())
                    break
        else:
            representative_row = scan_table.iloc[0]
            list_of_taxa_assignment_rows.append([scan] + [representative_row['seq']] + [representative_row['postnovo score']] +\
                len(search_ranks) * ['N/A'])
    taxa_assignment_table = pd.DataFrame(list_of_taxa_assignment_rows, columns = taxa_assignment_table_headers)
    taxa_assignment_table.set_index('scan', inplace = True)

    taxa_count_table = pd.DataFrame()
    for rank in search_ranks:
        taxa_counts = Counter(taxa_assignment_table[rank])
        taxa_count_table = pd.concat([taxa_count_table,
                                      pd.Series([taxon for taxon in taxa_counts.keys()], name = rank + ' taxa'),
                                      pd.Series([count for count in taxa_counts.values()], name = rank + ' counts')],
                                     axis = 1)

    return taxa_assignment_table, taxa_count_table

if __name__ == '__main__':
    main()