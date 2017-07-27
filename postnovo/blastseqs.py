import argparse
import copy
import numpy as np
import os
import pandas as pd
import pkg_resources
import pickle as pkl
import re
import subprocess
import sys
import time

from pkg_resources import resource_filename
from xml.etree import ElementTree as ElementTree
from Bio import Entrez
Entrez.email = 'samuelmiller@uchicago.edu'
from collections import OrderedDict, Counter
from functools import partial
from multiprocessing import Pool, current_process, cpu_count
multiprocessing_taxon_count = 0

test_list = ['', '', '', '', '', '', '']

raw_blast_table_headers = \
    ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart',
     'send', 'evalue', 'bitscore', 'sallseqid', 'score', 'nident', 'positive', 'gaps', 'ppos',
     'qframe', 'sframe', 'qseq', 'sseq', 'qlen', 'slen', 'salltitles']

superkingdoms = ['Archaea', 'Bacteria', 'Eukaryota']
search_ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
taxon_assignment_threshold = 0.9
#postnovo_score_penalties = {2: 2, 3: 1, 4: 0}

hmmer_seq_count_limit = 5000
eggnog_output_headers = ['query', 'seed ortholog', 'evalue', 'score', 'predicted name',
                         'go terms', 'kegg pathways', 'tax scope', 'eggnog ogs', 'best og',
                         'cog cat', 'eggnog hmm desc']

taxa_profile_evalue = 0.1

def main():

    #args = parse_args()
    #args = make_fasta(args)

    #split_fasta_pathname_list = split_fasta(args.faa_fp, args.cores, args.max_seqs_per_process)

    ##split_fasta_pathname_list = ['/home/samuelmiller/metagenome_vs_postnovo/blast_output/postnovo_seqs_' + str(i) + '.faa'
    ##                             for i in range(1, 31)]

    #run_blast(split_fasta_pathname_list, args.blastp_fp, args.db_fp, args.cores)

    #last_file_number = int(split_fasta_pathname_list[-1].split('.faa')[0].split('_')[-1])
    #file_prefix = split_fasta_pathname_list[0].split('_1.faa')[0]
    #xml_files = [file_prefix + '_' + str(i) + '.out' for i in range(1, last_file_number + 1)]
    #xml_out = file_prefix + '.merged.xml'
    ## write the full xml output of the BLAST search for use in BLAST2GO as needed
    #merge_xml(xml_files, xml_out)

    ##split_fasta_pathname_list = ['C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_' + str(i) + '.faa'
    ##                             for i in range(1, 31)]
    ##xml_files = ['C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_' + str(i) + '.out'
    ##             for i in range(1, 31)]
    ##split_fasta_pathname_list = ['/home/samuelmiller/metagenome_vs_postnovo/blast_output/postnovo_seqs_test_' + str(i) + '.faa'
    ##                             for i in range(1, 2)]
    ##xml_files = ['/home/samuelmiller/metagenome_vs_postnovo/blast_output/postnovo_seqs_test_' + str(i) + '.out'
    ##             for i in range(1, 2)]

    ## convert xml to tabular format
    #multiprocessing_pool = Pool(args.cores)
    #blast_table_df_list = multiprocessing_pool.map(xml_to_tabular, xml_files)
    #multiprocessing_pool.close()
    #multiprocessing_pool.join()
    #merged_blast_table = pd.concat(blast_table_df_list, axis=0)

    #merged_blast_table.to_csv('/home/samuelmiller/7-25-17/27_2/merged_blast_table.csv', index=False)

    #parsed_blast_table = parse_blast_table(args.from_postnovo, args.faa_fp, merged_blast_table)

    #parsed_blast_table.to_csv('/home/samuelmiller/7-25-17/27_2/parsed_blast_table.csv', index=False)
    ##parsed_blast_table = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/parsed_blast_table.csv', header=0)

    #high_prob_df, low_prob_df, filtered_blast_table = filter_blast_table(parsed_blast_table, args.from_postnovo, args.cores)

    #high_prob_df.to_csv('/home/samuelmiller/7-25-17/27_2/high_prob_df.csv', index=False)
    #low_prob_df.to_csv('/home/samuelmiller/7-25-17/27_2/low_prob_df.csv', index=False)
    #filtered_blast_table.to_csv('/home/samuelmiller/7-25-17/27_2/filtered_df.csv', index=False)
    ##high_prob_df = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/high_prob_df_test_1.csv', header=0)
    ##low_prob_df = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/low_prob_df_test_1.csv', header=0)
    ##filtered_blast_table = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/filtered_df.csv', header=0)

    #augmented_blast_table = retrieve_taxonomy(filtered_blast_table, args.cores, args.from_postnovo)

    #augmented_blast_table.to_csv('/home/samuelmiller/7-25-17/27_2/augmented_blast_table.csv', index=False)
    ##augmented_blast_table = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/augmented_blast_table.csv', header=0)

    #merge_df = augmented_blast_table[['scan_list', 'hit'] + search_ranks]
    #high_prob_df['scan_list'] = high_prob_df['scan_list'].apply(str)
    #high_prob_df = high_prob_df.merge(merge_df, on=['scan_list', 'hit'])
    #low_prob_df['scan_list'] = low_prob_df['scan_list'].apply(str)
    #low_prob_df = low_prob_df.merge(merge_df, on=['scan_list', 'hit'])

    #high_prob_df.to_csv('/home/samuelmiller/7-25-17/27_2/high_prob_df.csv', index=False)
    #low_prob_df.to_csv('/home/samuelmiller/7-25-17/27_2/low_prob_df.csv', index=False)
    ##high_prob_df = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/high_prob_df.csv', header=0)
    ##low_prob_df = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/low_prob_df.csv', header=0)

    #high_prob_taxa_assign_df, high_prob_taxa_count_df = \
    #    find_parsimonious_taxonomy(high_prob_df, args.from_postnovo)
    #high_prob_taxa_assign_df.to_csv('/home/samuelmiller/7-25-17/27_2/high_prob_taxa_assign_df.csv', index=False)
    #high_prob_taxa_count_df.to_csv('/home/samuelmiller/7-25-17/27_2/high_prob_taxa_count_df.csv', index=False)
    ##high_prob_taxa_assign_df = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/high_prob_taxa_assign_df.csv', header=0)
    ##high_prob_taxa_count_df = pd.read_csv('/home/samuelmiller/metagenome_vs_postnovo/blast_output/high_prob_taxa_count_df.csv', header=0)    

    #taxa_profile_dict = {}
    #name_list_raw = high_prob_taxa_count_df['species taxa'].tolist()
    #count_list_raw = high_prob_taxa_count_df['species counts'].tolist()
    #name_list = []
    #count_list = []
    #for i in range(len(name_list_raw)):
    #    if pd.notnull(name_list_raw[i]):
    #        name_list.append(name_list_raw[i])
    #        count_list.append(count_list_raw[i])
    #count_dict = {name_list[i]: count_list[i] for i in range(len(name_list))}
    #taxa_profile_dict['species'] = [i for i in count_dict if count_dict[i] > 1]

    #name_list_raw = high_prob_taxa_count_df['genus taxa'].tolist()
    #count_list_raw = high_prob_taxa_count_df['genus counts'].tolist()
    #name_list = []
    #count_list = []
    #for i in range(len(name_list_raw)):
    #    if pd.notnull(name_list_raw[i]):
    #        name_list.append(name_list_raw[i])
    #        count_list.append(count_list_raw[i])
    #count_dict = {name_list[i]: count_list[i] for i in range(len(name_list))}
    #taxa_profile_dict['genus'] = [i for i in count_dict if count_dict[i] > 1]

    #name_list_raw = high_prob_taxa_count_df['family taxa'].tolist()
    #count_list_raw = high_prob_taxa_count_df['family counts'].tolist()
    #name_list = []
    #count_list = []
    #for i in range(len(name_list_raw)):
    #    if pd.notnull(name_list_raw[i]):
    #        name_list.append(name_list_raw[i])
    #        count_list.append(count_list_raw[i])
    #count_dict = {name_list[i]: count_list[i] for i in range(len(name_list))}
    #taxa_profile_dict['family'] = [i for i in count_dict if count_dict[i] > 1]

    #is_in_profile_list = [False for i in range(len(low_prob_df))]
    #for level in ['family', 'genus', 'species']:
    #    taxa_profile_list = taxa_profile_dict[level]
    #    low_prob_taxa_list = low_prob_df[level].tolist()
    #    for i, low_prob_taxon in enumerate(low_prob_taxa_list):
    #        if not is_in_profile_list[i]:
    #            if low_prob_taxon in taxa_profile_list:
    #                is_in_profile_list[i] = True
    #low_prob_df['is_in_profile'] = is_in_profile_list

    #low_prob_df.to_csv('/home/samuelmiller/7-25-17/27_2/low_prob_df1.csv', index=False)

    #low_prob_profile_df = low_prob_df[low_prob_df['is_in_profile']]

    #low_prob_profile_df.to_csv('/home/samuelmiller/7-25-17/27_2/low_prob_profile_df.csv', index=False)

    #profile_df = pd.concat([high_prob_df, low_prob_profile_df])
    #profile_df.to_csv('/home/samuelmiller/7-25-17/27_2/profile_df.csv', index=False)
    ## Draw up to 10 hits from each scan list group
    ## Evenly sample scan list groups larger than 10, starting with hit 0
    #scan_list_groups = profile_df.groupby('scan_list')
    #sampled_df = scan_list_groups.apply(sample_hits)
    #sampled_df.to_csv('/home/samuelmiller/7-25-17/27_2/sampled_df.csv', index=False)
    ## Make fasta file for eggnog-mapper
    ## Header: >(scan_list)scan lists(hit)hit number
    ## Seq: full subject seq for each hit
    ## Sort into fasta files based on superkingdom (up to 3 files total)
    #eggnog_mapper_faa = os.path.join(
    #    os.path.dirname(args.faa_fp),
    #    os.path.splitext(os.path.basename(args.faa_fp))[0] + '_eggnog_mapper.faa')
    #eggnog_fasta_path_list = make_full_hit_seq_fasta(sampled_df, eggnog_mapper_faa, args.cores)

    #sys.exit()

    ## Run eggnog-mapper on each file using HMMER
    ## Download annotations
    #print('Run eggnog-mapper with the fasta files.')
    #input('Press enter to continue once you have placed the eggnog-mapper output in the fasta directory.')

    eggnog_fasta_path_list = [
        'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\2_2_eggnog\\postnovo_seqs_eggnog_mapper.archaea_0.faa',
        'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\2_2_eggnog\\postnovo_seqs_eggnog_mapper.bacteria_0.faa',
        'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\2_2_eggnog\\postnovo_seqs_eggnog_mapper.bacteria_1.faa',
        'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\2_2_eggnog\\postnovo_seqs_eggnog_mapper.eukaryota_0.faa'
        ]

    # Load as dataframe 
    # Assign predefined column names
    # Concat annotation dfs
    eggnog_output_path_list = [faa + '.emapper.annotations' for faa in eggnog_fasta_path_list]
    eggnog_df = pd.DataFrame(columns=eggnog_output_headers)
    for output_path in eggnog_output_path_list:
        eggnog_output_df = pd.read_csv(output_path, sep='\t', header=None, names=eggnog_output_headers)
        eggnog_df = pd.concat([eggnog_df, eggnog_output_df], axis=0)
    # Split header into two cols for scan lists and hits
    query_list = eggnog_df['query'].tolist()
    query_list = [query.split('(scan_list)')[1] for query in query_list]
    temp_list_of_lists = [query.split('(hit)') for query in query_list]
    scan_list_list = [temp_list[0] for temp_list in temp_list_of_lists]
    hit_list = [temp_list[1] for temp_list in temp_list_of_lists]
    eggnog_df.drop('query', axis=1, inplace=True)
    eggnog_df['scan_list'] = scan_list_list
    eggnog_df['hit'] = hit_list

    scan_list_set_list = list(set(scan_list_list))
    conserv_func_df_list = []
    eggnog_df_groups = eggnog_df.groupby('scan_list')
    for scan_list in scan_list_set_list:
        eggnog_df_group = eggnog_df_groups.get_group(scan_list)
        if (eggnog_df_group['eggnog hmm desc'] == eggnog_df_group['eggnog hmm desc'].iloc[0]).all():
            conserv_func_df_list.append(eggnog_df_group)
    conserv_func_df = pd.concat(conserv_func_df_list)

    # Merge rows of each group
    conserv_func_df = conserv_func_df.fillna('')
    scan_list_set_list = list(set(conserv_func_df['scan_list'].tolist()))
    parsed_conserv_func_df_headers_list = ['seed ortholog', 'predicted name', 'go terms', 'kegg pathways',
                              'tax scope', 'eggnog ogs', 'best og', 'cog cat', 'eggnog hmm desc']
    parsed_conserv_func_df_cols_dict = OrderedDict([(header, []) for header in parsed_conserv_func_df_headers_list])
    parsed_conserv_func_df = pd.DataFrame()
    parsed_conserv_func_df['scan_list'] = scan_list_set_list
    conserv_func_df_groups = conserv_func_df.groupby('scan_list')
    # Loop through each group
    for scan_list in scan_list_set_list:
        conserv_func_df_group = conserv_func_df_groups.get_group(scan_list)
        # Go through each col in group
        # seed ortholog: set intersection
        parsed_conserv_func_df_cols_dict['seed ortholog'].append(
            ','.join(list(set(conserv_func_df_group['seed ortholog']))))
        # evalue: ignore
        # score: ignore
        # predicted name: set intersection
        parsed_conserv_func_df_cols_dict['predicted name'].append(
            ','.join(list(set(conserv_func_df_group['predicted name']))))
        # go terms: set intersection
        list_of_lists = conserv_func_df_group['go terms'].apply(lambda x: x.split(',')).tolist()
        l = [i for sublist in list_of_lists for i in sublist]
        parsed_conserv_func_df_cols_dict['go terms'].append(','.join(list(set(l))))
        # kegg pathways: set intersection
        list_of_lists = conserv_func_df_group['kegg pathways'].apply(lambda x: x.split(',')).tolist()
        l = [i for sublist in list_of_lists for i in sublist]
        parsed_conserv_func_df_cols_dict['kegg pathways'].append(
            ','.join(list(set(l))))
        # tax scope: set intersection
        parsed_conserv_func_df_cols_dict['tax scope'].append(
            ','.join(list(set(conserv_func_df_group['tax scope']))))
        # eggnog ogs: set intersection
        list_of_lists = conserv_func_df_group['eggnog ogs'].apply(lambda x: x.split(',')).tolist()
        l = [i for sublist in list_of_lists for i in sublist]
        parsed_conserv_func_df_cols_dict['eggnog ogs'].append(
            ','.join(list(set(l))))
        # best og: set intersection of first substring before |
        parsed_conserv_func_df_cols_dict['best og'].append(
            ','.join(list(set(
                conserv_func_df_group['best og'].apply(lambda x: x.split('|')[0])))))
        # cog cat: set intersection
        parsed_conserv_func_df_cols_dict['cog cat'].append(
            ','.join(list(set(conserv_func_df_group['cog cat']))))
        # eggnog hmm desc: first row (all rows the same)
        parsed_conserv_func_df_cols_dict['eggnog hmm desc'].append(
            conserv_func_df_group['eggnog hmm desc'].iloc[0])
        # scan_list: already placed in parsed_conserv_func_df
        # hit: ignore
    for header, col in parsed_conserv_func_df_cols_dict.items():
        parsed_conserv_func_df[header] = col
    bob = 1

    low_prob_profile_df = pd.read_csv('C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\low_prob_profile_df.csv', header=0)
    low_prob_profile_df.set_index('scan_list', inplace=True)
    scan_list_set_list = set(scan_list_set_list).intersection(set(low_prob_profile_df.index.tolist()))
    low_prob_profile_df = low_prob_profile_df.loc[scan_list_set_list]
    low_prob_profile_df.reset_index(inplace = True)
    low_prob_profile_df.drop('is_in_profile', axis=1, inplace=True)
    #low_prob_profile_taxa_assign_df, low_prob_profile_taxa_count_df = \
    #    find_parsimonious_taxonomy(low_prob_profile_df, args.from_postnovo)
    low_prob_profile_taxa_assign_df, low_prob_profile_taxa_count_df = \
        find_parsimonious_taxonomy(low_prob_profile_df, True)
    profile_taxa_assign_df = pd.concat([high_prob_taxa_assign_df, low_prob_profile_taxa_assign_df])
    df = parsed_conserv_func_df.merge(profile_taxa_assign_df, how='inner', on='scan_list')

    # Filter low prob profile df by scans in parsed_df
    # Make taxa assignment df from filtered df
    # Merge taxa assign df with filtered df to add seq and taxa info
    

    #augmented_blast_table.to_csv(
    #    os.path.join(os.path.dirname(args.faa_fp),
    #                 os.path.splitext(os.path.basename(args.faa_fp))[0] + '_augmented_blast_table.tsv'),
    #    sep='\t', header=True)
    #taxa_assignment_table.to_csv(
    #    os.path.join(os.path.dirname(args.faa_fp), 
    #                 os.path.splitext(os.path.basename(args.faa_fp))[0] + '_taxa_assignment_table.tsv'),
    #    sep='\t', header=True)
    #taxa_count_table.to_csv(
    #    os.path.join(os.path.dirname(args.faa_fp),
    #                 os.path.splitext(os.path.basename(args.faa_fp))[0] + '_taxa_count_table.tsv'),
    #    sep='\t', header=True)

    #augmented_blast_table = pd.read_csv(
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_augmented_blast_table.tsv',
    #    sep='\t',
    #    header=0
    #    )
    #taxa_assignment_table = pd.read_csv(
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_taxa_assignment_table.tsv',
    #    sep='\t',
    #    header=0
    #    )
    #taxa_count_table = pd.read_csv(
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_taxa_count_table.tsv',
    #    sep='\t',
    #    header=0
    #    )

    #sys.exit(0)

    ## Plan to parse BLAST table
    ## Create taxonomically annotated table
    ## Add a hit col to df
    #augmented_blast_table.reset_index(inplace=True)
    #scan_list_list = augmented_blast_table['scan_list'].tolist()
    #last_scan_list = scan_list_list[0]
    #hit_list = [0]
    #for scan_list in scan_list_list[1:]:
    #    if scan_list != last_scan_list:
    #        last_scan_list = scan_list
    #        hit_list.append(0)
    #    else:
    #        hit_list.append(hit_list[-1] + 1)
    #augmented_blast_table['hit'] = hit_list

    ## Draw up to 10 hits from each scan list group
    ## Evenly sample scan list groups larger than 10, starting with hit 0
    #scan_list_groups = augmented_blast_table.groupby('scan_list')
    #sampled_table = scan_list_groups.apply(sample_hits)
    ## Make fasta file for eggnog-mapper
    ## Header: >(scan_list)scan lists(hit)hit number
    ## Seq: full subject seq for each hit
    ## Sort into fasta files based on superkingdom (up to 3 files total)
    #eggnog_mapper_first_faa = os.path.join(
    #    os.path.dirname(args.faa_fp),
    #    os.path.splitext(os.path.basename(args.faa_fp))[0] + '_eggnog_mapper_first_round.faa')
    #eggnog_fasta_path_list = make_full_hit_seq_fasta(sampled_table, eggnog_mapper_first_faa, args.cores)

    #eggnog_fasta_path_list = [
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.archaea_0.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.bacteria_0.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.bacteria_1.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.bacteria_2.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.bacteria_3.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.bacteria_4.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.eukaryota_0.faa',
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_eggnog_mapper_first_round.eukaryota_1.faa'
    #    ]

    #eggnog_fasta_path_list = [
    #    '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.archaea_0.faa',
    #    '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.bacteria_0.faa',
    #    '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.bacteria_1.faa',
    #    '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.eukaryota_0.faa'
    #    ]

    ## Run eggnog-mapper on each file using HMMER
    ## Download annotations
    #print('Run eggnog-mapper with the fasta files.')
    #input('Press enter to continue once you have placed the eggnog-mapper output in the fasta directory.')
    ## Load as dataframe 
    ## Assign predefined column names
    ## Concat annotation dfs
    eggnog_output_path_list = [faa + '.emapper.annotations' for faa in eggnog_fasta_path_list]
    eggnog_first_round_df = pd.DataFrame(columns=eggnog_output_headers)
    for output_path in eggnog_output_path_list:
        eggnog_output_df = pd.read_csv(output_path, sep='\t', header=None, names=eggnog_output_headers)
        eggnog_first_round_df = pd.concat([eggnog_first_round_df, eggnog_output_df], axis=0)
    # Split header into two cols for scan lists and hits
    query_list = eggnog_first_round_df['query'].tolist()
    query_list = [query.split('(scan_list)')[1] for query in query_list]
    temp_list_of_lists = [query.split('(hit)') for query in query_list]
    scan_list_list = [temp_list[0] for temp_list in temp_list_of_lists]
    hit_list = [temp_list[1] for temp_list in temp_list_of_lists]
    eggnog_first_round_df.drop('query', axis=1, inplace=True)
    eggnog_first_round_df['scan_list'] = scan_list_list
    eggnog_first_round_df['hit'] = hit_list

    # Loop through each scan_list group
    scan_list_set_list = list(set(scan_list_list))
    conserv_func_df_list = []
    nonconserv_func_df_list = []
    for scan_list in scan_list_set_list:
        scan_list_df = eggnog_first_round_df[eggnog_first_round_df['scan_list'] == scan_list]
        # If all the eggnog annotations are the same, the group is functionally conserved
        if (scan_list_df['eggnog hmm desc'] == scan_list_df['eggnog hmm desc'].iloc[0]).all():
            conserv_func_df_list.append(scan_list_df)
        else:
            nonconserv_func_df_list.append(scan_list_df)
    conserv_func_df = pd.concat(conserv_func_df_list)
    nonconserv_func_df = pd.concat(nonconserv_func_df_list)

    # Attach BLAST scores to conserv_func_df
    blast_score_df = augmented_blast_table[['scan_list', 'hit', 'evalue']]
    conserv_func_df = conserv_func_df.merge(augmented_blast_table, how='inner', on=['scan_list', 'hit'])
    # Attach consensus taxonomic info to conserv_func_df
    conserv_func_df = conserv_func_df.merge(taxa_assignment_table, how='inner', on=['scan_list'])

    # Find the scan_lists in conserv_func_df which
    # 1. are taxonomically constrained: all high-quality hits are to organisms from the same family, at least
    # 2. have a query seq of high probability
    # 3. have at least one hit with a high probability of not being random
    # Call this the taxonomic profile of the dataset
    taxa_profile_df = conserv_func_df[(conserv_func_df['seq_score'] >= min_taxa_profile_score) &
                                      (conserv_func_df['evalue'] <= min_taxa_profile_evalue) & 
                                      ((pd.notnull(conserv_func_df['species'])) |
                                       (pd.notnull(conserv_func_df['genus'])) | 
                                       (pd.notnull(conserv_func_df['family'])))]
    conserv_taxon_dict = {}
    conserv_taxon_dict['species'] = list(set(taxa_profile_df['species'].tolist()))
    conserv_taxon_dict['species'].remove(np.nan)
    conserv_taxon_dict['genus'] = list(set(taxa_profile_df['genus'].tolist()))
    conserv_taxon_dict['genus'].remove(np.nan)
    conserv_taxon_dict['family'] = list(set(taxa_profile_df['family'].tolist()))
    conserv_taxon_dict['family'].remove(np.nan)

    # The usefulness of the results in nonconserv_func_df is as yet unknown
    # Find scan_lists that fit the taxonomic profile
    nonconserv_func_df = nonconserv_func_df.merge(augmented_blast_table, how='inner', on=['scan_list', 'hit'])
    is_in_profile_list = [0 for i in range(len(nonconserv_func_df))]
    for taxon in ['family', 'genus', 'species']:
        conserv_taxon_set_list = conserv_taxon_dict[taxon]
        nonconserv_taxon_list = nonconserv_func_df[taxon].tolist()
        for i, nonconserv_taxon in nonconserv_taxon_list:
            if not is_in_profile_list[i]:
                if nonconserv_taxon_list[i] in conserv_taxon_set_list:
                    is_in_profile_list[i] = 1

    taxa_profile_df = conserv_func_df[(conserv_func_df['seq_score'] >= min_taxa_profile_score) &
                                      ((pd.notnull(conserv_func_df['species'])) |
                                       (pd.notnull(conserv_func_df['genus'])) | 
                                       (pd.notnull(conserv_func_df['family'])))]


    nonconserv_func_df = nonconserv_func_df.merge(augmented_blast_table, how='inner', on='scan_list')

    # Find scan groups in df1 that have the same taxa at the species, genus or family level and seq scores > 0.8
    # Place these groups into df1.1
    # Make lists, then sets, of species, genus and family in df1.1
    # Remove empty strings from sets but not lists
    # Count each taxon in each list
    # Make corresponding dicts indicating the proportion of each taxon in df1.1
    # Loop through each scan group in df2.1
    # Loop through each row of group
    # Make a list for row taxon presence in df1.1 (initialize with empty strings)
    # If row has family ID, search for ID in df1.1 family set
    # If found, mark taxon presence as ID, else leave blank
    # If row does not have family ID, do the same for genus and species
    # If row does not have family, genus, or species ID, break
    # After assembling list, use the occurrence dicts to make a list of proportions
    # If one taxon has proportion >= 0.01, and no others do, transfer the good rows to df2.2
    # Group df2.2 by scan, then transfer top-scoring row of each group to df2.3
    # Merge df2.3 with taxonomic df by scan, hit, only retaining rows from original df2.3
    # Group df1 by scan, then transfer top-scoring row of each group to df1.2
    # Concat df1.2 with df2.3 to produce df3
    # For those rows of df3 that do not have functional annotation
    # Generate fasta files and run through eggnog-mapper as before
    # Merge results into df3, so that every row should have a functional annotation


def sample_hits(scan_group, sample_size = 10):

    scan_group_size = len(scan_group)
    if scan_group_size <= 10:
        return scan_group
    else:
        scan_group_rows = list(range(scan_group_size))
        div, mod = divmod(len(scan_group_rows), sample_size)
        sample_rows = [scan_group_rows[i * div + min(i, mod)] for i in range(sample_size)]
        return scan_group.iloc[sample_rows]

def make_full_hit_seq_fasta(df, write_path, cores):
    
    gi_list = df['sallseqid'].tolist()
    multiprocessing_pool = Pool(cores)
    full_hit_seq_list = multiprocessing_pool.map(query_ncbi_protein, gi_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    df['scan_list'] = df['scan_list'].apply(str)
    df['hit'] = df['hit'].apply(str)
    df['full seq'] = full_hit_seq_list

    eggnog_fasta_path_list = []
    for superkingdom in superkingdoms:
        superkingdom_df = df[df['superkingdom'] == superkingdom]
        superkingdom_scan_list = superkingdom_df['scan_list'].tolist()
        superkingdom_hit_list = superkingdom_df['hit'].tolist()
        superkingdom_seq_list = superkingdom_df['full seq'].tolist()
        superkingdom_header_list = [
            '>' + '(scan_list)' + superkingdom_scan_list[i] + '(hit)' + superkingdom_hit_list[i]
            for i in range(len(superkingdom_scan_list))
            ]
        superkingdom_header_seq_dict = {
            superkingdom_header_list[i]: superkingdom_seq_list[i]
            for i in range(len(superkingdom_header_list))
            }

        previous_last_row = 0
        for i in range(len(superkingdom_header_list) // hmmer_seq_count_limit + 1):
            superkingdom_write_path = write_path.replace('.faa', '.' + superkingdom.lower() + '_' + str(i) + '.faa')
            eggnog_fasta_path_list.append(superkingdom_write_path)
            with open(superkingdom_write_path, 'w') as f:
                for j, header in enumerate(superkingdom_header_list[previous_last_row:]):
                    f.write(header + '\n')
                    f.write(superkingdom_header_seq_dict[header] + '\n')
                    if j+1 == hmmer_seq_count_limit:
                        previous_last_row = (j+1) * (i+1)
                        break
    return eggnog_fasta_path_list
    
def query_ncbi_protein(gi):

    while True:
        try:
            full_seq = Entrez.read(Entrez.efetch(db='Protein', id=gi, retmode='xml'))[0]['GBSeq_sequence']
            break
        except:
            print(gi + ': Entrez query no response')
            time.sleep(10)

    return full_seq

def xml_to_tabular(merged_xml_filename):

    colnames = raw_blast_table_headers

    out_fmt = 'ext'
    extended = True
    cols = None

    merged_blast_table = []
    merged_blast_table.append(colnames)
    re_default_query_id = re.compile("^Query_\d+$")
    assert re_default_query_id.match("Query_101")
    assert not re_default_query_id.match("Query_101a")
    assert not re_default_query_id.match("MyQuery_101")
    re_default_subject_id = re.compile("^Subject_\d+$")
    assert re_default_subject_id.match("Subject_1")
    assert not re_default_subject_id.match("Subject_")
    assert not re_default_subject_id.match("Subject_12a")
    assert not re_default_subject_id.match("TheSubject_1")

    blast_program = None
    context = ElementTree.iterparse(merged_xml_filename, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
    for event, elem in context:
        if event == "end" and elem.tag == "BlastOutput_program":
            blast_program = elem.text
        if event == "end" and elem.tag == "Iteration":
            qseqid = elem.findtext("Iteration_query-ID")
            if re_default_query_id.match(qseqid):
                qseqid = elem.findtext("Iteration_query-def").split(None, 1)[0]
            qlen = int(elem.findtext("Iteration_query-len"))

            for hit in elem.findall("Iteration_hits/Hit"):
                sseqid = hit.findtext("Hit_id").split(None, 1)[0]
                hit_def = sseqid + " " + hit.findtext("Hit_def")
                if re_default_subject_id.match(sseqid) and sseqid == hit.findtext("Hit_accession"):
                    hit_def = hit.findtext("Hit_def")
                    sseqid = hit_def.split(None, 1)[0]
                if sseqid.startswith("gnl|BL_ORD_ID|") and sseqid == "gnl|BL_ORD_ID|" + hit.findtext("Hit_accession"):
                    hit_def = hit.findtext("Hit_def")
                    sseqid = hit_def.split(None, 1)[0]
                for hsp in hit.findall("Hit_hsps/Hsp"):
                    nident = hsp.findtext("Hsp_identity")
                    length = hsp.findtext("Hsp_align-len")
                    pident = "%0.3f" % (100 * float(nident) / float(length))

                    q_seq = hsp.findtext("Hsp_qseq")
                    h_seq = hsp.findtext("Hsp_hseq")
                    m_seq = hsp.findtext("Hsp_midline")
                    assert len(q_seq) == len(h_seq) == len(m_seq) == int(length)
                    gapopen = str(len(q_seq.replace('-', ' ').split()) - 1 +
                                    len(h_seq.replace('-', ' ').split()) - 1)

                    mismatch = m_seq.count(' ') + m_seq.count('+') - q_seq.count('-') - h_seq.count('-')
                    expected_mismatch = len(q_seq) - sum(1 for q, h in zip(q_seq, h_seq)
                                                            if q == h or q == "-" or h == "-")
                    xx = sum(1 for q, h in zip(q_seq, h_seq) if q == "X" and h == "X")
                    if not (expected_mismatch - q_seq.count("X") <= int(mismatch) <= expected_mismatch + xx):
                        sys.exit("%s vs %s mismatches, expected %i <= %i <= %i"
                                    % (qseqid, sseqid, expected_mismatch - q_seq.count("X"),
                                    int(mismatch), expected_mismatch))

                    expected_identity = sum(1 for q, h in zip(q_seq, h_seq) if q == h)
                    if not (expected_identity - xx <= int(nident) <= expected_identity + q_seq.count("X")):
                        sys.exit("%s vs %s identities, expected %i <= %i <= %i"
                                    % (qseqid, sseqid, expected_identity, int(nident),
                                    expected_identity + q_seq.count("X")))

                    evalue = hsp.findtext("Hsp_evalue")
                    if evalue == "0":
                        evalue = "0.0"
                    else:
                        evalue = "%0.0e" % float(evalue)

                    bitscore = float(hsp.findtext("Hsp_bit-score"))
                    if bitscore < 100:
                        bitscore = "%0.1f" % bitscore
                    else:
                        bitscore = "%i" % bitscore

                    values = [qseqid,
                              sseqid,
                              pident,
                              length,  # hsp.findtext("Hsp_align-len")
                              str(mismatch),
                              gapopen,
                              hsp.findtext("Hsp_query-from"),  # qstart,
                              hsp.findtext("Hsp_query-to"),  # qend,
                              hsp.findtext("Hsp_hit-from"),  # sstart,
                              hsp.findtext("Hsp_hit-to"),  # send,
                              evalue,  # hsp.findtext("Hsp_evalue") in scientific notation
                              bitscore  # hsp.findtext("Hsp_bit-score") rounded
                              ]

                    if extended:
                        try:
                            sallseqid = ";".join(name.split(None, 1)[0] for name in hit_def.split(" >"))
                            salltitles = "<>".join(name.split(None, 1)[1] for name in hit_def.split(" >"))
                        except IndexError as e:
                            sys.exit("Problem splitting multuple hits?\n%r\n--> %s" % (hit_def, e))
                        # print(hit_def, "-->", sallseqid)
                        positive = hsp.findtext("Hsp_positive")
                        ppos = "%0.2f" % (100 * float(positive) / float(length))
                        qframe = hsp.findtext("Hsp_query-frame")
                        sframe = hsp.findtext("Hsp_hit-frame")
                        if blast_program == "blastp":
                            # Probably a bug in BLASTP that they use 0 or 1 depending on format
                            if qframe == "0":
                                qframe = "1"
                            if sframe == "0":
                                sframe = "1"
                        slen = int(hit.findtext("Hit_len"))
                        values.extend([sallseqid,
                                       hsp.findtext("Hsp_score"),  # score,
                                       nident,
                                       positive,
                                       hsp.findtext("Hsp_gaps"),  # gaps,
                                       ppos,
                                       qframe,
                                       sframe,
                                       q_seq,
                                       h_seq,
                                       str(qlen),
                                       str(slen),
                                       salltitles,
                                       ])
                    if cols:
                        # Only a subset of the columns are needed
                        values = [values[colnames.index(c)] for c in cols]
                    merged_blast_table.append(values)
            # prevents ElementTree from growing large datastructure
            root.clear()
            elem.clear()

    merged_blast_table = pd.DataFrame(merged_blast_table[1:], columns=merged_blast_table[0])

    return merged_blast_table

def parse_args():

    parser = argparse.ArgumentParser(
        description = 'BLAST+ and taxonomic identification of peptide sequences'
        )
    parser.add_argument('--from_postnovo',
                        #default=True,
                        action='store_true',
                        help='seqs were generated by postnovo')
    seq_input_group = parser.add_mutually_exclusive_group()
    seq_input_group.add_argument('--faa_fp',
                                 default='C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs.faa',
                                 #default='/home/samuelmiller/metagenome_vs_postnovo/blast_output/postnovo_seqs.faa',
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
                        default=4,
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

    if not args.faa_fp:
        args.faa_fp = os.path.join(os.path.dirname(args.seq_table_fp),
                                   os.path.splitext(args.seq_table_fp)[0] + '.faa')
        seq_table = pd.read_csv(args.seq_table_fp, header=None)
        seq_table.columns = ['headers', 'seqs']
        seq_table['headers'] = seq_table['headers'].apply(lambda header: '>' + header)
        remove_psm_fn = partial(re.sub, pattern='\(.*\)|\[.*\]|\||\^|\+|\-|\.|[0-9]', repl='')
        seq_table['seqs'] = seq_table['seqs'].apply(lambda seq: remove_psm_fn(string=seq))
        seq_table = seq_table.stack()
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
    
    # Script name modified
    blast_batch_fp = resource_filename('postnovo', 'bashscripts/blast_batch_xml.sh')
    with open(blast_batch_fp) as blast_batch_template_file:
        blast_batch_template = blast_batch_template_file.read()
    temp_blast_batch_script = blast_batch_template
    temp_blast_batch_script = temp_blast_batch_script.replace('FASTA_FILES=', 'FASTA_FILES=({})'.format(' '.join(fasta_pathname_list)))
    temp_blast_batch_script = temp_blast_batch_script.replace('MAX_PROCESSES=', 'MAX_PROCESSES={}'.format(cores - 1))
    temp_blast_batch_script = temp_blast_batch_script.replace('BLASTP_PATH=', 'BLASTP_PATH={}'.format(blastp_fp))
    temp_blast_batch_script = temp_blast_batch_script.replace('DB_DIR=', 'DB_DIR={}'.format(db_fp))
    # Script name modified
    temp_blast_batch_fp = os.path.join(os.path.dirname(blast_batch_fp), 'blast_batch_xml~.sh')
    with open(temp_blast_batch_fp, 'w') as temp_blast_batch_file:
        temp_blast_batch_file.write(temp_blast_batch_script)
    os.chmod(temp_blast_batch_fp, 0o777)
    subprocess.call([temp_blast_batch_fp])

def merge_xml(xml_files, xml_out):
    """
    Adapted from Galaxy toolkit BlastXMLmerge script
    Merge BLAST xml output and write to file
    Return file path
    """

    out = open(xml_out, "w")
    h = None
    for f in xml_files:
        h = open(f)
        body = False
        header = h.readline()
        if not header:
            out.close()
            h.close()
            raise ValueError("BLAST XML file %s was empty" % f)
        if header.strip() != '<?xml version="1.0"?>':
            out.write(header) # for diagnosis
            out.close()
            h.close()
            raise ValueError("%s is not an XML file!" % f)
        line = h.readline()
        header += line
        if line.strip() not in [
            '<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" '
            '"http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">',
            '<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" "NCBI_BlastOutput.dtd">'
            ]:
            out.write(header) # for diagnosis
            out.close()
            h.close()
            raise ValueError("%s is not a BLAST XML file!" % f)
        while True:
            line = h.readline()
            if not line:
                out.write(header) # for diagnosis
                out.close()
                h.close()
                raise ValueError("BLAST XML file %s ended prematurely" % f)
            header += line
            if "<Iteration>" in line:
                break
            if len(header) > 10000:
                # Something has gone wrong, don't load too much into memory!
                # Write what we have to the merged file for diagnostics
                out.write(header)
                out.close()
                h.close()
                raise ValueError("BLAST XML file %s has too long a header!" % f)
        if "<BlastOutput>" not in header:
            out.close()
            h.close()
            raise ValueError("%s is not a BLAST XML file:\n%s\n..." % (f, header))
        if f == xml_files[0]:
            out.write(header)
            old_header = header
        elif old_header[:300] != header[:300]:
		    # Enough to check <BlastOutput_program> and <BlastOutput_version> match
            out.close()
            h.close()
            raise ValueError(
                "BLAST XML headers don't match for %s and %s - have:\n%s\n...\n\nAnd:\n%s\n...\n" \
				 % (xml_files[0], f, old_header[:300], header[:300])
                 )
        else:
            out.write("    <Iteration>\n")
        for line in h:
            if "</BlastOutput_iterations>" in line:
                break
            #TODO - Increment <Iteration_iter-num> and if required automatic query names
            #like <Iteration_query-ID>Query_3</Iteration_query-ID> to be increasing?
            out.write(line)
        h.close()
    out.write("  </BlastOutput_iterations>\n")
    out.write("</BlastOutput>\n")
    out.close()

def parse_blast_table(from_postnovo, faa_fp, raw_blast_table):

    postnovo_merged_headers = ['scan_list', 'xle_permutation', 'precursor_mass', 'seq_score', 'seq_origin'] + raw_blast_table_headers[1:]

    if from_postnovo:
        id_type = 'scan_list'
    else:
        id_type = 'qseqid'

    if from_postnovo:
        # Split the info in the headers into cols of info
        qseqid_list = raw_blast_table['qseqid'].tolist()
        # qseqid format is, ex., (scan_list)1,2(xle_permutation)0(precursor_mass)1000.000(seq_score)0.55(seq_origin)postnovo
        qseqid_list = [qseqid.split('(scan_list)')[1] for qseqid in qseqid_list]
        temp_list_of_lists = [qseqid.split('(xle_permutation)') for qseqid in qseqid_list]
        scan_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(precursor_mass)') for temp_list in temp_list_of_lists]
        permut_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(seq_score)') for temp_list in temp_list_of_lists]
        mass_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(seq_origin)') for temp_list in temp_list_of_lists]
        score_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        origin_col = pd.Series(
            [temp_list[1] for temp_list in temp_list_of_lists])
        parsed_blast_table = pd.concat(
            [scan_col,
             permut_col,
             mass_col,
             score_col,
             origin_col,
             raw_blast_table[raw_blast_table.columns[1:]].reset_index(drop=True)],
            axis = 1)
        parsed_blast_table.columns = postnovo_merged_headers
    else:
        parsed_blast_table = raw_blast_table

    # Add a hit column to df
    scan_list_list = parsed_blast_table[id_type].tolist()
    last_scan_list = scan_list_list[0]
    hit_list = [0]
    for scan_list in scan_list_list[1:]:
        if scan_list != last_scan_list:
            last_scan_list = scan_list
            hit_list.append(0)
        else:
            hit_list.append(hit_list[-1] + 1)
    parsed_blast_table['hit'] = hit_list

    #if from_postnovo:
    #    parsed_blast_table.set_index(['scan_list', 'xle_permutation'], inplace = True)
    #else:
    #    parsed_blast_table.set_index(['qseqid'], inplace = True)

    seq_table = tabulate_fasta(from_postnovo, faa_fp)
    #merged_blast_table = seq_table.join(parsed_blast_table)
    if from_postnovo:
        merged_blast_table = seq_table.merge(parsed_blast_table, on=['scan_list', 'xle_permutation'])
    else:
        merged_blast_table = seq_table.merge(parsed_blast_table, on='qseqid')

    return merged_blast_table

def tabulate_fasta(from_postnovo, faa_fp):

    raw_fasta_input = pd.read_table(faa_fp, header = None)
    fasta_headers_list = raw_fasta_input.ix[::2, 0].tolist()
    seq_col = raw_fasta_input.ix[1::2, 0]
    seq_col.index = range(len(seq_col))

    if from_postnovo:
        # header format is, ex., >(scan_list)1,2(xle_permutation)0(precursor_mass)1000.000(seq_score)0.55(seq_origin)postnovo
        fasta_headers_list = [fasta_header.strip('>(scan_list)') for fasta_header in fasta_headers_list]
        temp_list_of_lists = [header.split('(xle_permutation)') for header in fasta_headers_list]
        scan_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(precursor_mass)') for temp_list in temp_list_of_lists]
        permut_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        seq_table = pd.concat([scan_col, permut_col, seq_col], axis = 1)
        seq_table.columns = ['scan_list', 'xle_permutation', 'seq']
        #seq_table.set_index(['scan_list', 'xle_permutation'], inplace = True)
    else:
        id_col = pd.Series([fasta_header.strip('>') for fasta_header in fasta_headers_list])
        seq_table = pd.concat([id_col, seq_col], axis=1)
        seq_table.columns = ['qseqid', 'seq']
        #seq_table.set_index(['qseqid'], inplace=True)
    seq_table['len'] = seq_table['seq'].apply(lambda seq: len(seq))

    return seq_table

def filter_blast_table(blast_table, from_postnovo, cores):

    if from_postnovo:
        id_type = 'scan_list'
    else:
        id_type = 'qseqid'

    blast_table['evalue'] = blast_table['evalue'].apply(float)
    id_set_list = list(set(blast_table[id_type].tolist()))
    high_prob_df_list = []
    low_prob_df_list = []
    filtered_df_list = []
    gb = blast_table.groupby(id_type)
    for id in id_set_list:
        id_df = gb.get_group(id)
        if id_df['evalue'].min() <= taxa_profile_evalue:
            high_prob_df = id_df[id_df['evalue'] <= taxa_profile_evalue]
            high_prob_df_list.append(high_prob_df)
            filtered_df_list.append(high_prob_df)
        else:
            low_prob_df_list.append(id_df)
            filtered_df_list.append(id_df)
    high_prob_df = pd.concat(high_prob_df_list)
    low_prob_df = pd.concat(low_prob_df_list)
    filtered_df = pd.concat(filtered_df_list)

    salltitles = filtered_df['salltitles'].tolist()
    sseqids = filtered_df['sseqid'].tolist()
    subject_info_list = list(zip(salltitles, sseqids))
    multiprocessing_pool = Pool(cores)
    taxa_list = multiprocessing_pool.map(extract_taxon, subject_info_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()
    filtered_df['taxon'] = taxa_list

    # Multithreading is not perfect
    missed_rows = filtered_df[pd.isnull(filtered_df['taxon'])]
    if len(missed_rows) > 0:
        salltitles = missed_rows['salltitles'].tolist()
        sseqids = missed_rows['sseqid'].tolist()
        subject_info_list = list(zip(salltitles, sseqids))
        missed_taxa = []
        for l in subject_info_list:
            missed_taxa.append(extract_taxon(l))
        for i, j in enumerate(missed_rows.index):
            taxa_list[j] = missed_taxa[i]
        filtered_df['taxon'] = taxa_list

    merge_df = filtered_df[[id_type, 'hit', 'taxon']]
    high_prob_df = high_prob_df.merge(merge_df, on=[id_type, 'hit'])
    low_prob_df = low_prob_df.merge(merge_df, on=[id_type, 'hit'])

    return high_prob_df, low_prob_df, filtered_df

#def filter_blast_table(blast_table, from_postnovo):

#    # Example to explain calculation of deletion count and nonidentical residues:
#    # One deletion, one insertion, one substitution
#    # pident = 26/29 * 100
#    # Query  1    SRRTKGNNPVLIGEP-VGKTAIVDGLAQK  28
#    #             SRRTK NNPVLIGEP VGKTAIV+GLAQK
#    # Sbjct  221  SRRTK-NNPVLIGEPGVGKTAIVEGLAQK  248
#    # dels = (gaps - (qend - qstart) - (send - sstart)) / 2
#    #      = (2    - (28   - 1)      - (248  - 221))    / 2 = 1
#    # nonident = len - (qend - qstart + 1 + del) * pident / 100
#    #          = 28  - (28   - 1      + 1 + 1)   * 26     / 29 = 2
#    # This implies that 2 residues (at the insertion and substitution sites)
#    # are nonidentical in the query seq

#    blast_table[['gaps', 'qend', 'qstart', 'send', 'sstart']] = \
#        blast_table[['gaps', 'qend', 'qstart', 'send', 'sstart']].applymap(int)
#    blast_table['pident'] = blast_table['pident'].apply(float)
#    blast_table['del'] = (blast_table['gaps'] -
#                          ((blast_table['qend'] - blast_table['qstart']) - 
#                          (blast_table['send'] - blast_table['sstart']))) / 2
#    blast_table['nonident'] = \
#        round(
#            blast_table['len'] \
#                - (blast_table['qend'] - blast_table['qstart'] + 1 + blast_table['del']) \
#                * (blast_table['pident'] / 100),
#            0)

#    if from_postnovo:
#        #blast_table['postnovo score'] = blast_table['postnovo score'].apply(float)
#        #blast_table['score penalty'] = blast_table['postnovo score'].apply(
#        #    lambda score: postnovo_score_penalties[int(score / 0.2)])
#        #filtered_blast_table = blast_table[
#        #    blast_table['len'] - blast_table['nonident'] - blast_table['score penalty'] >= 9]
#        filtered_blast_table = blast_table
#        scan_groups = filtered_blast_table.groupby(level='scan_list', group_keys=False)
#        filtered_blast_table = scan_groups.apply(
#            lambda g: g[(g['nonident'] == g['nonident'].min()) | (g['bitscore'] == g['bitscore'].max())])
#        filtered_blast_table.index = filtered_blast_table.index.droplevel('xle_permutation')
#    else:
#        #filtered_blast_table = blast_table[
#        #    blast_table['len'] - blast_table['nonident'] >= 9]
#        filtered_blast_table = blast_table
#        qseqid_groups = filtered_blast_table.groupby(filtered_blast_table.index, group_keys=False)
#        filtered_blast_table = qseqid_groups.apply(
#            lambda g: g[(g['nonident'] == g['nonident'].min()) | (g['bitscore'] == g['bitscore'].max())])

#    filtered_blast_table['taxon'] = filtered_blast_table['salltitles'].apply(
#        lambda x: extract_taxon(x))

#    return filtered_blast_table

def extract_taxon(l):
    salltitle = l[0]
    sseqid = l[1]
    if '<>' in salltitle:
        # Example 1
        # salltitle = '[Salmonella] <> [a]'
        # Example 2
        # '[a]b[Proteobacteria] <> [d]'
        # Example 3
        # 'DNA-directed RNA polymerase,possible RNA polymerase A/beta'/A'' subunit, long PHYSPTS repeat at'
        hits = salltitle.split('<>')
        hit0 = hits[0]
    else:
        # More frequently, there are not multiple parts to the hit
        hit0 = salltitle

    in_brackets = re.search('\[(.*)\]', hit0)
    # Example 1
    # in_brackets.group(1) = 'Salmonella'
    # Example 2
    # in_brackets.group(1) = 'a]b[Proteobacteria'
    # Example 3
    # in_brackets = None
    if in_brackets == None:
        taxon = Entrez.read(Entrez.efetch(db='Protein', id=sseqid, retmode='xml'))[0]['GBSeq_organism']
        return taxon
        # Example 3 returns 'Cryptosporidium parvum Iowa II' from sseqid, gi|66359288|ref|XP_626822.1|
    else:
        try:
            re.search('\](.*)\[', in_brackets.group(1)).groups()
            # Example 1 throws error here
            innermost_brackets = re.findall(r'(?<=\[).+?(?=\])', hit0)
            # Example 2
            # innermost_brackets = ['a', 'Proteobacteria']
            for i in innermost_brackets:
                try:
                    if Entrez.read(Entrez.esearch(db='Taxonomy', term='\"' + i + '\"'))['IdList']:
                        return i
                        # Example 2
                        # returns 'Proteobacteria'
                except:
                    pass
        except AttributeError:
            return in_brackets.group(1)
            # Example 1
            # returns 'Salmonella'

def retrieve_taxonomy(filtered_blast_table, cores, from_postnovo):

    unique_taxon_list = filtered_blast_table['taxon'].unique().tolist()
    unique_taxon_dict = {}.fromkeys(unique_taxon_list)
    one_percent_number_taxa = len(unique_taxon_list) / 100 / cores
    rank_dict = OrderedDict().fromkeys(search_ranks)
    search_ranks_set = set(search_ranks)
    
    #taxon_taxa_lists = []
    #for taxon in unique_taxon_list:
    #    taxon_taxa_lists.append(query_entrez_taxonomy_db(taxon, rank_dict, search_ranks_set, one_percent_number_taxa, cores))
        
    single_var_query_entrez_taxonomy_db = partial(query_entrez_taxonomy_db,
            rank_dict = rank_dict, search_ranks_set = search_ranks_set,
            one_percent_number_taxa = one_percent_number_taxa, cores = cores)
    multiprocessing_pool = Pool(cores)
    taxon_taxa_lists = multiprocessing_pool.map(single_var_query_entrez_taxonomy_db, unique_taxon_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    with open('/home/samuelmiller/metagenome_vs_postnovo/blast_output/unique_taxon_dict.pkl', 'wb') as f:
        pkl.dump(unique_taxon_dict, f, 2)
    with open('/home/samuelmiller/metagenome_vs_postnovo/blast_output/taxon_taxa_lists.pkl', 'wb') as f:
        pkl.dump(taxon_taxa_lists, f, 2)

    for i, taxon in enumerate(unique_taxon_list):
        unique_taxon_dict[taxon] = taxon_taxa_lists[i]

    list_of_rank_taxa_table_rows = []
    for taxon in filtered_blast_table['taxon'].tolist():
        list_of_rank_taxa_table_rows.append(unique_taxon_dict[taxon])
    rank_taxa_table = pd.DataFrame(list_of_rank_taxa_table_rows, columns = search_ranks)
    augmented_blast_table = pd.concat([filtered_blast_table, rank_taxa_table], axis = 1)

    #if from_postnovo:
    #    filtered_blast_table.set_index('scan_list', inplace=True)
    #else:
    #    filtered_blast_table.set_index('qseqid', inplace=True)

    return augmented_blast_table

def query_entrez_taxonomy_db(taxon, rank_dict, search_ranks_set, one_percent_number_taxa, cores):

    #if current_process()._identity[0] % cores == 1:
    #    global multiprocessing_taxon_count
    #    multiprocessing_taxon_count += 1
    #    if int(multiprocessing_taxon_count % one_percent_number_taxids) == 0:
    #        percent_complete = int(multiprocessing_taxon_count / one_percent_number_taxids)
    #        if percent_complete <= 100:
    #            utils.verbose_print_over_same_line('Entrez taxonomy search progress: ' + str(percent_complete) + '%')

    #print(taxon)
    taxon_ranks_set = set()
    no_response_count = 0
    no_response_count_limit = 5
    while True:
        try:
            taxon_id_list = Entrez.read(Entrez.esearch(db='Taxonomy', term='\"' + taxon + '\"'))['IdList']
            # Favor the lowest taxon id
            # There are occasional taxonomic confusions such as the walking stick genus, Bacillus,
            # which result in multiple returned taxon id's:
            # the lower the taxid number, the more general the rank, the more likely the assignment
            taxon_id = min([int(id) for id in taxon_id_list])
            taxon_info = Entrez.read(Entrez.efetch(db='Taxonomy', id=str(taxon_id)))
            taxon_rank = taxon_info[0]['Rank']
            lineage_info = taxon_info[0]['LineageEx']
            break
        except:
            print(taxon + ': Entrez query no response', flush=True)
            no_response_count += 1
            if no_response_count == no_response_count_limit and '\'' in taxon:
                taxon = taxon.replace('\'', '')
            elif no_response_count > no_response_count_limit:
                taxon_rank = ''
                lineage_info = {}
                print('no response count limit exceeded: ')
                print(taxon)
                break
            time.sleep(30)
    if taxon_rank in search_ranks_set:
        rank_dict[taxon_rank] = taxon
        taxon_ranks_set.add(taxon_rank)
    for entry in lineage_info:
        rank = entry['Rank']
        if rank in search_ranks_set:
            rank_dict[rank] = entry['ScientificName']
            taxon_ranks_set.add(rank)
    for rank in search_ranks_set.difference(taxon_ranks_set):
        rank_dict[rank] = ''

    lineage_list = [level for level in rank_dict.values()]
    if lineage_list == test_list:
        print('empty lineage: ')
        print(taxon)
    return lineage_list

def find_parsimonious_taxonomy(df, from_postnovo):

    if from_postnovo:
        df.set_index('scan_list', inplace=True)
        id_type = 'scan_list'
    else:
        df.set_index('qseqid', inplace=True)
        id_type = 'qseqid'
    list_of_taxa_assignment_rows = []
    for id in df.index.get_level_values(id_type).unique():
        id_table = df.loc[[id]]
        id_table = id_table.drop_duplicates(subset = ['taxon'])
        for rank_index, rank in enumerate(search_ranks):
            most_common_taxon_count = Counter(id_table[rank]).most_common(1)[0]
            if most_common_taxon_count[0] != '' and pd.notnull(most_common_taxon_count[0]):
                if most_common_taxon_count[1] >= taxon_assignment_threshold * len(id_table):
                    id_table.reset_index(inplace = True)
                    try:
                        representative_row = id_table.ix[
                            id_table[
                                id_table[rank] == most_common_taxon_count[0]
                                ][rank].first_valid_index()
                            ]
                    except:
                        print('id: ' + str(id), flush=True)
                        print('rank: ' + str(rank), flush=True)
                        print('is pd.null:')
                        print(pd.isnull(most_common_taxon_count[0]))
                        sys.exit()
                    if from_postnovo:
                        list_of_taxa_assignment_rows.append(
                            [id] + \
                                [representative_row['hit']] + \
                                [representative_row['seq']] + \
                                [representative_row['precursor_mass']] + \
                                [representative_row['seq_score']] + \
                                [representative_row['seq_origin']] + \
                                rank_index * ['N/A'] + \
                                representative_row[rank:].tolist()
                            )
                    else:
                        list_of_taxa_assignment_rows.append(
                            [id] + \
                                [representative_row['hit']] + \
                                [representative_row['seq']] + \
                                rank_index * ['N/A'] + \
                                representative_row[rank:].tolist()
                            )
                    break
        else:
            representative_row = id_table.iloc[0]
            if from_postnovo:
                list_of_taxa_assignment_rows.append(
                    [id] + \
                        [representative_row['hit']] + \
                        [representative_row['seq']] + \
                        [representative_row['precursor_mass']] + \
                        [representative_row['seq_score']] + \
                        [representative_row['seq_origin']] + \
                        len(search_ranks) * ['N/A']
                    )
            else:
                list_of_taxa_assignment_rows.append(
                    [id] + \
                        [representative_row['hit']] + \
                        [representative_row['seq']] + \
                        len(search_ranks) * ['N/A']
                    )

    if from_postnovo:
        taxa_assignment_table = pd.DataFrame(
            list_of_taxa_assignment_rows,
            columns=['scan_list', 'hit', 'seq', 'precursor_mass', 'seq_score', 'seq_origin'] + search_ranks)
    else:
        taxa_assignment_table = pd.DataFrame(
            list_of_taxa_assignment_rows,
            columns=['qseqid', 'hit', 'seq'] + search_ranks)
    #taxa_assignment_table.set_index(id_type, inplace = True)

    taxa_count_table = pd.DataFrame()
    for rank in search_ranks:
        taxa_counts = Counter(taxa_assignment_table[rank])
        taxa_count_table = pd.concat(
            [taxa_count_table,
             pd.Series([taxon for taxon in taxa_counts.keys()], name = rank + ' taxa'),
             pd.Series([count for count in taxa_counts.values()], name = rank + ' counts')],
             axis = 1)

    return taxa_assignment_table, taxa_count_table

if __name__ == '__main__':
    #d = 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test'
    #df = pd.read_csv(os.path.join(d, 'high_prob_df1.csv'), header=0)
    #find_parsimonious_taxonomy(df, True)
    main()