import argparse
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

def main():

    args = parse_args()
    args = make_fasta(args)

    split_fasta_pathname_list = split_fasta(args.faa_fp, args.cores, args.max_seqs_per_process)
    run_blast(split_fasta_pathname_list, args.blastp_fp, args.db_fp, args.cores)

    last_file_number = int(split_fasta_pathname_list[-1].split('.faa')[0].split('_')[-1])
    file_prefix = split_fasta_pathname_list[0].split('_1.faa')[0]
    xml_files = [file_prefix + '_' + str(i) + '.out' for i in range(1, last_file_number + 1)]
    xml_out = file_prefix + '.merged.xml'
    # write the full xml output of the BLAST search for use in BLAST2GO as needed
    merge_xml(xml_files, xml_out)

    # convert xml to tabular format
    multiprocessing_pool = Pool(args.cores)
    blast_table_df_list = multiprocessing_pool.map(xml_to_tabular, xml_files)
    multiprocessing_pool.close()
    multiprocessing_pool.join()
    merged_blast_table = pd.concat(blast_table_df_list, axis=0)

    parsed_blast_table = parse_blast_table(args.from_postnovo, args.faa_fp, merged_blast_table)
    filtered_blast_table = filter_blast_table(parsed_blast_table, args.from_postnovo)
    with open('/home/samuelmiller/metagenome_vs_postnovo/toolik_2_2_filtered_blast_table.pkl', 'wb') as f:
        pkl.dump(filtered_blast_table, f, 2)
    import sys
    sys.exit()
    #with open('C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\filtered_blast_table.pkl', 'rb') as f:
    #    filtered_blast_table = pkl.load(f)
    augmented_blast_table = retrieve_taxonomy(filtered_blast_table, args.cores, args.from_postnovo)
    taxa_assignment_table, taxa_count_table = \
        find_parsimonious_taxonomy(augmented_blast_table, args.from_postnovo)

    augmented_blast_table.to_csv(
        os.path.join(os.path.dirname(args.faa_fp),
                     os.path.splitext(os.path.basename(args.faa_fp))[0] + '_augmented_blast_table.tsv'),
        sep='\t', header=True)
    taxa_assignment_table.to_csv(
        os.path.join(os.path.dirname(args.faa_fp), 
                     os.path.splitext(os.path.basename(args.faa_fp))[0] + '_taxa_assignment_table.tsv'),
        sep='\t', header=True)
    taxa_count_table.to_csv(
        os.path.join(os.path.dirname(args.faa_fp),
                     os.path.splitext(os.path.basename(args.faa_fp))[0] + '_taxa_count_table.tsv'),
        sep='\t', header=True)

    #augmented_blast_table = pd.read_csv(
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs_augmented_blast_table.tsv',
    #    sep='\t',
    #    header=0
    #    )
    #augmented_blast_table = pd.read_csv(
    #    '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_augmented_blast_table.tsv',
    #    sep='\t',
    #    header=0
    #    )
    #taxa_assignment_table = pd.read_csv(
    #    'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\042017_toolik_core_2_2_1_1_sem.ERR1022687_taxa_assignment_table.tsv',
    #    sep='\t',
    #    header=0
    #    )

    # Write a fasta file containing the full subject sequences of the top hit
    full_hit_seq_fasta_path = os.path.join(
        os.path.dirname(args.faa_fp),
        os.path.splitext(os.path.basename(args.faa_fp))[0] + '_full_hit_seq.faa')
    make_full_hit_seq_fasta(augmented_blast_table, taxa_assignment_table, full_hit_seq_fasta_path, args.from_postnovo, args.cores)

    sys.exit(0)



    # Plan to parse BLAST table
    # Create taxonomically annotated table
    # Add a hit col to df
    scan_list = augmented_blast_table['scan'].tolist()
    last_scan = scan_list[0]
    hit_list = [0]
    for scan in scan_list[1:]:
        if scan != last_scan:
            last_scan = scan
            hit_list.append(0)
        else:
            hit_list.append(hit_list[-1] + 1)
    augmented_blast_table['hit'] = hit_list

    # Draw up to 10 hits from each scan group
    # Evenly sample scan groups over 10, starting with hit 0
    scan_groups = augmented_blast_table.groupby('scan')
    sampled_table = scan_groups.apply(sample_hits)
    # Make fasta file for eggnog-mapper
    # Header: >(scan)scan(hit)hit number
    # Seq: full subject seq for each hit
    # Sort into fasta files based on superkingdom (up to 3 files total)
    eggnog_mapper_first_faa = os.path.join(
        os.path.dirname(args.faa_fp),
        os.path.splitext(os.path.basename(args.faa_fp))[0] + '_eggnog_mapper_first_round.faa')
    #eggnog_fasta_path_list = make_full_hit_seq_fasta1(sampled_table, eggnog_mapper_first_faa, args.cores)

    eggnog_fasta_path_list = [
        '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.archaea_0.faa',
        '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.bacteria_0.faa',
        '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.bacteria_1.faa',
        '/home/samuelmiller/6-23-17/postnovo_test/postnovo_seqs_eggnog_mapper_first_round.eukaryota_0.faa'
        ]

    # Run eggnog-mapper on each file using HMMER
    # Download annotations
    print('Run eggnog-mapper with the fasta files.')
    #input('Press enter to continue once you have placed the eggnog-mapper output in the fasta directory.')
    # Load as dataframe 
    # Assign predefined column names
    # Concat (up to 3) annotation dfs
    eggnog_output_path_list = [faa + '.emapper.annotations' for faa in eggnog_fasta_path_list]
    eggnog_first_round_df = pd.DataFrame(columns=eggnog_output_headers)
    for output_path in eggnog_output_path_list:
        eggnog_output_df = pd.read_csv(output_path, sep='\t', header=None, names=eggnog_output_headers)
        eggnog_first_round_df = pd.concat([eggnog_first_round_df, eggnog_output_df], axis=0)
    # Split header into two cols for scans and hits
    query_list = eggnog_first_round_df['query'].tolist()
    query_list = [query.split('(scan)')[1] for query in query_list]
    temp_list_of_lists = [query.split('(hit)') for query in query_list]
    scan_list = [temp_list[0] for temp_list in temp_list_of_lists]
    hit_list = [temp_list[1] for temp_list in temp_list_of_lists]
    eggnog_first_round_df.drop('query', axis=1, inplace=True)
    eggnog_first_round_df['scan'] = scan_list
    eggnog_first_round_df['hit'] = hit_list

    # Loop through each scan group
    scan_set_list = list(set(scan_list))
    conserv_func_df_list = []
    for scan in scan_set_list:
        scan_df = eggnog_first_round_df[eggnog_first_round_df['scan'] == scan]
        # Are all the eggnog descriptions the same?
        if (scan_df['eggnog hmm desc'] == scan_df['eggnog hmm desc'].iloc[0]).all():
            # If true, the group is functionally conserved
            conserv_func_df_list.append(scan_df)
        # If false, make a list of each COG category, splitting by comma (e.g., 'C, T')
        else:
            cog_cat_list
    # Loop through each COG category in the list
    # Is the category found in each entry?
    # If so, break and the group is maintained
    # If there is functional conservation, place the group into df1, else df2
    # Merge with taxonomically annotated table to attach taxonomic info to df1 results
    # The usefulness of the results in df2 is unknown as yet
    # Recover groups from taxonomic df corresponding to groups in df2
    # Place these groups in df2.1
    # Analyze the groups in df2.1 based on taxonomic profile of taxonomically conserved groups of df1
    # Find scan groups in df1 that have the same taxa at the species, genus or family level
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

def make_full_hit_seq_fasta1(df, write_path, cores):
    
    gi_list = df['sallseqid'].tolist()
    multiprocessing_pool = Pool(cores)
    full_hit_seq_list = multiprocessing_pool.map(query_ncbi_protein, gi_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    df['scan'] = df['scan'].apply(str)
    df['hit'] = df['hit'].apply(str)
    df['full seq'] = full_hit_seq_list

    eggnog_fasta_path_list = []
    for superkingdom in superkingdoms:
        superkingdom_df = df[df['superkingdom'] == superkingdom]
        superkingdom_scan_list = superkingdom_df['scan'].tolist()
        superkingdom_hit_list = superkingdom_df['hit'].tolist()
        superkingdom_seq_list = superkingdom_df['full seq'].tolist()
        superkingdom_header_list = [
            '>' + '(scan)' + superkingdom_scan_list[i] + '(hit)' + superkingdom_hit_list[i]
            for i in range(len(superkingdom_scan_list))
            ]
        superkingdom_header_seq_dict = {
            superkingdom_header_list[i]: superkingdom_seq_list[i]
            for i in range(len(superkingdom_header_list))
            }

        for i in range(len(superkingdom_header_list) // hmmer_seq_count_limit + 1):
            superkingdom_write_path = write_path.replace('.faa', '.' + superkingdom.lower() + '_' + str(i) + '.faa')
            eggnog_fasta_path_list.append(superkingdom_write_path)
            with open(superkingdom_write_path, 'w') as f:
                for j, header in enumerate(superkingdom_header_list):
                    f.write(header + '\n')
                    f.write(superkingdom_header_seq_dict[header] + '\n')
                    if j + 1 == hmmer_seq_count_limit * (i + 1):
                        break
    return eggnog_fasta_path_list
    
def make_full_hit_seq_fasta(augmented_blast_table, taxa_assignment_table, write_path, from_postnovo, cores):

    # augmented_blast_table groupby scan (qseqid) index
    # make df of first row of each group
    # make list of scans (qseqids) from df index
    # make list of gi's from df sallseqid col
    # recover full seq from Entrez using gi
    # make dict with scan keys and full seqs
    # taxa_assignment_table extract rows with NaN superkingdom
    # make list of scans (qseqids) from df index
    # for each superkingdom
    # taxa_assignment_table extract rows for superkingdom
    # make list of scans (qseqids) from df index
    # open fasta file to write
    # for each scan (qseqid) in superkingdom list and null list
    # write scan (qseqid) as header
    # write seq recovered from dict


    if from_postnovo:
        index_label = 'scan'
    else:
        index_label = 'qseqid'
    augmented_blast_table.reset_index(inplace=True)
    first_hits_df = augmented_blast_table.groupby(index_label).first()

    first_hits_df.reset_index(inplace=True)
    full_header_list = first_hits_df[index_label].apply(str).tolist()
    full_gi_list = first_hits_df['sallseqid'].tolist()

    #full_hit_seq_list = []
    #for gi in gi_list:
    #    full_hit_seq_list.append(query_ncbi_protein(gi))

    multiprocessing_pool = Pool(cores)
    full_hit_seq_list = multiprocessing_pool.map(query_ncbi_protein, full_gi_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    #with open('C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\full_hit_seq_list.pkl', 'wb') as f:
    #    pkl.dump(full_hit_seq_list, f, 2)
    #with open('C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\full_hit_seq_list.pkl', 'rb') as f:
    #    full_hit_seq_list = pkl.load(f)

    header_seq_dict = {full_header_list[i]: full_hit_seq_list[i] for i in range(len(full_header_list))}

    taxa_assignment_table.reset_index(inplace=True)
    null_superkingdom_df = taxa_assignment_table[pd.isnull(taxa_assignment_table['superkingdom'])]
    null_superkingdom_header_list = null_superkingdom_df[index_label].apply(str).tolist()

    for superkingdom in superkingdoms:
        superkingdom_df = taxa_assignment_table[
            taxa_assignment_table['superkingdom'] == superkingdom]
        superkingdom_header_list = superkingdom_df[index_label].apply(str).tolist()

        superkingdom_write_path = write_path.replace('.faa', '.' + superkingdom.lower() + '.faa')
        with open(superkingdom_write_path, 'w') as f:
            for header in superkingdom_header_list:
                f.write('>' + header + '\n')
                f.write(header_seq_dict[header] + '\n')
            for header in null_superkingdom_header_list:
                f.write('>' + header + '\n')
                f.write(header_seq_dict[header] + '\n')

def query_ncbi_protein(gi):

    while True:
        try:
            full_seq = Entrez.read(Entrez.efetch(db='Protein', id=gi, retmode='xml'))[0]['GBSeq_sequence']
            break
        except:
            print(gi + ': Entrez query no response')
            time.sleep(30)

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
                        action='store_true',
                        help='seqs were generated by postnovo')
    seq_input_group = parser.add_mutually_exclusive_group()
    seq_input_group.add_argument('--faa_fp',
                                 default='C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\blast_seqs_test\\postnovo_seqs.faa',
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

    postnovo_merged_headers = ['scan_list', 'xle permutation', 'precursor_mass', 'score', 'seq_origin'] + raw_blast_table_headers[1:]

    if from_postnovo:
        qseqid_list = raw_blast_table['qseqid'].tolist()
        # qseqid format is, ex., (scan_list)1,2(xle_permutation)0(precursor_mass)1000.000(score)0.55(seq_origin)postnovo
        qseqid_list = [qseqid.split('(scan_list)')[1] for qseqid in qseqid_list]
        temp_list_of_lists = [qseqid.split('(xle_permutation)') for qseqid in qseqid_list]
        scan_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(precursor_mass)') for temp_list in temp_list_of_lists]
        permut_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(score)') for temp_list in temp_list_of_lists]
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

    if from_postnovo:
        parsed_blast_table.set_index(['scan_list', 'xle permutation', 'precursor_mass', 'score', 'seq_origin'], inplace = True)
    else:
        parsed_blast_table.set_index(['qseqid'], inplace = True)

    seq_table = tabulate_fasta(from_postnovo, faa_fp)
    merged_blast_table = seq_table.join(parsed_blast_table)

    return merged_blast_table

def tabulate_fasta(from_postnovo, faa_fp):

    raw_fasta_input = pd.read_table(faa_fp, header = None)
    fasta_headers_list = raw_fasta_input.ix[::2, 0].tolist()
    seq_col = raw_fasta_input.ix[1::2, 0]
    seq_col.index = range(len(seq_col))

    if from_postnovo:
        # header format is, ex., >(scan_list)1,2(xle_permutation)0(precursor_mass)1000.000(score)0.55(seq_origin)postnovo
        fasta_headers_list = [fasta_header.strip('>(scan_list)') for fasta_header in fasta_headers_list]
        temp_list_of_lists = [qseqid.split('(xle_permutation)') for qseqid in qseqid_list]
        scan_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        temp_list_of_lists = [temp_list[1].split('(precursor_mass)') for temp_list in temp_list_of_lists]
        permut_col = pd.Series(
            [temp_list[0] for temp_list in temp_list_of_lists])
        seq_table = pd.concat([scan_col, permut_col, seq_col], axis = 1)
        seq_table.columns = ['scan_list', 'xle permutation', 'seq']
        seq_table.set_index(['scan', 'xle permutation'], inplace = True)
    else:
        id_col = pd.Series([fasta_header.strip('>') for fasta_header in fasta_headers_list])
        seq_table = pd.concat([id_col, seq_col], axis=1)
        seq_table.columns = ['qseqid', 'seq']
        seq_table.set_index(['qseqid'], inplace=True)
    seq_table['len'] = seq_table['seq'].apply(lambda seq: len(seq))

    return seq_table

def filter_blast_table(blast_table, from_postnovo):

    # Example to explain calculation of deletion count and nonidentical residues:
    # One deletion, one insertion, one substitution
    # pident = 26/29 * 100
    # Query  1    SRRTKGNNPVLIGEP-VGKTAIVDGLAQK  28
    #             SRRTK NNPVLIGEP VGKTAIV+GLAQK
    # Sbjct  221  SRRTK-NNPVLIGEPGVGKTAIVEGLAQK  248
    # dels = (gaps - (qend - qstart) - (send - sstart)) / 2
    #      = (2    - (28   - 1)      - (248  - 221))    / 2 = 1
    # nonident = len - (qend - qstart + 1 + del) * pident / 100
    #          = 28  - (28   - 1      + 1 + 1)   * 26     / 29 = 2
    # This implies that 2 residues (at the insertion and substitution sites)
    # are nonidentical in the query seq

    blast_table[['gaps', 'qend', 'qstart', 'send', 'sstart']] = \
        blast_table[['gaps', 'qend', 'qstart', 'send', 'sstart']].applymap(int)
    blast_table['pident'] = blast_table['pident'].apply(float)
    blast_table['del'] = (blast_table['gaps'] -
                          ((blast_table['qend'] - blast_table['qstart']) - 
                          (blast_table['send'] - blast_table['sstart']))) / 2
    blast_table['nonident'] = \
        round(
            blast_table['len'] \
                - (blast_table['qend'] - blast_table['qstart'] + 1 + blast_table['del']) \
                * (blast_table['pident'] / 100),
            0)

    if from_postnovo:
        #blast_table['postnovo score'] = blast_table['postnovo score'].apply(float)
        #blast_table['score penalty'] = blast_table['postnovo score'].apply(
        #    lambda score: postnovo_score_penalties[int(score / 0.2)])
        #filtered_blast_table = blast_table[
        #    blast_table['len'] - blast_table['nonident'] - blast_table['score penalty'] >= 9]
        filtered_blast_table = blast_table
        scan_groups = filtered_blast_table.groupby(level='scan_list', group_keys=False)
        filtered_blast_table = scan_groups.apply(
            lambda g: g[g['nonident'] == g['nonident'].min()])
        filtered_blast_table.index = filtered_blast_table.index.droplevel('xle permutation')
    else:
        #filtered_blast_table = blast_table[
        #    blast_table['len'] - blast_table['nonident'] >= 9]
        filtered_blast_table = blast_table
        qseqid_groups = filtered_blast_table.groupby(filtered_blast_table.index, group_keys=False)
        filtered_blast_table = qseqid_groups.apply(
            lambda g: g[g['nonident'] == g['nonident'].min()])

    filtered_blast_table['taxon'] = filtered_blast_table['salltitles'].apply(
        lambda x: extract_taxon(x))

    return filtered_blast_table

def extract_taxon(salltitle):
    if '<>' in salltitle:
        # Example 1
        # '[Salmonella] <> [a]'
        # Example 2
        # '[a]b[Proteobacteria] <> [d]'
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
    for i, taxon in enumerate(unique_taxon_list):
        unique_taxon_dict[taxon] = taxon_taxa_lists[i]

    list_of_rank_taxa_table_rows = []
    for taxon in filtered_blast_table['taxon']:
        list_of_rank_taxa_table_rows.append(unique_taxon_dict[taxon])
    rank_taxa_table = pd.DataFrame(list_of_rank_taxa_table_rows, columns = search_ranks)
    filtered_blast_table.reset_index(inplace=True)
    filtered_blast_table = pd.concat([filtered_blast_table, rank_taxa_table], axis = 1)
    if from_postnovo:
        filtered_blast_table.set_index('scan_list', inplace=True)
    else:
        filtered_blast_table.set_index('qseqid', inplace=True)

    return filtered_blast_table

def query_entrez_taxonomy_db(taxon, rank_dict, search_ranks_set, one_percent_number_taxa, cores):

    #if current_process()._identity[0] % cores == 1:
    #    global multiprocessing_taxon_count
    #    multiprocessing_taxon_count += 1
    #    if int(multiprocessing_taxon_count % one_percent_number_taxids) == 0:
    #        percent_complete = int(multiprocessing_taxon_count / one_percent_number_taxids)
    #        if percent_complete <= 100:
    #            utils.verbose_print_over_same_line('Entrez taxonomy search progress: ' + str(percent_complete) + '%')

    print(taxon)
    taxon_ranks_set = set()
    no_response_count = 0
    no_response_count_limit = 5
    while True:
        try:
            taxon_id_list = Entrez.read(Entrez.esearch(db='Taxonomy', term='\"' + taxon + '\"'))['IdList']
            # Favor the lowest taxon id
            # There are occasional taxonomic confusions such as the walking stick genus, Bacillus,
            # which result in multiple returned taxon id's:
            # the lower the number, the more general the rank, the more likely the assignment
            taxon_id = min([int(id) for id in taxon_id_list])
            taxon_info = Entrez.read(Entrez.efetch(db='Taxonomy', id=str(taxon_id)))
            taxon_rank = taxon_info[0]['Rank']
            lineage_info = taxon_info[0]['LineageEx']
            break
        except:
            print(taxon + ': Entrez query no response')
            no_response_count += 1
            if no_response_count == no_response_count_limit and '\'' in taxon:
                taxon = taxon.replace('\'', '')
            elif no_response_count > no_response_count_limit:
                taxon_rank = ''
                lineage_info = {}
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

    return [taxon for taxon in rank_dict.values()]

def find_parsimonious_taxonomy(augmented_blast_table, from_postnovo):

    if from_postnovo:
        id_type = 'scan'
    else:
        id_type = 'qseqid'
    list_of_taxa_assignment_rows = []
    for id in augmented_blast_table.index.get_level_values(id_type).unique():
        id_table = augmented_blast_table.loc[[id]]
        id_table = id_table.drop_duplicates(subset = ['taxon'])
        for rank_index, rank in enumerate(search_ranks):
            most_common_taxon_count = Counter(id_table[rank]).most_common(1)[0]
            if most_common_taxon_count[0] != '':
                if most_common_taxon_count[1] >= taxon_assignment_threshold * len(id_table):
                    id_table.reset_index(inplace = True)
                    representative_row = id_table.ix[
                        id_table[id_table[rank] == \
                            most_common_taxon_count[0]][rank].first_valid_index()
                        ]
                    if from_postnovo:
                        list_of_taxa_assignment_rows.append(
                            [id] + \
                                [representative_row['seq']] + \
                                [representative_row['mass']] + \
                                rank_index * ['N/A'] + \
                                representative_row[rank:].tolist()
                            )
                    else:
                        list_of_taxa_assignment_rows.append(
                            [id] + \
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
                        [representative_row['seq']] + \
                        [representative_row['mass']] + \
                        len(search_ranks) * ['N/A']
                    )
            else:
                list_of_taxa_assignment_rows.append(
                    [id] + \
                        [representative_row['seq']] + \
                        len(search_ranks) * ['N/A']
                    )

    if from_postnovo:
        taxa_assignment_table = pd.DataFrame(
            list_of_taxa_assignment_rows,
            columns=['scan', 'seq', 'mass'] + search_ranks)
    else:
        taxa_assignment_table = pd.DataFrame(
            list_of_taxa_assignment_rows,
            columns=['qseqid', 'seq'] + search_ranks)
    taxa_assignment_table.set_index(id_type, inplace = True)

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
    main()