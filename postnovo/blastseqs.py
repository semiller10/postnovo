import argparse
import os
import stat
import pkg_resources
import subprocess

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
    
    ## FOR DEBUGGING PURPOSES: REMOVE
    blast_batch_pathname = '/home/samuelmiller/5-9-17/blast_batch.sh'
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
    os.chmod(temp_blast_batch_pathname, 0o555)
    subprocess.call([temp_blast_batch_pathname])

    return

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

    #merged_blast_output = merge_blast_output(
    #    run_blast(
    #        split_fasta(args.fasta_path, args.cores, args.max_seqs_per_process
    #                    ),
    #        args.blastp_path, args.db_dir, args.cores
    #        )
    #    )

    #fasta_ref_path = 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\test\\human.faa'
    #target_confidence_level = 0.95
    #cores = 3
    #print('Minimum sequence length required = ' +
    #      str(find_min_seq_len(fasta_ref_path = fasta_ref_path, target_confidence_level = target_confidence_level, cores = cores)))

    fasta_path = '/home/samuelmiller/5-9-17/test.faa'
    cores = 2
    max_seqs_per_process = 1
    blastp_path = '/home/samuelmiller/ncbi-blast-2.6.0+/bin/blastp'
    db_dir = '/home/samuelmiller/refseq_protein/refseq_protein'

    split_fasta_pathname_list = split_fasta(fasta_path, cores, max_seqs_per_process)
    run_blast(split_fasta_pathname_list, blastp_path, db_dir, cores)