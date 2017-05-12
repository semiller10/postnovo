import argparse
import os.path
import pkg_resources

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

def blast(fasta_pathname_list, blastp_path, db_dir, cores, max_seqs_per_process):
    
    with open(pkg_resources.resource_filename('postnovo', 'blast_batch.sh'), 'r') as blast_batch_template_file:
        blast_batch_template = blast_batch_template_file.read()
        temp_blast_batch = blast_batch_template.replace('FASTA_FILES=', 'FASTA_FILES=({})'.format(' '.join(fasta_pathname_list)))


    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Set up BLAST+ search for de novo sequences')
    parser.add_argument('--fasta_path',
                        help = 'fasta input filepath')
    parser.add_argument('--blastp_path',
                        help = 'blastp filepath')
    parser.add_argument('--db_dir',
                        help = 'directory containing BLAST database')
    parser.add_argument('--cores', default = 1, type = int,
                        help = 'number of cores to use')
    parser.add_argument('--max_seqs_per_process', default = 1000, type = int,
                        help = 'maximum number of query seqs per BLAST+ instance')
    args = parser.parse_args()

    blast(split_fasta(args.fasta_path, args.cores, args.max_seqs_per_process), args.blastp_path, args.db_dir, args.cores, args.max_seqs_per_process)

    #fasta_ref_path = 'C:\\Users\\Samuel\\Documents\\Visual Studio 2015\\Projects\\postnovo\\test\\human.faa'
    #target_confidence_level = 0.95
    #cores = 3
    #print('Minimum sequence length required = ' +
    #      str(find_min_seq_len(fasta_ref_path = fasta_ref_path, target_confidence_level = target_confidence_level, cores = cores)))