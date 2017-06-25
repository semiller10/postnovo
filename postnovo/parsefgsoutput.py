import argparse
import os.path

def main():
    args = get_args()
    parse_fgs_output(args)

def parse_fgs_output(args):
    output_fp = os.path.splitext(args.fgs_faa_fp)[0] + '.pep_list.faa'
    min_len = args.min_len
    with open(args.fgs_faa_fp) as in_f, open(output_fp, 'w') as out_f:
        for line in in_f:
            if line.startswith('>'):
                header = line.strip('\n')
            else:
                seqs = line.split('*')
                long_seqs = [seq.strip('\n') for seq in seqs if len(seq.strip('\n')) >= min_len]
                for i, long_seq in enumerate(long_seqs):
                    out_f.write(header + '_seq' + str(i) + '\n')
                    out_f.write(long_seq + '\n')

def get_args():
    parser = argparse.ArgumentParser(
        description='Parse FragGeneScan output to create a faa file with seqs >= user-defined length'
        )

    parser.add_argument(
        '-f',
        '--fgs_faa_fp',
        help='FGS faa output filepath'
        )
    parser.add_argument(
        '-l',
        '--min_len',
        type = int,
        help='min length of translated gene fragment to preserve'
        )

    return parser.parse_args()

if __name__ == '__main__':
    main()