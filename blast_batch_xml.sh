#!/bin/bash

FASTA_FILES=
MAX_PROCESSES=
BLASTP_PATH=
DB_DIR=

process_ids=( $(pgrep -u $(id -u -n) blastp) )

for fasta_file in ${FASTA_FILES[*]}
do
	# While the last possible process is running (non-zero length string for last possible process)
	while [ -n "${process_ids[$MAX_PROCESSES]}" ]
	# Search for the list of process ids (infinite loop)
	do
		process_ids=( $(pgrep -u $(id -u -n) blastp) )
		sleep 5
	done

	# When a process can be added, prepare a new BLAST+ instance
	out_file="$(dirname $fasta_file)/$(basename $fasta_file .faa).out"

	nohup $BLASTP_PATH -task blastp-short -db $DB_DIR -query $fasta_file -out $out_file -evalue 1000000 -max_target_seqs 500 -max_hsps 1 -comp_based_stats 0 -outfmt 5 &
	process_ids=( $(pgrep -u $(id -u -n) blastp) )
done

while [ -n "${process_ids[*]}" ]
do
	process_ids=( $(pgrep -u $(id -u -n) blastp) )
	sleep 5
done