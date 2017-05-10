# postnovo
Post-processing peptide de novo sequences to improve their accuracy

## Requirements
Python 3.4 or higher

### Training model: a pickled dictionary of random forests
#### A file named *forest_dict.pkl* must go in the *postnovo/training* directory.
#### The default *forest_dict.pkl* file can be downloaded here:
<http://bit.ly/2po5wRt>
#### The user can generate or add to *forest_dict.pkl* by using the train or optimize modes of postnovo.
All changes will be made to a file named *forest_dict.pkl*.

See `--ref_file` description below

## Installation
`pip install postnovo`

**or**

Download the postnovo source and run the setup script from the postnovo directory.

1. `python setup.py build`
2. `python setup.py install`

## Usage
`python postnovo.py <--iodir> <--other options>`

#### Output goes to *postnovo/output* directory.

### Four modes
Predict (DEFAULT) = post-process de novo sequences for data WITHOUT reference

Test = post-process de novo sequences for data WITH reference

Train = update postnovo model with de novo sequences and reference

Optimize = same as train, but some random forest parameters are tuned

### Command line options
#### postnovo I/O directory: always required, all input files beside mgf file (if used) must be stored here
`<--iodir "/home/postnovo_io">`

#### Choosing any of these flags overrides default predict mode.
`[--test]`

`[--train]`

`[--optimize]`

#### Options for generating de novo output files from mgf file with *DeNovoGUI* and automatically using as postnovo input
##### Overrides --novor_files and --pn_files (see below)
##### The full mgf file path should be used (no need to place the file in the IO directory).
`[--denovogui_path "/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar"]`

`[--denovogui_mgf_path "/home/ms_files/spectra.mgf"]`

#### Options for using as postnovo input the *Novor* and *PepNovo+* output files corresponding to 7 fragment mass tolerances
##### Files must be in order of fragment mass tolerance from lowest to highest.
##### Use of *DeNovoGUI* on an mgf file overrides these options (see --denovogui_path and --denovogui_mgf_path above).

`[--novor_files "novor_output_0.2.novor.csv, novor_output_0.3.novor.csv, novor_output_0.4.novor.csv, novor_output_0.5.novor.csv, novor_output_0.6.novor.csv, novor_output_0.7.novor.csv"]`

`[--pn_files "pn_output_0.2.mgf.out, pn_output_0.3.mgf.out, pn_output_0.4.mgf.out, pn_output_0.5.mgf.out, pn_output_0.6.mgf.out, pn_output_0.7.mgf.out"]`

#### Cores used by postnovo and *DeNovoGUI*: default of 1, but *multiple cores are intended to be used*
`[--cores 16]`

#### Minimum length and probability of sequences reported by postnovo: defaults of 7 and 0.5, respectively
`[--min_len 9]`

`[--min_prob 0.75]`

#### A tab-delimited .txt reference file is required in test, train and optimize modes of postnovo.
##### Sequences with FDRs up to 0.05 (default medium confidence in *Proteome Discoverer*) should be retained in the reference file.
##### This file can be the *unmodified* exported tab-delimited file from the *Proteome Discoverer* consensus workflow PSM results sheet.
##### If the reference input file does not come from *Proteome Discoverer*, then the required columns of the file are, in order, 1. scan number, 2. sequence (with symbols beside letters for canonical amino acids removed) and 3. database search false discovery rate (e.g., *Percolator* q-value).
`[--db_search_ref_file "proteome_discoverer_psm_table.txt"]`

#### A protein fasta reference file is also required in test, train and optimize modes.
##### This should be the file used by the database search algorithm to generate the database search reference file.
##### The purpose of this fasta reference is to find correct de novo sequences that are not identified by the database search algorithm.
`[--fasta_ref_file "fasta.faa"]`

#### Turn off verbose mode
`[--quiet]`

#### Usage help
`[--help]`

#### Use a json parameter file instead of command line arguments
##### The use of a parameter file can streamline the input of postnovo arguments.
##### This option excludes all other command line options beside --iodir.
##### The param file must be stored in iodir.
`[--param_file "param.json"]`

### Example command (default predict mode)
`postnovo --denovogui_path "/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar" --denovogui_mgf_path "/home/ms_files/spectra.mgf" --cores 8`

### Parameter file substitute for command line arguments

A parameter file template, param_template.json, can be downloaded:
<http://bit.ly/2qhhvgP>

This file lists all possible options with default arguments.

Many of these options are mutually exclusive (see above), so it must be modified to mirror a set of possible command line arguments.

#### Example valid param file
"--iodir" = "/home/postnovo_io",

"--test" = true,

"--frag_mass_tols" = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7"],

"--denovogui_path" = "/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar",

"--denovogui_mgf_path" = "/home/ms_files/spectra.mgf",

"--db_search_ref_file" = "proteome_discoverer_psm_table.txt",

"--fasta_ref_file" = "fasta.faa",

"--cores" = 8,

"--min_len" = 9,

"--min_prob" = 0.75
