# postnovo
Post-processing peptide de novo sequences to improve their accuracy

## Requirements
Python 3.4 or higher

### Training model: a pickled dictionary of random forests
#### A file named *forest_dict.pkl* must go in the *postnovo/training* directory.
#### The default training model will be downloaded from the following site if *forest_dict.pkl* is not present (e.g., on first run):
<http://www.mediafire.com>
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
`python postnovo.py <--frag_mass_tols> <--other options>`

#### Output goes to *postnovo/output* directory.

### Four modes
Predict (DEFAULT) = post-process de novo sequences for data WITHOUT reference

Test = post-process de novo sequences for data WITH reference

Train = update postnovo model with de novo sequences and reference

Optimize = same as train, but some random forest parameters are tuned

### Command line options
#### postnovo I/O directory: always required, all input files beside mgf file (if used) must be placed here
`<--iodir "/home/postnovo_io">`

#### Choosing any of these flags overrides default predict mode.
`[--test]`

`[--train]`

`[--optimize]`
#### Fragment mass tolerance(s) of input files: always required
`<--frag_mass_tols "0.3, 0.5">`

#### *Novor* and *PepNovo+* output files corresponding to fragment mass tolerance(s)
##### Use of *DeNovoGUI* on an mgf file (see below) overrides these options.

`[--novor_files "novor_output_0.3.novor.csv, novor_output_0.5.novor.csv"]`

`[--pn_files "pn_output_0.3.mgf.out, pn_output_0.5.mgf.out"]`

#### Options for generating de novo output files with *DeNovoGUI* and automatically using as postnovo input
##### These override --novor_files and --pn_files.
##### The full mgf file path should be used (no need to place the file in the IO directory).
`[--denovogui_path "/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar"]`

`[--denovogui_mgf_path "/home/ms_files/spectra.mgf"]`

#### A tab-delimited .txt reference file is required in test, train and optimize modes.
##### Sequences with FDRs up to 0.05 (default medium confidence in *Proteome Discoverer*) should be retained in the reference file.
##### This file can be the *unmodified* exported tab-delimited file from the *Proteome Discoverer* consensus workflow PSM results sheet.
##### The required columns of a reference file that does not come from *Proteome Discoverer* are, in order, 1. scan number, 2. sequence (with symbols beside letters for canonical amino acids removed) and 3. database search false detection rate (e.g., *Percolator* q-value).
`[--db_search_ref_file "proteome_discoverer_psm_table.txt"]`

#### A protein fasta reference file is also required in test, train and optimize modes.
##### This should be the file used by the database search algorithm in generating the database search reference file.
##### The purpose of this reference is to find correct de novo sequences that are not identified by the database search algorithm.
`[--fasta_ref_file "fasta.faa"]`

#### Cores used by postnovo and *DeNovoGUI*: default of 1, but *multiple cores are intended to be used*
`[--cores 16]`

#### Minimum length and probability of sequences reported by postnovo: default of 6 and 0.5, respectively
`[--min_len 9]`

`[--min_prob 0.75]`

#### Turn off verbose mode
`[--quiet]`

#### Usage help
`[--help]`

#### To use a json parameter file instead of command line arguments: this option excludes other options beside iodir
`[--param_file "param.json"]`

### Example command (default predict mode)
`postnovo --frag_mass_tols "0.2, 0.3, 0.4, 0.5, 0.6, 0.7" --denovogui_path "/home/DeNovoGUI-1.15.5/DeNovoGUI-1.15.5.jar" --denovogui_mgf_path "/home/ms_files/spectra.mgf" --cores 8`

### Parameter file substitute for command line arguments

A parameter file template, param_template.json, is found at the following site:
<http://www.mediafire.com>

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
