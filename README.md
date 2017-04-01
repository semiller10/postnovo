# postnovo
Post-processing peptide de novo sequences to improve their accuracy

## Requirements
Python 3.4 or higher

### Training model: a pickled dictionary of random forests
#### A file named *forest_dict.pkl* must go in the *postnovo/training* directory
#### The default training model will be downloaded from the following site if *forest_dict.pkl* is not present (e.g., on first run)
link
#### The user can generate or add to *forest_dict.pkl* by using the train or optimize modes of postnovo
All changes will be made to a file named *forest_dict.pkl*

See `--ref_file` description below

## Installation
`pip install postnovo`

**or**

Download the postnovo source and run the setup script from the postnovo directory

1. `python setup.py build`
2. `python setup.py install`

## Usage
`python postnovo.py <--frag_mass_tols> <--other options>`

#### Output goes to *postnovo/output* directory

### Four modes
Predict (DEFAULT) = post-process de novo sequences for data WITHOUT reference

Test = post-process de novo sequences for data WITH reference

Train = update postnovo model with de novo sequences and reference

Optimize = same as train, but some random forest parameters are tuned

### Command line options
#### Choosing any of these flags overrides default predict mode
`[--test]`

`[--train]`

`[--optimize]`
#### Fragment mass tolerance(s) of input files: always required
`<--frag_mass_tols "0.3, 0.5">`

#### Novor and PepNovo+ output files corresponding to fragment mass tolerance(s)
#### Output files should be placed in *postnovo/userfiles* directory
#### Use of *DeNovoGUI* (see below) overrides these options

`[--novor_files "novor_output_0.3.novor.csv, novor_output_0.5.novor.csv"]`

`[--pn_files "pn_output_0.3.mgf.out, pn_output_0.5.mgf.out"]`

#### Options for generating de novo output files with *DeNovoGUI* and automatically using as postnovo input
##### These override --novor_files and --pn_files
`[--denovogui_path "C:\Program Files (x86)\DeNovoGUI-1.15.5-windows\DeNovoGUI-1.15.5\DeNovoGUI-1.15.5.jar"]`

`[--denovogui_mgf_path "C:\Documents\mgf_files\spectra.mgf"]`

#### A tab-delimited .txt reference file is required in test, train and optimize modes
##### This file can be the exported tab-delimited file from the Proteome Discoverer consensus workflow PSM results sheet
##### The required columns of a reference file are 1. scan number, 2. sequence (with non-alphabetical symbols removed) and 3. database search false detection rate (e.g., Percolator q-value)
##### If the reference file is not generated via Proteome Discoverer, the order of the columns in the tab-delimited .txt file must be 1. scan number, 2. sequence, and 3. FDR
##### Sequences with FDR's up to 0.05 (medium confidence in Proteome Discoverer) should be retained in the reference file
`[--ref_file proteome.faa]`

#### 1 core used by default, but more are intended to be used
`[--cores 8]`

#### Minimum length and probability of sequences reported by postnovo are optional
##### These default to 6 and 0.5, respectively
`[--min_len 9]`

`[--min_prob 0.75]`

#### Use a json parameter file instead of command line arguments
`[--paramfile "param.json]`

#### Turn off verbose mode
`[--quiet]`

#### Usage help
`[--help]`

#### Example command (default predict mode)
`postnovo --frag_mass_tols "0.2, 0.3, 0.4, 0.5, 0.6, 0.7" --denovogui_path "C:\Program Files (x86)\DeNovoGUI-1.15.5-windows\DeNovoGUI-1.15.5\DeNovoGUI-1.15.5.jar" --denovogui_mgf_path "C:\Documents\mgf_files\spectra.mgf" --cores 8`

### Parameter file substitute for command line arguments

A parameter file template, param_template.json, is found in postnovo/test directory

This file lists all of the possible options and can be modified to mirror possible command line arguments

#### Example param file contents
"--test" = true,

"--frag_mass_tols" = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7"],

"--denovogui_path" = "C:\Program Files (x86)\DeNovoGUI-1.15.5-windows\DeNovoGUI-1.15.5\DeNovoGUI-1.15.5.jar",

"--denovogui_mgf_path" = "C:\Documents\mgf_files\spectra.mgf",

"--ref_file" = "proteome_discoverer_psm_table.txt",

"--cores" = 8,

"--min_len" = 9,

"--min_prob" = 0.75
