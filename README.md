# postnovo
## Post-processing peptide de novo sequences to improve their accuracy

## Requirements
Python 3.4 or higher

The Anaconda Python 3.x distribution includes the following postnovo dependencies:

numpy, pandas, sk-learn

### Training model
#### pkl file must go in postnovo/training directory
Default postnovo model available at:

link

## Run program
`python postnovo.py <--frag_mass_tols> <--other options>`

#### Output goes to postnovo/output directory

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
#### Output files should be placed in postnovo/userfiles directory
#### Use of DeNovoGUI (see below) overrides these options

`[--novor_files "novor_output_0.3.novor.csv, novor_output_0.5.novor.csv"]`

`[--pn_files "pn_output_0.3.mgf.out, pn_output_0.5.mgf.out"]`
#### 1 core used by default, but more are intended to be used
`[--cores 8]`
#### Minimum length and probability of sequences reported by postnovo are optional
##### These default to 6 and 0.5, respectively
`[--min_len 9]`

`[--min_prob 0.75]`
#### faa reference file is required in test, train and optimize modes
`[--ref_file proteome.faa]`
#### Options for generating de novo output files with DeNovoGUI and automatically using as postnovo input
##### These override --novor_files and --pn_files
`[--denovogui_path "C:\Program Files (x86)\DeNovoGUI-1.15.5-windows\DeNovoGUI-1.15.5\DeNovoGUI-1.15.5.jar"]`

`[--denovogui_mgf_path "C:\Documents\mgf_files\spectra.mgf"]`

### Parameter file substitute for command line arguments

A parameter file template, param_template.json, is found in postnovo/test directory

postnovo will use a file called param.json for user input if a file with that name is in postnovo/test

Modify param_template.json and save as param.json to override command line arguments

There is an example parameter file in postnovo/test called param_example.json
