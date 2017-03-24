# postnovo
#### Post-processing peptide de novo sequences

Supported on any OS running Python

### Requirements
Python 3.4 or higher
conda Python 3 distribution will include following dependencies:
numpy, pandas, sk-learn

### Training model
#### pkl file must go in postnovo/training directory
Default postnovo model available at:
link

### Run program
python postnovo.py (args)

Output goes to postnovo/output directory

#### Four modes:
Predict (DEFAULT MODE) = post-process de novo sequences for data WITHOUT reference

Test = post-process de novo sequences for data WITH reference

Train = update postnovo model with de novo sequences with reference

Optimize = same as train, but a few model parameters are tuned

### Command line options
#### Choosing any of these flags overrides default predict mode
--test

--train

--optimize
#### At least 1 fragment mass tolerance and corresponding Novor and PepNovo+ output files are required
#### These files should be placed in postnovo/userfiles directory
--frag_mass_tols "0.3, 0.5"

--novor_files "novor_output_0.3.novor.csv, novor_output_0.5.novor.csv"

--pn_files "pn_output_0.3.mgf.out, pn_output_0.5.mgf.out"
#### postnovo uses 1 core by default, but more are intended to be used
--cores 3
#### Minimum length and probability of sequences reported by postnovo are optional
##### These default to 6 and 0.5, respectively
--min_len 9

--min_prob 0.75
#### faa reference file is mandatory in test, train and optimize modes
--ref_file proteome.faa
#### Options for generating de novo output with DeNovoGUI and then using as postnovo input
##### These are mutually exclusive with --novor_files and --pn_files
--denovogui_path "C:\Program Files (x86)\DeNovoGUI-1.15.5-windows\DeNovoGUI-1.15.5\DeNovoGUI-1.15.5.jar"

--denovogui_mgf_path "C:\Documents\mgf_files\spectra.mgf"

### Parameter file substitute for command line arguments

Parameter file template, param_template.json, found in postnovo/test directory

Save as param.json, and postnovo will attempt to use this file instead of command line arguments
