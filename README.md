# Postnovo
Post-processing peptide de novo sequences to improve their accuracy

## Quick start
1. Use a powerful server or desktop with plenty of available disk space (~25 GB).
2. Download and decompress the latest [release](https://github.com/semiller10/postnovo/releases).
3. Use Python 3.
[The Anaconda distribution](https://www.anaconda.com/distribution/) comes with all necessary package dependencies.
4. Download [DeNovoGUI](http://compomics.github.io/projects/denovogui.html) and large pre-trained models.
   For low-res MS2 data:
   
   `python main.py setup --denovogui --postnovo_low --deepnovo_low`
   
   For high-res MS2 data:
   
   `python main.py setup --denovogui --postnovo_high --deepnovo_high`
5. [Format the input MGF file.](https://github.com/semiller10/postnovo/wiki/MGF-Input-File-Setup)

   `python main.py format_mgf --mgfs /path/to/preformatted_spectra.mgf --out /path/to/spectra.mgf`
   
6. [Set up a container with TensorFlow](https://github.com/semiller10/postnovo/wiki/DeepNovo-Installation) to run DeepNovo.
Superuser privileges may be required.

   `singularity build tensorflow.simg docker://tensorflow/tensorflow:latest`
7. Generate [DeepNovo](https://github.com/nh2tran/DeepNovo) de novo sequences (can take up to ~12 hours).
The following examples consider low-res MS2 spectra.
Processing high-res spectra requires more memory (see [link](https://github.com/semiller10/postnovo/wiki/Training-and-Running-DeepNovo)).
Postnovo only supports standard fixed C and variable M modifications (due to Novor and PepNovo+).

   Using single machine with 32 cores:

   `python main.py predict_deepnovo --mgf /path/to/spectra.mgf --container /path/to/tensorflow.simg --frag_resolution low --cpus 32`
   
   Using compute cluster via Slurm with 16 cores per node:
   
   `python main.py predict_deepnovo --mgf /path/to/spectra.mgf --container /path/to/tensorflow.simg --frag_resolution low --cpus 16 --slurm --partition partition_with_16GB_mem --time_limit 36`
8. Download and run [MaRaCluster](https://github.com/statisticalbiotechnology/maracluster) (from the link) to cluster spectra by peptide species.
As input, use the reformatted MGF file.
Set "log10(p-value) threshold" to -2.

9. [Run Postnovo](https://github.com/semiller10/postnovo/wiki/Predicting-with-Postnovo), generating [Novor](https://www.rapidnovor.com/download/) and [PepNovo+](http://proteomics.ucsd.edu/Software/PepNovo/) de novo sequences via DeNovoGUI (can take up to ~3 hours).
Results are written by default to the directory of the MGF input.

   `python main.py predict --mgf /path/to/spectra.mgf --clusters /path/to/MaRaCluster.clusters_p2.tsv --frag_method CID --frag_resolution low --denovogui --deepnovo --cpus 32`
