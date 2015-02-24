----
work in process, complete code will be uploaded soon
----

The experiments have been run on Ubuntu 14.04 64bit

all command should be executed from the /code directory

run experiments
===============
first install the dependencies

run the experiments for our relational feature:
python single_feature_experiments

run the experiments that combine NCC with other relational features:
python combined_feature_experiments

run the experiments for figure 10:
python bow_feature_experiments

run the netkit (wvRN and nLB) experiments with:
python netkit_experiments

run the alchemy experiments with (can take a couple days):
python alchemy experiments


generate plots
==============
tested with: R version 3.1.0 (2014-04-10) and ggplot2_1.0.0

aggregate result files:
python merge_results

generate plots:
Rscript paper_figs.R

dependencies
============
install all python packages listed in dependencies.txt

install netkit-srl-1.4

install alchemy-2
-----------------
the alchemy binaries is not pre compiled (they are rather large 400MB)
if you want to rerun the alchemy experiments please compile the files install
code/alchemy-2

install metis
-------------
http://glaros.dtc.umn.edu/gkhome/metis/metis/download

1) download: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
2) $ gunzip metis-metis-5.1.0.tar.gz
3) $ cd metis-5.1.0
4) $ make config shared=1
5) $ sudo make install (should install to /usr/local/lib/libmetis.so)
if a different install location is choosen adjust path in at feature_extract.py line 84
