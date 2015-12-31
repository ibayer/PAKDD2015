=============================================================
Graph Based Relational Features for Collective Classification
=============================================================

Please cite the following paper if you use our code:

```
@incollection{bayer2015graph,
  title={Graph Based Relational Features for Collective Classification},
  author={Bayer, Immanuel and Nagel, Uwe and Rendle, Steffen},
  booktitle={Advances in Knowledge Discovery and Data Mining},
  pages={447--458},
  year={2015},
  publisher={Springer}
}
```

Disclaimer
---------
This code has only been run on Ubuntu 14.04 64bit and not tested elsewhere!
Is not very well documented (please refere to the paper for details) and has
not been written with performance in mind. We hope you find it still useful
to reproduce our experiments.


Run Experiments
===============
All command should be executed from the ``/code`` directory

run the experiments for our relational feature:
``python single_feature_experiments.py``

run the experiments that combine NCC with other relational features:
``python combined_feature_experiments.py``

run the experiments for figure 10:
``python bow_feature_experiments.py``

run the netkit (wvRN and nLB) experiments with:
``python netkit_experiments.py``

run the alchemy experiments with (can take a couple days):
``python alchemy_experiments.py``


Generate Plots
==============
tested with: R version 3.1.0 (2014-04-10) and ggplot2_1.0.0

Aggregate result files:
``python merge_results.py``

Generate plots:
``Rscript paper_figs.R``

Dependencies
============
1) "Metis for Python" ``pip install https://bitbucket.org/kw/metis-python/get/1ba8f5f85608.zip``

2) ``pip install  numpy==1.8.0``

3) the remaining dependencies listed in dependencies.txt
``pip install -r code/dependencies.txt``

Install metis
-------------
http://glaros.dtc.umn.edu/gkhome/metis/metis/download

1) download: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz

2) $ gunzip metis-metis-5.1.0.tar.gz

3) $ cd metis-5.1.0

4) $ make config shared=1

5) $ sudo make install (should install to ``/usr/local/lib/libmetis.so``)
if a different install location is chosen adjust path ``feature_extract.py line 84``

Dependencies For Benchmark Models
=================================

Install alchemy-2
-----------------
1) download from https://code.google.com/p/alchemy-2/

2) install to ``code/alchemy-2``, be careful it's > 400MB compiled and the
experiments take a looong time.


Install netkit-srl-1.4
----------------------
1) download it from http://sourceforge.net/projects/netkit-srl/files/netkit-srl/netkit-srl-v1.4.0/

2) extract to ``code/netkit-srl-1.4.0/``
