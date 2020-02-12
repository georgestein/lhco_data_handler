# Scripts to transform particle event data to easier summary statistics, used in my submission to the LHCO2020

make_pyjet_images.py: First code to run, to find jets from lhc events, make jet images and get jet summaries, and save to disk as .npz files

lhc_data_loader.py: class to help with loading in data from the .npz files and convert to desired formats

use_lhc_data.py: script to load in data with specified inputs

# useful links:                                                                                                                                                                   

Base script for running pyjet at https://github.com/lhcolympics2020/parsingscripts/blob/master/LHCOlympics2020_pyjetexample.ipynb

Particle data for the LHC Olympics 2020 can be downloaded from https://zenodo.org/record/3596919#.XkSG1BNKjVo

Jet to image described in https://arxiv.org/abs/1803.00107

Particle to Jet overview - https://arxiv.org/pdf/0906.1833.pdf

jet image talk https://indico.cern.ch/event/613571/contributions/2617924/attachments/1496948/2329564/deepjets.pdf

