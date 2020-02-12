#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

import lhc_data_loader as lhc

BlackBox    = int(sys.argv[1])
signal_frac = float(sys.argv[2])

mmin = 2250. 
mmax = 4750.
msig = 3500.

fields      = ['mass', 'tau_21'] 
sort_all_by = 'mass' 
sort_use_by = 'mass' 

delta_mass = True

load_frac   = 1.0
test_frac = 0.5
if BlackBox > -1:
    signal_frac = 0.0
jets_use = [0,1]
njets = len(jets_use)
njet_sort = njets

# Load Data
DataLoader = lhc.lhcDataLoader(BlackBox=BlackBox,  RandDdir='Path to R&D .npz files', BlackBoxdir='Path to blackbox .npz files')
ind_events, data, cond_param, truth = DataLoader.load_jet_data(mmin=mmin, mmax=mmax,
                                                                fields=fields, 
                                                                load_frac=load_frac, 
                                                                signal_frac=signal_frac,
                                                                jets_use=jets_use,
                                                                sort_all_by=sort_all_by,
                                                                sort_use_by=sort_use_by) 

if delta_mass:
    data[:,1] = data[:,0] - data[:,1]
    
