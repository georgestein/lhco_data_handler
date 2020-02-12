import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

class lhcDataLoader:
    '''class to load in generic lhc datafiles created with make_pyjet_images.py,
    process the fields into the desired format
    and return 'data', 'signal', 'conditional parameter', and 'ind_events' 
    '''
    def __init__(self, BlackBox=-1,
                 verbose=True,
                 RandDdir='/global/cscratch1/sd/gstein/machine_learning/lhc_olympics/data/test_dataset/RandD/',
                 BlackBoxdir='/global/cscratch1/sd/gstein/machine_learning/lhc_olympics/data/BlackBox'):

        self.BlackBox      = BlackBox
        self.RandDdir      = RandDdir
        self.BlackBoxdir   = BlackBoxdir
        self.verbose       = verbose
        
        if self.BlackBox==-1:
            # Load R&D dataset
            self.fdir = self.RandDdir
            
        if self.BlackBox>=0:
            self.fdir = self.BlackBoxdir+str(BlackBox)+'/'

        self.fdir += 'jets/'
        
    # Get mass of lead njet_mjj jets
    def get_mjj(self, datain):
        # last 4 indices are E, px, py, pz
        return (np.sqrt(np.sum(datain[:,self.jets_mjj,-4], axis=1)**2 -
                        np.sum(datain[:,self.jets_mjj,-3], axis=1)**2 -
                        np.sum(datain[:,self.jets_mjj,-2], axis=1)**2 - 
                        np.sum(datain[:,self.jets_mjj,-1], axis=1)**2))

    def load_jet_data(self, fname_head='jet_images_events', load_frac=.1,
                      signal_frac=0.005, mmin = 1750., mmax=6250.,
                      fields=['mass', 'tau_21'], sort_all_by='mass',
                      sort_use_by='mass',
                      jets_use=[0,1], jets_mjj=[0,1]):

        ### LOAD IN FILES
        self.fname_head = fname_head
        self.load_frac  = load_frac
        self.fields     = fields
        self.jets_use   = jets_use
        self.jets_mjj   = jets_mjj

        self.njets_use = len(self.jets_use)
        self.njets_mjj = len(self.jets_mjj)

        #sort_by: index of field to sort event jets by, in descending order
        self.sort_all_by = sort_all_by
        self.sort_use_by = sort_use_by
        self.signal_frac = signal_frac

        files = sorted(glob.glob(self.fdir+self.fname_head+'*'))
        nfiles = len(files)

        # use only the fraction of files desired
        nfiles = int(nfiles*self.load_frac)

        if self.verbose: print('\nloading %d files'%nfiles)
        files  = files[:nfiles]

        
        # loop over files to find total number of events that made it through cuts 
        nevents_tot = 0
        for i, file in enumerate(files):
            filein       = np.load(file)
            nevents_i    = filein['signal'].shape[0]
            self.summary_info = filein['img_summary_info']

            nevents_tot += nevents_i

        if self.verbose:
            print('\ntotal number of events is ', nevents_tot)
            print('\nimage summaries contain: ', self.summary_info)

        # check if desired fields exist
        for field in self.fields:
            assert field in np.append(self.summary_info,['tau_21', 'tau_32', 'tau_43','tau_31', 'deltaR']), 'field %s does not exist, fields are: %s'%(field, self.summary_info)

        njets       = filein['img_summary'].shape[1] 

        # initialize empty arrays to put data into
        ind_events = np.arange(nevents_tot)
        signal      = np.zeros(nevents_tot, dtype=np.bool)
        img_summary = np.zeros((nevents_tot, njets, len(self.summary_info)))

        nevents_prev = 0
        for i, file in enumerate(files):
            if self.verbose: print("\nloading data from ", file)

            filein       = np.load(file)      
            signali      = filein['signal']      
            nevents_i     = len(signali)

            signal[nevents_prev:(nevents_prev+nevents_i)]       = signali         
            img_summary[nevents_prev:(nevents_prev+nevents_i)]  = filein['img_summary']
    
            nevents_prev += nevents_i


        # get other desired jet summaries

        # usual fields: ['sum_pT' 'pT' 'eta' 'phi' 'mass' 'E' 'px' 'py' 'pz' 'tau_1' 'tau_2','tau_3' 'tau_4']
        # get indices of desired fields
        fields_collect = [self.sort_all_by] + self.fields

        if 'tau_21' in self.fields:
            # get where tau_21 is in fields, to replace with tau_2 and tau_1
            ind_tau_21  = fields_collect.index("tau_21")
            #replace tau_21 with [tau2, tau1]
            fields_collect = fields_collect[:ind_tau_21] + ['tau_1', 'tau_2'] +fields_collect[ (ind_tau_21+1):]


        if 'tau_32' in self.fields:
            # get where tau_32 is in fields, to replace with tau_2 and tau_1
            ind_tau_32  = fields_collect.index("tau_32")
            #replace tau_32 with [tau2, tau1]
            fields_collect = fields_collect[:ind_tau_32] + ['tau_2', 'tau_3'] +fields_collect[ (ind_tau_32+1):]


        if 'tau_43' in self.fields:
            # get where tau_43 is in fields, to replace with tau_2 and tau_1
            ind_tau_43  = fields_collect.index("tau_43")
            #replace tau_32 with [tau2, tau1]
            fields_collect = fields_collect[:ind_tau_43] + ['tau_3', 'tau_4'] +fields_collect[ (ind_tau_43+1):]

        if 'tau_31' in self.fields:
            # get where tau_32 is in fields, to replace with tau_2 and tau_1
            ind_tau_31  = fields_collect.index("tau_31")
            #replace tau_32 with [tau2, tau1]
            fields_collect = fields_collect[:ind_tau_31] + ['tau_1', 'tau_3'] +fields_collect[ (ind_tau_31+1):]
        if 'deltaR' in self.fields:
            # get where deltaR is in fields, to replace with eta and phi
            ind_deltaR = fields_collect.index("deltaR")
            #replace deltaR with [eta, phi]   
            fields_collect = fields_collect[:ind_deltaR] + ['eta', 'phi'] +fields_collect[ (ind_deltaR+1):]


        # remove duplicates, if any
        fields_collect = list(dict.fromkeys(fields_collect))
        # add to get mass, then crop later
        fields_collect += ['E', 'px', 'py', 'pz']

        ind_fields = [int(np.where(self.summary_info == fieldi)[0]) for fieldi in fields_collect]
        if self.verbose:
            print('\nfields to collect = ', fields_collect[:-4])
            print('\nindex of fields to collect = ', ind_fields[:-4])
                

        # get data array constructed from desired fields
        #        data = img_summary[:, 0:njet_use, [4,5,6,7]].reshape((img_summary.shape[0],-1), order='F') #order F gives [m1, m2, tau1, tau2] instead of [m1, tau1, m2, tau2]

        # LOAD IN DATA
        # load all jet data to perform sort
        data = img_summary[:,:,ind_fields]

        # sort data by specified axis, in descending order
        data = np.array([data[i,ind] for i, ind in enumerate(np.argsort(data[:,:,0], axis=1)[:,::-1])])
        if self.sort_all_by not in self.fields:
            # removing sorting paramater
            data = data[:,:,1:]
            fields_collect = fields_collect[1:]


        # cut to only desired jets
        data = data[:, jets_use]
        # sort remaining jets by desired paramater, in descending order
        ind_sort = fields_collect.index(self.sort_use_by)
        data = np.array([data[i,ind] for i, ind in enumerate(np.argsort(data[:,:,ind_sort], axis=1)[:,::-1])])
        
        # get conditional parameter
        mjj   = self.get_mjj(data)
        data  = data[:,:,:-4]

        # get subjetiness ratios, and/or deltaR, and keep only necessary columns
        ind_tau_keep = []
        ind_tau_del  = []
        if 'tau_43' in self.fields:
            ind_tau_3  = fields_collect.index("tau_3")
            ind_tau_4  = fields_collect.index("tau_4")

            ind_tau_keep.append(ind_tau_4)
            ind_tau_del.append(ind_tau_3)

            # toss out any strange values of subjettiness
            dm1   = (data[:,:,ind_tau_4] == 0.)
            dm2   = (data[:,:,ind_tau_3] == 0.)

            data[:,:,ind_tau_4] /= data[:,:,ind_tau_3]
            
            data[dm1, ind_tau_4] = 1.
            data[dm2, ind_tau_4] = 0.

        if 'tau_32' in self.fields:
            ind_tau_2  = fields_collect.index("tau_2")
            ind_tau_3  = fields_collect.index("tau_3")

            ind_tau_keep.append(ind_tau_3)
            ind_tau_del.append(ind_tau_2)
            # toss out any strange values of subjettiness
            dm1   = (data[:,:,ind_tau_3] == 0.)
            dm2   = (data[:,:,ind_tau_2] == 0.)

            data[:,:,ind_tau_3] /= data[:,:,ind_tau_2]
            
            data[dm1, ind_tau_3] = 1.
            data[dm2, ind_tau_3] = 0.

        if 'tau_21' in self.fields:

            ind_tau_1  = fields_collect.index("tau_1")
            ind_tau_2  = fields_collect.index("tau_2")

            ind_tau_keep.append(ind_tau_2)
            ind_tau_del.append(ind_tau_1)

            # toss out any strange values of subjettiness
            dm1   = (data[:,:,ind_tau_2] == 0.)
            dm2   = (data[:,:,ind_tau_1] == 0.)

            data[:,:,ind_tau_2] /= data[:,:,ind_tau_1]
            
            data[dm1, ind_tau_2] = 1.
            data[dm2, ind_tau_2] = 0.

        if 'tau_31' in self.fields:
            ind_tau_1  = fields_collect.index("tau_1")
            ind_tau_3  = fields_collect.index("tau_3")

            ind_tau_keep.append(ind_tau_3)
            ind_tau_del.append(ind_tau_1)
            # toss out any strange values of subjettiness
            dm1   = (data[:,:,ind_tau_3] == 0.)
            dm2   = (data[:,:,ind_tau_1] == 0.)

            data[:,:,ind_tau_3] /= data[:,:,ind_tau_1]
            
            data[dm1, ind_tau_3] = 1.
            data[dm2, ind_tau_3] = 0.
            
        if 'deltaR' in self.fields:

            ind_eta  = fields_collect.index("eta")
            ind_phi  = fields_collect.index("phi")
            ind_mass = fields_collect.index("mass")

            if 'eta' not in self.fields:
                ind_tau_del.append(ind_eta)
            if 'phi' not in self.fields:
                ind_tau_del.append(ind_phi)

            # calculate deltaR's
            ndeltaR = int(self.njets_use * (self.njets_use - 1)/2) # number of pairs of vertices
            deltaR = np.zeros((data.shape[0], ndeltaR))
            deltaR_ind = 0
            for jeti in self.jets_use[:-1]:
                for jetj in self.jets_use[(jeti+1):]:
                    deltaR[:,deltaR_ind] = np.sqrt( (data[:,jeti,ind_eta] - data[:,jetj,ind_eta])**2 +
                                                    ( (data[:,jeti,ind_phi] - data[:,jetj,ind_phi]) % np.pi)**2 )
                    # if jet does not exist (mass = 0), then set distance to 0
                    dm = (data[:, jeti, ind_mass] == 0.) | (data[:, jetj, ind_mass] == 0.)
                    deltaR[dm, deltaR_ind] = 0.
                    
                    deltaR_ind +=1
                    

        if 'sum_pT' in self.fields:
            ind_sum_pT  = fields_collect.index("sum_pT")
            ind_tau_del.append(ind_sum_pT)
            sum_pT = data[:,0,ind_sum_pT]

        ind_del = [ind not in ind_tau_keep for ind in ind_tau_del]
        ind_del = [ind for i, ind in enumerate(ind_tau_del) if ind_del[i]==True]
        data = np.delete(data, ind_del, 2)

        data = data.reshape(data.shape[0], -1, order='F')

        if 'sum_pT' in self.fields:
            # append deltaR's to columns of data
            data = np.c_[data, sum_pT]

        if 'deltaR' in self.fields:
            # append deltaR's to columns of data
            data = np.c_[data, deltaR]
            
        
        np.random.seed(13579)
        ind_rand = np.random.permutation(data.shape[0])

        ind_events = ind_events[ind_rand]
        data   = data[ind_rand]
        signal = signal[ind_rand]
        mjj    = mjj[ind_rand]

        nsig_lt = np.cumsum(signal)

        dm_sig = (signal == 1)
        dm_bg  = ~dm_sig
        
        nbg = signal[dm_bg].shape[0]

        nsig = int(nbg * self.signal_frac)
        
        print(nsig)
        # make dataset signal_frac signal
        if nsig < np.sum(signal):
            isig_cut = np.where(nsig_lt == nsig)[0][0]
            dm_sig[isig_cut:] = False 

            dm = (np.logical_or(dm_sig,dm_bg))

            ind_events = ind_events[dm]
            data   = data[dm]
            signal = signal[dm]
            mjj    = mjj[dm]
            if self.verbose: print('\nN_signal=%d, N_background=%d, ratio=%.4f'%(dm_sig.sum(), dm_bg.sum(), signal.sum()/len(signal) ))
        else:
            print('\n%f signal_frac desired is greater than exists in dataset (tot=%f, frac=%f), not modifying'%(self.signal_frac, np.sum(signal), np.sum(signal)/nbg) )

        # shuffle again to evenly distrubute signal
        ind_rand = np.random.permutation(data.shape[0])
        ind_events = ind_events[ind_rand]
        data   = data[ind_rand]
        signal = signal[ind_rand]
        mjj    = mjj[ind_rand]

        # Cut data not in broad mass range
        dm = (mjj > mmin) & (mjj < mmax)
        ind_events = ind_events[dm]
        data = data[dm]
        signal = signal[dm]
        mjj    = mjj[dm]


        # if lots of data=0, it breaks. add small fluctuation
        ind = np.where(data == 0.)
        print('\nNumber of events with zeros = ',len(np.unique(np.where(data == 0.)[0])))
        data[ind] += np.random.uniform(0,1e-6, size=len(ind[0]))
        if self.verbose: print('\nFinal data shape, means = ', data.shape, np.mean(data,axis=0))

        return ind_events, data, mjj, signal 



    




















