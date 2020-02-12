# Script to load in data, find jets using pyjet, and make jet images

# useful links:
# Base script for running pyjet at https://github.com/lhcolympics2020/parsingscripts/blob/master/LHCOlympics2020_pyjetexample.ipynb
# Jet data can be downloaded from https://zenodo.org/record/3596919#.XkSG1BNKjVo

# Jet to image described in https://arxiv.org/abs/1803.00107
# Particle Jet overview - https://arxiv.org/pdf/0906.1833.pdf
# jet image talk https://indico.cern.ch/event/613571/contributions/2617924/attachments/1496948/2329564/deepjets.pdf

import h5py    
import numpy as np 
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd

# each event contains pt, eta, phi for up to 700 particles (0 padded if none available),
# and the last column contains 0:background , 1:signal
nparticles = 700
R = 1.
eta_cut=2.5

nevents_tot   = 1000000
chunksize     = 100000

# number of jets to keep 
njet_keep    = 6

# calculate subjettiness paramaters
subjettiness = True

# dataset to use
BlackBox=1


# R&D DATASET
#nevents_tot   = 1100000
#fdir = 'PATH TO DATA/RandD/'
#fname  = fdir+'events_anomalydetection.h5'
#supervised = True

# BLACK BOX 1-3
fdir  = 'PATH TO DATA/BlackBox%d/'%BlackBox
fname = fdir+'events_LHCO2020_BlackBox%d.h5'%BlackBox
supervised = False

sigtypes = ['background','signal']

images_summary_info = ['sum_pT', 'pT', 'eta', 'phi', 'mass', 'E', 'px', 'py', 'pz']
if subjettiness:
    images_summary_info += ['tau_1', 'tau_2', 'tau_3', 'tau_4']
nchunks = nevents_tot//chunksize

def get_datachunk(fname, nevent_start=0, nevent_end=5000, supervised=False):

    
    events = pd.read_hdf(fname, start=nevent_start, stop=nevent_end)

    
    print("Memory in GB:",sum(events.memory_usage(deep=True)) / (1024**3))


    # signal is binary (nevents), events are (nevents,nparticles,3), 3:pt,eta,phi
    if supervised:
        signal = events.values[:,2100].astype('int')
        events = events.values[:,:-1].reshape(nevent_end-nevent_start,nparticles,-1)

    else:
        events = events.values.reshape(nevent_end-nevent_start,nparticles,-1)
        signal = np.zeros(events.shape[0]).astype('int')

    print(events.shape, signal.shape)        
    print("Ratio of BSM events to background = ", signal.sum()/len(signal))


    return signal, events



def get_jets(events, signal, R=1.0, eta_cut=2.5):
    # Cluster jets using pyjet
    # pyjet usage cleaned up from  https://github.com/lhcolympics2020/parsingscripts/blob/master/LHCOlympics2020_pyjetexample.ipynb

    alljets = {}
    for mytype in sigtypes:
        alljets[mytype]=[]

    for i in range(events.shape[0]):
        if (i%10000==0): print('done ', i)

        pseudojets_input = np.zeros(len([x for x in events[i,:,0] if x > 0]), dtype=DTYPE_PTEPM)
        for j in range(pseudojets_input.shape[0]):
            pseudojets_input[j]['pT'] = events[i,j,0]
            pseudojets_input[j]['eta'] = events[i,j,1]
            pseudojets_input[j]['phi'] = events[i,j,2]

        sequence = cluster(pseudojets_input, R=R, p=-1)
        jets     = sequence.inclusive_jets(ptmin=50)
        # jets come out ranked in descending order of pt

        # cut jets not within |eta_cut|
        jets_keep = []
        for j in range(len(jets)):
            # print(j, jets[j])
            if abs(jets[j].eta) < eta_cut:
                jets_keep += [jets[j]]

        if jets_keep == []: 
            print("NO JETS FOUND WITHIN |ETA|<%.3f, THROWING OUT EVENT"%eta_cut, i, mytype, jets)
        else:
            alljets[sigtypes[signal[i]]] += [jets_keep]

    return alljets


def jet_to_image(event, nbins=37, bin_minmax=R, ievent=0, check=False):
    # Construct images from jets
    binedges = np.linspace(-bin_minmax,bin_minmax,nbins+1)
    images_event = []

    for i in range(len(event)):
        # get jet properties
        jeti      = event[i]
        # get properties of particles in jet
        jeti_part = event[i].constituents_array()#ep=energy) 
        
        jet_part_eta = jeti_part['eta']
        jet_part_phi = jeti_part['phi']
        jet_part_pt  = jeti_part['pT']
    
        # if eta is near pi some particles can be near -pi, 
        # some near pi, so average does not work to get center.
        # therefore wrap to <pi/2 if |phi| > pi/2 
        if np.mean(abs(jet_part_phi)) > np.pi/2: 
            jet_part_phi = -np.pi + (jet_part_phi)%(2*np.pi)
  

            # pT weighted center of jet
        jet_eta = np.sum(jet_part_eta*jet_part_pt)/np.sum(jet_part_pt)
        jet_phi = np.sum(jet_part_phi*jet_part_pt)/np.sum(jet_part_pt)

        # get 2d array to calculate eigenvectors and eigenvalues
        xy      = np.c_[jet_part_eta-jet_eta, jet_part_phi-jet_phi]
        weights = jet_part_pt * np.sum(xy**2,axis=1)
        #     weights = jet_part_pt * np.sqrt(np.sum(xy**2,axis=0))

        if xy.shape[0] > 1: # if only one particle in jet orientation doesn't work
    

            cov = np.cov(xy, rowvar=False, aweights=weights)
            # return eigenvalues in decreasing order
            val, vec = np.linalg.eigh(cov)
            # print(ev, eig)
            
            xy = (vec.dot(xy.T)).T

        # compute values in quadrants
        l = (xy[:,0] < 0.)
        r = (xy[:,0] > 0.)
        d = (xy[:,1] < 0.)
        u = (xy[:,1] > 0.)
        
        tl = jet_part_pt[l].sum()
        tr = jet_part_pt[r].sum()
        td = jet_part_pt[d].sum()
        tu = jet_part_pt[u].sum()
        
        if tl > tr: # flip left-right
            xy[:,0] *= -1.
            
        if td > tu: # flip upper-lower
            xy[:,1] *= -1.
                
                
        # Check image was oriented properly
        if check: 
            ul = (xy[:,0] < 0.) & (xy[:,1] > 0.)
            ur = (xy[:,0] > 0.) & (xy[:,1] > 0.)
            lr = (xy[:,0] > 0.) & (xy[:,1] < 0.)
            ll = (xy[:,0] < 0.) & (xy[:,1] < 0.)
            
            tul = jet_part['pT'][ul].sum()
            tur = jet_part['pT'][ur].sum()
            tlr = jet_part['pT'][lr].sum()
            tll = jet_part['pT'][ll].sum()
            print("ul, ur, lr, ll = ", tul, tur, tlr, tll)

        h, x, y = np.histogram2d(xy[:,1], xy[:,0], bins=binedges, weights=jet_part_pt)
        
        images_event += [h]

    return images_event


def jet_to_summary(event, subjettiness=False, R=R):

  summary_event = []

  # calculate total pT of all jets
  pttot = 0.
  for i in range(len(event)):
    # get jet properties
    jeti  = event[i]
    pttot += jeti.pt
  
  for i in range(len(event)):
    # get jet properties
    jeti      = event[i]

    pti  = jeti.pt
    etai = jeti.eta
    phii = jeti.phi
    mi   = jeti.mass
    Ei  = jeti.e
    pxi = jeti.px
    pyi = jeti.py
    pzi = jeti.pz

    four_vectors = [pttot, pti, etai, phii, mi, Ei, pxi, pyi, pzi]
    
    if subjettiness:
        # subjettiness defined in https://arxiv.org/abs/1011.2268
        particles = jeti.constituents_array()
        nparticles = len(particles['eta'])
        sequence  = cluster(particles, R=R, p=1)

        # calculate \tau_1, ..., \tau_N
        nsubjets = [1,2,3,4]
        tau = np.zeros(len(nsubjets))
        for nsubj in nsubjets:
            # get nsubj subjets from jet 
            if nsubj >= nparticles: continue 
            subjets = sequence.exclusive_jets(nsubj)
            
            # find subjet centrals and put in array
#             subjet_etaphi = np.zeros((nsubj,2))
            
            # calculate distance from each particle to each subjet center
    
            Delta_Rs      = np.zeros((nparticles,nsubj))
            for j, subj in enumerate(subjets):

                # if phi is near pi some particles can be near -pi, 
                # some near pi, so average does not work to get center.
                # therefore wrap to <pi/2 if |phi| > pi/2 
                if abs(subj.phi) > np.pi/2: 
                    Delta_Rs[:,j] = np.sqrt((particles['eta']-subj.eta)**2 +
                                            (particles['phi']%(2*np.pi) - subj.phi%(2*np.pi) )**2)
                else:
                    Delta_Rs[:,j] = np.sqrt((particles['eta']-subj.eta)**2 +
                                            (particles['phi']-subj.phi)**2)
    
            Delta_R = np.min(Delta_Rs, axis=1)
            
            # tau_N = 1/d0 \sum_k p_{T,k} min{\Delta R_{1,k}, ..., \Delta R_{N,k}}
            # d0 = \sum_k p_{T,k} R
            # \Delta R = sqrt((\Delta eta)^2+(\Delta phi)^2)
            tau[nsubj-1] = 1./np.sum(particles['pT']*R) * np.sum(particles['pT']*Delta_R)

            if tau[nsubj-1] > 1.:
                print("broke on event", event)
                print("pti, etai, phii = ",pti, etai, phii )

        summary_event += [four_vectors+list(tau)]
    else:
        summary_event += [four_vectors]

  return summary_event

# Construct images from jet events and get jet image summary values
def jet_to_image_all(alljets, subjettiness=True, bin_minmax=R):
    images = {}
    images_summary = {}

    for mytype in sigtypes:

      images[mytype] = []
      images_summary[mytype] = []

      for ievent in range(len(alljets[mytype])):
        if ievent % 10000 == 0: print("done event ", ievent)
        # print(mytype, ievent)
        images[mytype] += [jet_to_image(alljets[mytype][ievent], bin_minmax=bin_minmax, ievent=ievent)]
        images_summary[mytype] += [jet_to_summary(alljets[mytype][ievent], subjettiness=subjettiness)]


    return images, images_summary


def jet_dict_to_numpy(images, images_summary, njet_keep=4):
    # put images into numpy arrays for easier ml applications and to save to disk
    # keep leading njet_keep jets. Zero pad if they do not exist

    # some events may not have jets, so this can be smaller than number of input events 
    njetevents_keep = len(images['background'])+len(images['signal'])

    # number of pixels in 1D of image
    nbins = len(images['background'][0][0][0])

    # top njet_keep jet images
    img_events = np.zeros((njetevents_keep, njet_keep, nbins, nbins))
    # signal or background
    img_signal     = np.zeros(njetevents_keep)

    # E, px, py, pz for each jet
    img_summary = np.zeros( (njetevents_keep, njet_keep, len(images_summary['background'][0][0])))

    print(img_events.shape)

    event_i = 0
    for mytype in sigtypes:
      njetevents = len(images[mytype])
    
      for ievent in range(njetevents):
        njets = min(len(images[mytype][ievent]), njet_keep)
        img_events[event_i,:njets] = images[mytype][ievent][:njets]
        img_summary[event_i,:njets] = images_summary[mytype][ievent][:njets]

        if mytype == 'signal': img_signal[event_i] = 1
        event_i += 1
    
    return img_events, img_summary, img_signal

# RUN JET FINDER ON ALL EVENTS AND MAKE AND SAVE IMAGES
for i in range(nchunks):
    print('RUNNING ON DATACHUNK', i, 'OF',nchunks )
 
    event_start = i*chunksize
    event_end   = (i+1)*chunksize

    print('events in range ', event_start, event_end)

    # get data
    print('\nGetting Data\n'+'-'*18)
    signal, events = get_datachunk(fname, nevent_start=event_start, nevent_end=event_end, supervised=supervised)

    # find jets
    print('\nFinding Jets\n'+'-'*18)
    alljets = get_jets(events, signal, R=R, eta_cut=eta_cut)

    # make dictionary of images
    print('\nMaking Jet Images\n'+'-'*18)
    images, images_summary = jet_to_image_all(alljets, subjettiness=subjettiness, bin_minmax=R)

    # convert from dictionary to numpy arrays
    print('\nGetting Numpy arrays\n'+'-'*18)
    img_events, img_summary, img_signal = jet_dict_to_numpy(images, images_summary, njet_keep=njet_keep)

    # save to disk
    print('\nSaving to Disk\n'+'-'*18)
    np.savez(fdir+'jet_images_events_'+str(event_start).zfill(7)+'_'+str(event_end).zfill(7)+'_R'+str(R)+'.npz', 
             img_events=img_events.astype('float32'), 
             signal=img_signal.astype('bool'), 
             img_summary=img_summary.astype('float32'),
             img_summary_info=images_summary_info)


