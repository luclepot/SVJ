import numpy as np
import h5py
import ROOT as rt
import sys
import os

#####

try:
    DELPHES_DIR = os.environ["DELPHES_DIR"]
except:
    print("WARNING: Did you forget to source 'setup.sh'??")
    sys.exit(0)


rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))

#####

def PFCandsInJet(p, DR, PtMap):
    if p.Pt() <= 0.: return 0.
    # get the coordinates
    DeltaEta = PtMap[:,0] - p.Eta()
    DeltaPhi = PtMap[:,1] - p.Phi()
    pi = rt.TMath.Pi()
    DeltaPhi = DeltaPhi - 2*pi*(DeltaPhi >  pi) + 2*pi*(DeltaPhi < -1.*pi)
    isInJet = (DeltaEta*DeltaEta + DeltaPhi*DeltaPhi) < DR*DR
    DeltaEta = DeltaEta[isInJet]
    DeltaPhi = DeltaPhi[isInJet]
    pT = PtMap[isInJet,2]
    DeltaEta = np.reshape(DeltaEta, (DeltaEta.shape[0],1))
    DeltaPhi = np.reshape(DeltaPhi, (DeltaEta.shape[0],1))
    pT = np.reshape(pT, (DeltaEta.shape[0],1))
    myMap = np.concatenate((DeltaEta,DeltaPhi,pT), axis=1)
    #print(DeltaEta.min(), DeltaEta.max(), DeltaPhi.min(), DeltaPhi.max())
    return myMap

def ChPtMapp(event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowTrack:
        if h.PT<= 0.1: continue
        pTmap.append([h.Eta, h.Phi, h.PT])
    return np.asarray(pTmap)

def NeuPtMapp(event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowNeutralHadron:
        if h.ET<= 0.5: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
    return np.asarray(pTmap)

def PhotonPtMapp(event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowPhoton:
        if h.ET<= 0.2: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
    return np.asarray(pTmap)

#####

def Convert(filenameIN, filenameOUT, sideband = False, verbose = False):
    inFile = rt.TFile.Open(filenameIN)
    tree = inFile.Get("Delphes")
    q = rt.TLorentzVector()
    eventFeatureNames = ['mJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M', 'j1E', 'j2Pt', 'j2M', 'j2E', 'DeltaEtaJJ', 'DeltaPhiJJ']
    particleFeatureNames = ['pEta', 'pPhi', 'pPt']

    eventFeatures = np.array([])
    jetConstituentsList = np.array([])
    #Nevt = 0
    for i,event in enumerate(tree):

        if i < 1275:
            continue
        if i > 1275:
            break

        if verbose: print("New Event")
        #Nevt += 1
        #if Nevt>100: continue
        # apply dijet selection
        goodJets = np.asarray([np.array([jet.PT, jet.Eta, jet.Phi, jet.Mass]) for jet in event.Jet])
        # for jet in event.Jet:
        #     if verbose: print("New Jet")
        #     if jet.PT>30. and abs(jet.Eta)<2.5:
        #         myJet = np.array([jet.PT, jet.Eta, jet.Phi, jet.Mass])
        #         myJet = np.reshape(myJet, (1, 4))
        #         if goodJets.size:
        #             goodJets = np.concatenate((goodJets, myJet), axis=0)
        #         else:
        #             goodJets = myJet
                    
        # at least two jets found
        # sort by pT
        goodJets = goodJets[goodJets[:,0].argsort()]
        goodJets = goodJets[::-1]
        
        # PFcands (eta,phi) maps
        TrkPtMap = ChPtMapp(event)
        NeuPtMap = NeuPtMapp(event)
        PhotonPtMap = PhotonPtMapp(event)

        # loop over jets and make maps
        jetDR = 0.8
        thisJetConstituentsList = np.array([])

        for i in range(2):
            jet = goodJets[i,:]
            myjet = rt.TLorentzVector()
            myjet.SetPtEtaPhiM(jet[0], jet[1], jet[2], jet[3])
            # find jet costituents
            trk = PFCandsInJet(myjet, jetDR, TrkPtMap)
            gamma = PFCandsInJet(myjet, jetDR, PhotonPtMap)
            Neu = PFCandsInJet(myjet, jetDR, NeuPtMap)
            pList = np.concatenate([trk, gamma, Neu], axis = 0)            
            if pList[i,0].max()>jetDR:
                print(pList[:,0].max(), pList[:,0].min())
            if pList[i,0].min()<-1.*jetDR:
                print(pList[:,0].max(), pList[:,0].min())
            #for i in range(pList.shape[0]):
            ##    if pList[i,0]>jetDR or pList[i,0]<-1.*jetDR: 
            #        print pList[i,2]
            # sort constituents by pT
            pList = pList[pList[:,2].argsort()]
            pList = pList[::-1]
            # store particles
            print(pList.shape)
            myList = pList[:100,:]
            myZeros = np.zeros((100-myList.shape[0], myList.shape[1]))
            myList =  np.concatenate((myList,myZeros), axis=0)
            myList = np.reshape(myList, (1, myList.shape[0], myList.shape[1]))
            if thisJetConstituentsList.size:
                thisJetConstituentsList = np.concatenate((thisJetConstituentsList, myList), axis=0)
            else:
                thisJetConstituentsList = myList
        #print(thisJetConstituentsList[:,0].min(), thisJetConstituentsList[:,0].max(), thisJetConstituentsList[:,1].min(), thisJetConstituentsList[:,1].max())
        if verbose: print("Write Output")
        # now store candidates
        thisJetConstituentsList = np.reshape(thisJetConstituentsList, (1, thisJetConstituentsList.shape[0], 
                                                                       thisJetConstituentsList.shape[1], 
                                                                       thisJetConstituentsList.shape[2]))
        jetConstituentsList = np.concatenate((jetConstituentsList, thisJetConstituentsList), axis=0) if jetConstituentsList.size else thisJetConstituentsList
        # now store jet features
        j1 = rt.TLorentzVector()
        j1.SetPtEtaPhiM(goodJets[0,0], goodJets[0,1], goodJets[0,2], goodJets[0,3])
        j2 = rt.TLorentzVector()
        j2.SetPtEtaPhiM(goodJets[1,0], goodJets[1,1], goodJets[1,2], goodJets[1,3])
        thisEventfeatures = np.array([(j1+j2).M(), j1.Eta(), j1.Phi(), j1.Pt(), j1.M(), j1.E(), j2.Pt(), j2.M(), j2.E(), j1.Eta()-j2.Eta(), j1.DeltaPhi(j2)])
        thisEventfeatures = np.reshape(thisEventfeatures, (1, len(eventFeatureNames)))
        eventFeatures = np.concatenate((eventFeatures, thisEventfeatures), axis=0) if eventFeatures.size else thisEventfeatures


    return jetConstituentsList, eventFeatures
    #
    # f = h5py.File(filenameOUT, "w")
    # f.create_dataset('eventFeatures', data=eventFeatures, compression='gzip')
    # f.create_dataset('jetConstituentsList', data=jetConstituentsList, compression='gzip')
    # f.create_dataset('eventFeatureNames', data=eventFeatureNames, compression='gzip')
    # f.create_dataset('particleFeatureNames', data=particleFeatureNames, compression='gzip')
    # f.close()


if __name__ == "__main__":
    import sys
    print sys.argv[1]
    ret2 = Convert(sys.argv[1], sys.argv[2], sys.argv[3]=="sideband")
    
