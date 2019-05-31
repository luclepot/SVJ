import numpy as np
import math
import os
import argparse
import sys
import time
from traceback import format_exc

class Converter:

    LOGMSG = "Converter :: "

    def __init__(
        self,
        inputdir,
        outputdir,
        name,
        jetDR=0.8,
        n_constituent_particles=100,
    ):
        self.inputdir = Converter.smartpath(inputdir)
        self.outputdir = Converter.smartpath(outputdir)
        self.name = name

        try:
            self.DELPHES_DIR = os.environ["DELPHES_DIR"]
        except:
            self.log("WARNING: Did you forget to source 'setup.sh'??")
            self.log("QUITTING")
            sys.exit(1)

        try:
            import h5py
            from ROOT import TFile
            from ROOT.gSystem import Load as rtLoad
            from ROOT.gInterpreter import Declare as rtDeclare
            from ROOT.TMath import Pi
            rtLoad("{}/lib/libDelphes.so".format(self.DELPHES_DIR))
            rtDeclare('#include "{}/include/modules/Delphes.h"'.format(self.DELPHES_DIR))
            rtDeclare('#include "{}/include/classes/DelphesClasses.h"'.format(self.DELPHES_DIR))
            rtDeclare('#include "{}/include/classes/DelphesFactory.h"'.format(self.DELPHES_DIR))
            rtDeclare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(self.DELPHES_DIR))
        except:
            self.log(format_exc())
            self.log("QUITTING")
            sys.exit(1)


        self.filespec = self.check_for_default_file("filelist")
        self.spath = self.check_for_default_file("selection")

        with open(filespec) as f:
            self.inputfiles = [line.strip('\n').strip() for line in f.readlines()]

        # core tree, add files
        self.files = [TFile(f) for f in self.inputfiles]
        self.trees = [tf.Get("Delphes") for tf in self.files]
        self.sizes = [int(t.GetEntries()) for t in self.trees]
        self.nEvents = sum(self.sizes)

        self.jetDR = jetDR

        self.event_feature_names =  ['mJJ', 'j1Eta', 'j1Phi', 'j1Pt', 'j1M', 'j1E', 'j2Pt', 'j2M', 'j2E', 'DeltaEtaJJ', 'DeltaPhiJJ']
        self.jet_constituent_names = ['pEta', 'pPhi', 'pPt']
        self.n_constituent_particles=n_constituent_particles
        self.n_jets = 2
        self.jetvars = ['Pt', 'Eta', 'Phi', 'M', 'E']
        self.event_features = None
        self.jet_constituents = None

        spathout = os.path.join(self.outputdir, "{}_selection.txt".format(self.name))
        spathin = os.path.join(self.outputdir, "{}_selection.txt".format(self.name))
        spath = None
        if os.path.exists(spathout):
            spath = spathout
        elif os.path.exists(spathin):
            spath = spathin            
        else:
            raise AttributeError("Selection file does not exist in either input/output dirs!!")

        with open(spath) as f:
            self.selections = np.asarray(map(lambda x: map(long, x.split(',')), f.read().strip().split()))
        
        hlf_dict = {}
        particle_dict = {}

        self.selections_abs = np.asarray([sum(self.sizes[:s[0]]) + s[1] for s in self.selections])

    def check_for_default_file(
        self,
        file_pattern,
        suffix="txt"
    ):
        fname = "{}_{}.{}".format(self.name, file_pattern, suffix)
        spathout = os.path.join(self.outputdir, fname)
        spathin = os.path.join(self.inputdir, fname)
        if os.path.exists(spathout):
            return spathout
        elif os.path.exists(spathin):
            return spathin            
        else:
            raise AttributeError("Selection file does not exist in either input/output dirs!!")

    def log(
        self,
        msg
    ):
        if isinstance(msg, str):
            for line in msg.split('\n'):
                print self.LOGMSG + line
        else:
            print self.LOGMSG + str(msg)

    def convert(
        self,
        rng=(-1,-1),
    ):
        if rng[0] < 0 or rng[0] > self.nEvents:
            rng[0] = 0

        if rng[1] > self.nEvents or rng[1] < 0:
            rng[1] = self.nEvents
            
        selections_iter = self.selections[(self.selections_abs > nmin) & (self.selections_abs < nmax)]
        
        self.event_features = np.empty((len(selections_iter), len(self.event_feature_names)))
        self.jet_constituents = np.empty((len(selections_iter), self.n_jets, self.n_constituent_particles, len(self.jet_constituent_names)))

        ftn = 0
        # selection is implicit: looping only through total selectinos
        for count,(tree_n, i) in enumerate(selections_iter):
            
            self.log('tree {}, event {}'.format(tree_n, i))

            tree = self.trees[tree_n]
            tree.GetEntry(i)

            jets = [tree.Jet[jetn].P4() for jetn in range(self.n_jets)]

            for jetn,jet in enumerate(jets):

                # grab
                plist = np.concatenate([
                    self.get_jet_constituents(jet, self.jetDR, tree.EFlowTrack, 0.1, "PT"),
                    self.get_jet_constituents(jet, self.jetDR, tree.EFlowNeutralHadron, 0.5, "ET"),
                    self.get_jet_constituents(jet, self.jetDR, tree.EFlowPhoton, 0.2, "ET"),
                ], axis=0)
                
                # sort
                plist = plist[plist[:,2].argsort()][::-1][0:self.n_constituent_particles,:]

                # pad && add
                self.jet_constituents[count, jetn] = np.pad(plist, [(0, self.n_constituent_particles - plist.shape[0]),(0,0)], 'constant')

            self.event_features[count] = np.fromiter(self.get_jet_features(jets), float, count=len(self.event_feature_names))
            # HLF[count] = self.get_HLF(tree)
            # pvec.append(self.GetParticles(tree.EFlowTrack, "PT > 0.1", "PT"))
            # pvec.append(self.GetParticles(tree.EFlowNeutralHadron, "ET > 0.5", "ET"))
            # pvec.append(self.GetParticles(tree.EFlowPhoton, "ET > 0.2", "ET"))

            # particles.append(pvec)

        return self.jet_constituents, self.event_features
               
    def save(
        self,
        outputfile=None,
    ):
        outputfile = outputfile or os.path.join(self.outputdir, "{}_data.h5".format(self.name))

        if not outputfile.endswith(".h5"):
            outputfile += ".h5"
        
        f = h5py.File(outputfile, "w")
        f.create_dataset('event_feature_data', data=self.event_features)
        f.create_dataset('event_feature_names', data=self.event_feature_names)
        f.create_dataset('jet_constituent_data', data=self.jet_constituents)
        f.create_dataset('jet_constituent_names', data=self.jet_constituent_names)
        f.close()

    def get_jet_features(
        self,
        jets
    ):

        yield (jets[0] + jets[1]).M()       # Mjj
        yield jets[0].Eta()
        yield jets[0].Phi()
        for j in jets:
            yield j.Pt()
            yield j.M()
            yield j.E()
        yield jets[0].Eta() - jets[1].Eta()       # deltaeta
        yield jets[0].DeltaPhi(jets[1])           # deltaphi
        
    def get_jet_constituents(
        self,
        jet,
        dr,
        component,
        min_value,
        eType,
    ):
        selected = []
        for c in component:
            pt = getattr(c, eType)
            if pt > min_value:
                deltaEta = c.Eta - jet.Eta()
                deltaPhi = c.Phi - jet.Phi()
                deltaPhi = deltaPhi - 2*Pi()*(deltaPhi >  Pi()) + 2*Pi()*(deltaPhi < -1.*Pi())

                if deltaEta**2. + deltaPhi**2. < dr**2.:
                    selected.append([deltaEta, deltaPhi, pt])

        return np.asarray(selected)
 
    @staticmethod
    def smartpath(
        s
    ):
        if s.startswith('~'):
            return s
        return os.path.abspath(s)

if __name__ == "__main__":
    (_, inputdir, outputdir, name, dr, nc, rmin, rmax) = sys.argv
    core = Converter(inputdir, outputdir, name, float(dr), int(nc))
    ret = core.convert((int(rmin), int(rmax)))
    core.save()
