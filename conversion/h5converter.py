import numpy as np
import math
import os
import argparse
import sys
import time
from traceback import format_exc
import h5py
import ROOT as rt
from collections import OrderedDict as odict

DELPHES_DIR = os.environ["DELPHES_DIR"]
rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))

class Converter:

    LOGMSG = "Converter :: "

    def __init__(
        self,
        outputdir,
        filespec,
        spath,
        name,
        jetDR=0.8,
        n_constituent_particles=100,
        save_constituents=False,
    ):
        self.outputdir = outputdir

        self.name = name

        self.filespec = filespec
        self.spath = spath

        with open(filespec) as f:
            self.inputfiles = [line.strip('\n').strip() for line in f.readlines()]

        # core tree, add files
        self.files = [rt.TFile(f) for f in self.inputfiles]
        self.trees = [tf.Get("Delphes") for tf in self.files if tf.GetListOfKeys().Contains("Delphes")]
        self.sizes = [int(t.GetEntries()) for t in self.trees]
        self.nEvents = sum(self.sizes)

        self.jetDR = jetDR

        ## ADD MET
        self.event_feature_names =  [
            'j1Eta',
            'j1Phi',
            'j1Pt',
            'j1M',
            'j1E',
            'j2Pt',
            'j2M',
            'j2E',
            'DeltaEtaJJ',
            'DeltaPhiJJ',
            'mult',
            'ptd',
            'axis2'
        ]
        self.jet_constituent_names = ['pEta', 'pPhi', 'pPt']

        self.EFlow_types = [
            [ "EFlowTrack", 0.1, "PT" ],
            [ "EFlowNeutralHadron", 0.5, "ET" ],
            [ "EFlowPhoton", 0.2, "ET" ]
        ]

        self.EFlow_dict = odict()
        for i, eft in enumerate(self.EFlow_types):
            self.EFlow_dict[eft[0]] = i

        self.n_constituent_particles=n_constituent_particles
        self.n_jets = 2
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

        self.save_constituents = save_constituents
        self.selections_abs = np.asarray([sum(self.sizes[:s[0]]) + s[1] for s in self.selections])

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
        rng=(-1,-1)
    ):
        rng = list(rng)
        if rng[0] < 0 or rng[0] > self.nEvents:
            rng[0] = 0

        if rng[1] > self.nEvents or rng[1] < 0:
            rng[1] = self.nEvents

        nmin, nmax = rng
            
        selections_iter = self.selections[(self.selections_abs > nmin) & (self.selections_abs < nmax)]
        self.log("selecting on range {0}".format(rng))
        self.event_features = np.empty((len(selections_iter), len(self.event_feature_names)))
        self.log("event feature shapes: {}".format(self.event_features.shape))
        self.jet_constituents = np.empty((len(selections_iter), self.n_jets, self.n_constituent_particles, len(self.jet_constituent_names)))
        self.log("jet constituent shapes: {}".format(self.jet_constituents.shape))
            
        if not self.save_constituents:
            self.log("ignoring jet constituents")
        
        # self.log("selections: {}".format(selections_iter))

        ftn = 0

        # selection is implicit: looping only through total selectinos
        for count,(tree_n, i) in enumerate(selections_iter):
            
            self.log('tree {}, event {}'.format(tree_n, i))

            tree = self.trees[tree_n]
            tree.GetEntry(i)

            jets = [tree.Jet[jetn].P4() for jetn in range(self.n_jets)]
            
            track_index = -np.ones((len(jets), self.n_constituent_particles, 2), dtype=int)

            for jetn, jet in enumerate(jets):
                # grab

                plist = np.empty((0, len(self.jet_constituent_names)))
                subindex = np.empty((0, 2))
                for flow_type, pt_min, pt_type in self.EFlow_types:
                    new, indicies = self.get_jet_constituents(jet, self.jetDR, getattr(tree, flow_type), pt_min, pt_type, flow_type)
                    plist = np.concatenate([plist, new], axis=0)
                    subindex = np.concatenate([subindex, indicies], axis=0)
            
                # plist = np.concatenate([
                #     self.get_jet_constituents(jet, self.jetDR, tree.EFlowTrack, 0.1, "PT"),
                #     self.get_jet_constituents(jet, self.jetDR, tree.EFlowNeutralHadron, 0.5, "ET"),
                #     self.get_jet_constituents(jet, self.jetDR, tree.EFlowPhoton, 0.2, "ET"),
                # ], axis=0)
                
                # sort
                sort_index = plist[:,2].argsort()[::-1]

                plist = plist[sort_index][0:self.n_constituent_particles,:]
                subindex = subindex[sort_index][0:self.n_constituent_particles,:]

                # pad && add
                self.jet_constituents[count, jetn] = np.pad(plist, [(0, self.n_constituent_particles - plist.shape[0]),(0,0)], 'constant')
                track_index[jetn,:len(subindex)] = subindex

            self.event_features[count] = np.fromiter(self.get_jet_features(tree.Jet), float, count=len(self.event_feature_names))
            return tree, track_index
            
        if self.save_constituents:
            return self.jet_constituents, self.event_features

        return self.event_features

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
        if self.save_constituents:
            f.create_dataset('jet_constituent_data', data=self.jet_constituents)
            f.create_dataset('jet_constituent_names', data=self.jet_constituent_names)
        f.close()

    def get_jet_features(
        self,
        jets
    ):
        j1,j2 = jets[0].P4(), jets[1].P4()
        # yield (j1 + j2).M()       # Mjj
        yield j1.Eta()
        yield j1.Phi()
        yield j1.Pt()
        yield j1.M()
        yield j1.E()
        yield j2.Pt()
        yield j2.M()
        yield j2.E()
        yield j1.Eta() - j2.Eta()       # deltaeta
        yield j1.DeltaPhi(j2)           # deltaphi
        for value in self.get_jet_values():
            yield value
        yield 0 # ptd
        yield 0 # axis2

    # def get_mult(self, jet):

    def get_jet_values(
        self,
    ):

        return 1.0, 2.0, 3.0
        
    def get_jet_constituents(
        self,
        jet,
        dr,
        component,
        min_value,
        eType,
        flow_type
    ):
        rep = self.EFlow_dict[flow_type]
        selected = []
        indicies = []
        for i,c in enumerate(component):
            pt = getattr(c, eType)
            if pt > min_value:
                deltaEta = c.Eta - jet.Eta()
                deltaPhi = c.Phi - jet.Phi()
                deltaPhi = deltaPhi - 2*np.pi*(deltaPhi >  np.pi) + 2*np.pi*(deltaPhi < -1.*np.pi)

                if deltaEta**2. + deltaPhi**2. < dr**2.:
                    selected.append([deltaEta, deltaPhi, pt])
                    indicies.append([i, rep])
        
        if len(selected) == 0:
            return np.zeros((0,3))
        return np.asarray(selected), np.asarray(indicies)


if __name__ == "__main__":
    if len(sys.argv) == 10:
        (_, outputdir, filespec, pathspec, name, dr, nc, rmin, rmax, constituents) = sys.argv
        core = Converter(outputdir, filespec, pathspec, name, float(dr), int(nc), bool(int(constituents)))
        ret = core.convert((int(rmin), int(rmax)))
        core.save()
    # elif len(sys.argv) == 0:
    else:
        print "TEST MODE"
        try: 
            print tree
            print track_index.shape
        except:
            print "dang"
            core = Converter("../data/hlfSVJ", "../data/hlfSVJ/0.0_filelist.txt", "../data/hlfSVJ/0.0_selection.txt","0.0")
            tree,track_index = core.convert((0, 2000))


        def jets_axis2_pt2(
            track_index,
            tree,
            n_jets=2,
        ):
            for jetn in range(n_jets):
                mult = 0
                sum_weight = 0
                sum_pt = 0
                sum_deta = 0
                sum_dphi = 0
                sum_deta2 = 0
                sum_detadphi = 0 
                sum_dphi2 = 0

                jet = tree.Jet[jetn].P4()
                for i, eft in track_index[jetn]:
                    if i < 0:
                        break
                    c = getattr(tree, core.EFlow_types[eft][0])[i].P4()
                    if eft == 0 and c.Pt() > 1.0:
                        # means that the particle has charge, increase jet multiplicity
                        mult += 1

                    deta = c.Eta() - jet.Eta()
                    dphi = c.DeltaPhi(jet)
                    cpt = c.Pt()
                    weight = cpt*cpt

                    sum_weight += weight
                    sum_pt += cpt
                    sum_deta += deta*weight
                    sum_dphi += dphi*weight
                    sum_deta2 += deta*deta*weight
                    sum_detadphi += deta*dphi*weight
                    sum_dphi2 += dphi*dphi*weight

                a,b,c,ave_deta,ave_dphi,ave_deta2,ave_dphi2=0,0,0,0,0,0,0

                if sum_weight > 0:
                    ave_deta = sum_deta/sum_weight
                    ave_dphi = sum_dphi/sum_weight
                    ave_deta2 = sum_deta2/sum_weight
                    ave_dphi2 = sum_dphi2/sum_weight
                    a = ave_deta2 - ave_deta*ave_deta                                                    
                    b = ave_dphi2 - ave_dphi*ave_dphi                                                    
                    c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi)
                
                delta = np.sqrt(np.abs((a - b)*(a - b) + 4*c*c))
                axis2 = np.sqrt(0.5*(a+b-delta)) if a + b - delta > 0 else 0
                ptD = np.sqrt(sum_weight)/sum_pt if sum_weight > 0 else 0
                yield [mult, ptD, axis2]
