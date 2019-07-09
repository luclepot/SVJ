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

def get_data_dict(list_of_selections):
    ret = {}
    for sel in list_of_selections:
        with open(sel, 'r') as f:
            data = map(lambda x: x.strip('\n'), f.readlines())
        for elt in data:
            key, raw = elt.split(': ')
            ret[key] = map(int, raw.split())
    return ret

class Converter:

    LOGMSG = "Converter :: "

    def __init__(
        self,
        outputdir,
        spath,
        name,
        jetDR=0.8,
        n_constituent_particles=100,
        save_constituents=False,
    ):
        self.outputdir = outputdir

        self.name = name

        self.spath = spath

        self.selections = get_data_dict([spath])
        self.inputfiles = list(self.selections.keys())

        # core tree, add files, add all trees
        self.files = [rf for rf in [rt.TFile(f) for f in self.inputfiles] if rf.GetListOfKeys().Contains("Delphes")]
        self.trees = [tf.Get("Delphes") for tf in self.files]
        self.sizes = [int(t.GetEntries()) for t in self.trees]
        self.nEvents = sum(self.sizes)

        self.log("Found {0} files".format(len(self.files)))
        self.log("Found {0} delphes trees".format(len(self.trees)))
        self.log("Found {0} total events".format(self.nEvents))

        self.jetDR = jetDR

        self.n_jets = 2
        self.jet_base_names = [
            'Eta',
            'Phi',
            'Pt',
            'M',
            'ChargedFraction',
            'PTD',
            'Axis2',
            'Flavor',
            'Energy',
            'DeltaEta',
            'DeltaPhi',
            'MET',
            'METEta',
            'METPhi',
            # 'MT',
        ]

        ## ADD MET
        self.event_feature_names =  []

        for i in range(self.n_jets):
            self.event_feature_names += ['j{}'.format(i) + j for j in self.jet_base_names]
        
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
        self.event_features = None
        self.jet_constituents = None

        # spathout = os.path.join(self.outputdir, "{}_selection.txt".format(self.name))
        # spathin = os.path.join(self.outputdir, "{}_selection.txt".format(self.name))
        # spath = None
        # if os.path.exists(spathout):
        #     spath = spathout
        # elif os.path.exists(spathin):
        #     spath = spathin            
        # else:
        #     raise AttributeError("Selection file does not exist in either input/output dirs!!")

        hlf_dict = {}
        particle_dict = {}

        self.save_constituents = save_constituents
        # self.selections_abs = np.asarray([sum(self.sizes[:s[0]]) + s[1] for s in self.selections])
        self.log("found {0} selected events, out of a total of {1}".format(sum(map(len, self.selections.values())), self.nEvents))

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
        return_debug_tree=False
    ):
        rng = list(rng)

        gmin, gmax = min(self.sizes), max(self.sizes)

        if rng[0] < 0 or rng[0] > gmax:
            rng[0] = 0

        if rng[1] > gmax or rng[1] < 0:
            rng[1] = gmax

        nmin, nmax = rng
        selections_iter = self.selections.copy()
        for k,v in selections_iter.items():
            v = np.asarray(v).astype(int)
            selections_iter[k] = v[(v > nmin) & (v < nmax)]

        total_size = sum(map(len, selections_iter.values()))
        total_count = 0

        self.log("selecting on range {0}".format(rng))
        self.event_features = np.empty((total_size, len(self.event_feature_names)))
        self.log("event feature shapes: {}".format(self.event_features.shape))
        self.jet_constituents = np.empty((total_size, self.n_jets, self.n_constituent_particles, len(self.jet_constituent_names)))
        self.log("jet constituent shapes: {}".format(self.jet_constituents.shape))
            
        if not self.save_constituents:
            self.log("ignoring jet constituents")
        
        # self.log("selections: {}".format(selections_iter))

        ftn = 0

        # selection is implicit: looping only through total selectinos
        for tree_n,tree_name in enumerate(self.inputfiles):
            for event_n,event_index in enumerate(selections_iter[tree_name]):
                self.log('tree {0}, event {1}, index {2}'.format(tree_n, event_n, event_index))

                tree = self.trees[tree_n]
                tree.GetEntry(event_index)

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
                    self.jet_constituents[total_count, jetn] = np.pad(plist, [(0, self.n_constituent_particles - plist.shape[0]),(0,0)], 'constant')
                    track_index[jetn,:len(subindex)] = subindex

                self.event_features[total_count] = np.fromiter(self.get_jet_features(tree, track_index), float, count=len(self.event_feature_names))
                
                if return_debug_tree:
                    return tree, self.event_features[total_count]
                
                total_count += 1 

        if self.save_constituents:
            return self.jet_constituents, self.event_features

        return self.event_features

    def save(
        self,
        outputfile=None,
    ):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        outputfile = outputfile or os.path.join(self.outputdir, "{}_data.h5".format(self.name))

        if not outputfile.endswith(".h5"):
            outputfile += ".h5"
        
        # self.log(os.path.exists(outputfile))
        # self.log(os.path.exists(os.path.dirname(outputfile)))
        # self.log(outputfile)
        f = h5py.File(outputfile, "w")
        f.create_dataset('event_feature_data', data=self.event_features)
        f.create_dataset('event_feature_names', data=self.event_feature_names)
        if self.save_constituents:
            f.create_dataset('jet_constituent_data', data=self.jet_constituents)
            f.create_dataset('jet_constituent_names', data=self.jet_constituent_names)
        f.close()

    def get_jet_features(
        self,
        tree,
        track_index,
    ):
        # j1,j2 = tree.Jet[0].P4(), tree.Jet[1].P4()
        
        ## leading 2 jets
        for i in range(self.n_jets):
            j = tree.Jet[i].P4()
            yield j.Eta()
            yield j.Phi()
            yield j.Pt()
            yield j.M()

            nc, nn = map(float, [tree.Jet[i].NCharged, tree.Jet[i].NNeutrals])
            n_total = nc + nn
            yield nc / n_total if n_total > 0 else -1
        
            for value in self.jets_axis2_pt2(j, tree, track_index[i]):
                yield value

            yield tree.Jet[i].Flavor
            yield j.E()
            yield tree.Jet[i].DeltaEta
            yield tree.Jet[i].DeltaPhi
            yield tree.MissingET[0].MET
            yield tree.MissingET[0].Eta
            yield tree.MissingET[0].Phi

    def jets_axis2_pt2(
        self,
        jet,
        tree,
        track_index
    ):
        ret = np.empty((self.n_jets,3))
        # mult = 0
        sum_weight = 0
        sum_pt = 0
        sum_deta = 0
        sum_dphi = 0
        sum_deta2 = 0
        sum_detadphi = 0 
        sum_dphi2 = 0
        
        for i, eft in track_index:
            if i < 0:
                break
            c = getattr(tree, core.EFlow_types[eft][0])[i].P4()
            # if eft == 0 and c.Pt() > 1.0:
            #     # means that the particle has charge, increase jet multiplicity
            #     mult += 1

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
        return ptD, axis2
        
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
            return np.zeros((0,3)), -np.ones((0,2))
        return np.asarray(selected), np.asarray(indicies)

if __name__ == "__main__":
    if len(sys.argv) == 9:
        (_, outputdir, pathspec, name, dr, nc, rmin, rmax, constituents) = sys.argv
        print outputdir
        print pathspec
        # print filespec
        core = Converter(outputdir, pathspec, name, float(dr), int(nc), bool(int(constituents)))
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
            core = Converter(".", "../data/tightSVJ/s10/data_0_selection.txt", "data")
            ret = core.convert((0, 500), return_debug_tree=True)
