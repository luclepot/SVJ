import numpy as np
import h5py
import ROOT as rt
import math
import os
import argparse
import sys
import time
import glob

try:
    DELPHES_DIR = os.environ["DELPHES_DIR"]
except:
    print("WARNING: Did you forget to source 'setup.sh'??")
    sys.exit(0)

class Converter:

    LOGMSG = "Converter :: "

    def __init__(
        self,
        inputdir,
        outputdir,
        name,
        ffilter,
        jetDR=0.8,
    ):

        self.inputdir = inputdir
        self.outputdir = outputdir
        self.name = name

        if not ffilter.endswith('*.root'):
            ffilter += '*.root'

        self.inputfiles = glob.glob(os.path.join(inputdir, ffilter))

        # core tree, add files
        self.files = [rt.TFile(f) for f in self.inputfiles]
        self.trees = [tf.Get("Delphes") for tf in self.files]
        self.sizes = [int(t.GetEntries()) for t in self.trees]
        self.nEvents = sum(self.sizes)
        self.jetDR = jetDR

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
        rng=None,
        n_constituent_particles=100, 
    ):
        if rng is None:
            rng = (0, self.nEvents)
        nmin, nmax = rng

        selections_iter = self.selections[(self.selections_abs > nmin) & (self.selections_abs < nmax)]
        # HLF = np.empty((len(selections_iter),) + self.get_HLF().shape)

        jet_constituents = np.empty((len(selections_iter), 2, n_constituent_particles, 3))
        # selection is implicit: looping only through total selectinos
        for count,(tree_n, i) in enumerate(selections_iter):
            
            self.log('tree {}, event {}'.format(tree_n, i))

            tree = self.trees[tree_n]
            tree.GetEntry(i)

            for jetn in range(2):
                jet = tree.Jet[jetn].P4()

                # grab
                plist = np.concatenate([
                    self.get_jet_constituents(jet, self.jetDR, tree.EFlowTrack, 0.1, "PT"),
                    self.get_jet_constituents(jet, self.jetDR, tree.EFlowNeutralHadron, 0.5, "ET"),
                    self.get_jet_constituents(jet, self.jetDR, tree.EFlowPhoton, 0.2, "ET"),
                ], axis=0)
                
                # sort
                plist = plist[plist[:,2].argsort()][::-1][0:n_constituent_particles,:]

                # pad && add
                toadd= np.pad(plist, [(0, n_constituent_particles - plist.shape[0]),(0,0)], 'constant')
                jet_constituents[count, jetn] = toadd

            # HLF[count] = self.get_HLF(tree)
            # pvec.append(self.GetParticles(tree.EFlowTrack, "PT > 0.1", "PT"))
            # pvec.append(self.GetParticles(tree.EFlowNeutralHadron, "ET > 0.5", "ET"))
            # pvec.append(self.GetParticles(tree.EFlowPhoton, "ET > 0.2", "ET"))

            # particles.append(pvec)

        return jet_constituents

    def get_jet_constituents(
        self,
        jet,
        dr,
        component,
        min_value,
        eType,
    ):
        pi = rt.TMath.Pi()
        selected = []
        for c in component:
            pt = getattr(c, eType)
            if pt > min_value:
                deltaEta = c.Eta - jet.Eta()
                deltaPhi = c.Phi - jet.Phi()
                deltaPhi = deltaPhi - 2*pi*(deltaPhi >  pi) + 2*pi*(deltaPhi < -1.*pi)

                if deltaEta**2. + deltaPhi**2. < dr**2.:
                    selected.append([deltaEta, deltaPhi, pt])

        return np.asarray(selected)

    # def get_HLF(
    #     self,
    #     tree=None
    # ):
    #     ret = np.empty((2,4))
    #     if tree is None:
    #         return ret
            
    #     for i,jet in enumerate(tree.Jet):
    #         if i >= 2:
    #             break

    #         ret[i] = jet.PT, jet.Phi, jet.Eta, jet.Mass

    #     return ret

    def PtMap(
        self,
        vector_to_loop,
        req,
        eType,
    ):
        ptmap = []
        for h in vector_to_loop:
            if eval('h.' + req):
                ptmap.append([h.Eta, h.Phi, eval('h.' + eType)])
        return np.asarray(ptmap)
                
    def save(
        self,
        outputfile,
        outputarray,
    ):
        if not outputfile.endswith(".h5"):
            outputfile += ".h5"
        
        f = h5py.File(outputfile, "w")
        f.create_dataset('HLF', data=np.zeros(10), compression='gzip')

def smartpath(s):
    if s.startswith('~'):
        return s
    return os.path.abspath(s)

if __name__ == "__main__":


    def range_input(s):
        try:
            return tuple(map(int, s.strip().strip(')').strip('(').split(',')))
        except:
            raise argparse.ArgumentTypeError("-r input not in format: int,int")

    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir", action="store", type=str, help="input dir path")
    parser.add_argument("outputdir", action="store", type=str, help="output dir path")
    parser.add_argument('-n', '--name', dest='name', action='store', default='sample', help='sample save name')
    parser.add_argument('-f', '--filter', dest='filter', action='store', default='*', help='glob-style filter for root files in inputfile')
    parser.add_argument('-r', '--range', dest='range', action='store', type=range_input, default=None)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:])

    inputdir = smartpath(args.inputdir) 
    outputdir = smartpath(args.outputdir)

    rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))

    core = Converter(inputdir, outputdir, args.name, args.filter)
    ret = core.convert(rng=args.range)
    # core.save(outputdir)
