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

class Converter:
    def __init__(
        self,
        inputdir,
        name,
        ffilter,
        ):

        self.inputdir = inputdir
        self.name = name
        if not ffilter.endswith('*.root'):
            ffilter += '*.root'
        self.inputfiles = glob.glob(os.path.join(inputdir, ffilter))

        # core tfile collection
        self.fc = rt.TFileCollection(name, name)
        for f in self.inputfiles:
            self.fc.Add(f)

        # core tree, add files
        self.tree = rt.TChain("Delphes")
        self.tree.AddFileInfoList(self.fc.GetList())
        self.nEvents = self.tree.GetEntries()

        
    def convert(
        self,
        rng=None
    ):
        if rng is None:
            rng = (0, self.nEvents)
        nmin, nmax = rng
        for i,event in enumerate(self.tree):
            if i < nmin:
                continue
            if i >= nmax:
                break

    def save(
        self,
        outputfile
    ):
        if not outputfile.endswith(".h5"):
            outputfile += ".h5"
        
        f = h5py.File(outputfile, "w")
        f.create_dataset('HLF', data=np.zeros(10), compression='gzip')
        f.create_dataset('HLF_')

def smartpath(s):
    if s.startswith('~'):
        return s
    return os.path.abspath(s)


parser = argparse.ArgumentParser()

parser.add_argument("inputdir", action="store", type=str, help="input dir path")
parser.add_argument("outputdir", action="store", type=str, help="output dir path")
parser.add_argument('-n', '--name', dest='name', action='store', default='sample', help='sample save name')
parser.add_argument('-f', '--filter', dest='filter', action='store', default='*', help='glob-style filter for root files in inputfile')


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

core = Converter(inputdir, args.name, args.filter)
core.convert(rng=(100,200))
    # core.save(outputdir)

def cmt():
    # def testmethod(n):
    #     f = rt.TFile("qcd_sqrtshatTeV_13TeV_PU20_1.root")
    #     tree = f.Get("Delphes")
    #     t0 = time.time()
    #     jpts = []
    #     for i,event in enumerate(tree):
    #         if i >= n:
    #             break
    #         subpts = []
    #         for j,jet in enumerate(event.Jet):
    #             subpts.append(jet.PT)
    #         jpts.append(subpts)
    #     return time.time() - t0, jpts

    # def test(n, nprime, test1, test2):
    #     t1=np.empty(n)
    #     t2=np.empty(n)
    #     d1=[]
    #     d2=[]
    #     for i in range(n):
    #         pt, pd = test1(nprime)
    #         t1[i] = pt
    #         d1.append(pd)
    #         pt, pd = test2(nprime)
    #         t2[i] = pt
    #         d2.append(pd)
    #         print("iteration {0}".format(i))

    #     # d1 = np.asarray(d1)
    #     # d2 = np.asarray(d2)
    #     return (t1, t2), (d1, d2)
    pass
                