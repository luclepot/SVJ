import numpy as np
import h5py
import ROOT as rt
import math
import os
import argparse
import sys
import time

DELPHES_DIR = os.environ["DELPHES_DIR"]

rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))

class Converter:
    def __init__(
        self,
        inputfile
        ):

        if not inputfile.endswith(".root"):
            inputfile += ".root"
        # core tfile, ttree
        self.f = rt.TFile.Open(inputfile)    
        self.tree = self.f.Delphes
        
    def convert(
        self,
        n,
    ):
        for i,event in enumerate(self.tree):
            if i >= n:
                break
            subpts = []
            for j,jet in enumerate(event.Jet):
                subpts.append(jet.PT)
            jpts.append(subpts)

    def save(
        self,
        outputfile
    ):
        if not outputfile.endswith(".h5"):
            outputfile += ".h5"
        
        f = h5py.File(outputfile, "w")
        f.create_dataset('HLF', data=np.zeros(10), compression='gzip')
        f.create_dataset('HLF_')

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("inputfile", action="store", type=str, help="input file path")
    parser.add_argument("outputfile", action="store", type=str, help="output file path")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:])

    # core = Converter(args.inputfile)
    # core.convert()
    # core.save(args.outputfile)
    # core.testloop(3)


def testmethod(n):
    f = rt.TFile("qcd_sqrtshatTeV_13TeV_PU20_1.root")
    tree = f.Get("Delphes")
    t0 = time.time()
    jpts = []
    for i,event in enumerate(tree):
        if i >= n:
            break
        subpts = []
        for j,jet in enumerate(event.Jet):
            subpts.append(jet.PT)
        jpts.append(subpts)
    return time.time() - t0, jpts

def test(n, nprime, test1, test2):
    t1=np.empty(n)
    t2=np.empty(n)
    d1=[]
    d2=[]
    for i in range(n):
        pt, pd = test1(nprime)
        t1[i] = pt
        d1.append(pd)
        pt, pd = test2(nprime)
        t2[i] = pt
        d2.append(pd)
        print("iteration {0}".format(i))

    # d1 = np.asarray(d1)
    # d2 = np.asarray(d2)
    return (t1, t2), (d1, d2)

            