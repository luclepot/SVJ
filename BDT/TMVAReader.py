import optparse
import array

# Partse command line before ANY action
usage = 'usage: %prog -l lumi'
parser = optparse.OptionParser(usage)

import ROOT

parser = optparse.OptionParser(usage)

parser.add_option('-l', '--lumi', dest='lumi', type='float', default = '27.2', help='Luminosity')
(opt, args) = parser.parse_args()

import time
import sys
import ROOT
import copy
import commands, os.path
import numpy as n
from SVJ.MakePlot.samples.QCD import QCD
from SVJ.MakePlot.samples.signals import *

ROOT.gROOT.Reset();
ROOT.gROOT.SetStyle('Plain')
ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch()       #don't pop up canvanses
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)



def getSingleBDT(filename, label):

      try:
        f = ROOT.TFile.Open(filename)
      except IOError:
        print "Cannot open ", filename
      else:
        #print "Opening file ",  filename
        t = f.Get("tree")
        reader = ROOT.TMVA.Reader()
      
        mult = array.array('f',[0]) 
        girth = array.array('f',[0]) 
        tau21 = array.array('f',[0]) 
        tau32 = array.array('f',[0]) 
        axisminor = array.array('f',[0]) 
        deltaphi = array.array('f',[0]) 
        msd = array.array('f',[0])
        pt = array.array('f',[0]) 
      
        reader.AddVariable("mult", mult)
        reader.AddVariable("axisminor", axisminor)
        reader.AddVariable("girth", girth)
        reader.AddVariable("tau21", tau21)
        reader.AddVariable("tau32", tau32)
        reader.AddVariable("msd", msd)
        reader.AddVariable("deltaphi", deltaphi)
        reader.AddSpectator("pt", pt)
        
        weightsfile = "weights/TMVAClassification_BDTG.weights.xml"
        reader.BookMVA("BDTG", weightsfile)
        nentries = t.GetEntries()
    
        t.SetBranchAddress("axisminor",axisminor)
        t.SetBranchAddress("girth",girth)
        t.SetBranchAddress("tau21",tau21)
        t.SetBranchAddress("tau32",tau32)
        t.SetBranchAddress("msd",msd)
        t.SetBranchAddress("msd",msd)
        t.SetBranchAddress("deltaphi",deltaphi)
        t.SetBranchAddress("pt",pt)
      
        c = ROOT.TCanvas()

        hBDT = ROOT.TH1F("BDT", "BDT", 100, -1.0, 1.0)
        for i in t:    
          BDT = reader.EvaluateMVA("BDTG")
          #print "BDT value: ", BDT
          hBDT.Fill(BDT)


        hBDT.Draw()
        if(not os.path.isdir("plots")): os.system('mkdir plots')
        c.Print("plots"+label+"_BDT.pdf")
        return hBDT



def getBDT(path, sample):
    filename = path + "tree_" + sample.label + ".root"
    print "FILENAME: ",filename 
    hBDT = getSingleBDT(filename, sample.label)
    return hBDT

    
def getCompositeBDT(path, sample):

    hBDT = ROOT.TH1F("BDT", "BDT", 100, -1.0, 1.0)

    for s in sample.components:
        h = getBDT(path, s)
        sf = s.sigma/h.Integral()
        hBDT.Add(h, sf)

    return hBDT

    
def getEffs(hBDT, step):
    min = hBDT.GetXaxis().GetBinLowEdge(hBDT.GetXaxis().GetFirst())
    print "First ", min
    max = hBDT.GetXaxis().GetBinUpEdge(hBDT.GetXaxis().GetLast())
    print "Last ", max
    nBins = hBDT.GetNbinsX()
    x = []
    eff = 0.
    y = []
    x.append(-1.)
    y.append(0.) ## rejection 1-eff
    nPoints = int((max-min)/step)
    x_ = min
    while x_ < max:
        x_ = x_+step
        x.append(x_)
        eff = hBDT.Integral(hBDT.GetXaxis().FindBin(x_), nBins)/ hBDT.Integral()
        print "Eff: ", eff
        y.append(1. - eff)

    return y


def getROC(effSig, rejBkg):

    c = ROOT.TCanvas()
    y  = array.array('f', rejBkg)
    x  = array.array('f', effSig)


    graph = ROOT.TGraph(len(x), x, y )
    c.cd()
    graph.Draw("AP")
    if(not os.path.isdir("plots")): os.system('mkdir plots')
    c.Print("plots/graph.pdf")





path = "/mnt/t3nfs01/data01/shome/grauco/SVJAna/CMSSW_8_0_20/src/BDT/Ntuple/"            
#h = getBDT(path, QCD_Pt_470to600)
h = getBDT(path,SVJ_mZ3000_mDM1_rinv03_alpha02)
effBkg = getEffs(h, 0.01)
getROC(effBkg, effBkg)








