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

ROOT.gROOT.Reset();
ROOT.gROOT.SetStyle('Plain')
ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch()       #don't pop up canvanses
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)



def getBDT(filename):

    f = ROOT.TFile.Open(filename)
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
    return hBDT
    c.Print(filename+ "_BDT.pdf")


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

    return x,y


def getROC(effSig, rejBkg):

    c = ROOT.TCanvas()
    y  = array.array('f', rejBkg)
    x  = array.array('f', effSig)


    graph = ROOT.TGraph(len(x), x, y )
    c.cd()
    graph.Draw("AP")
    c.Print("graph.pdf")

    
h = getBDT("tree_QCD_Pt_470to600.root")
x, effBkg = getEffs(h, 0.01)
getROC(effBkg, effBkg)

