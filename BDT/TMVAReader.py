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
from SVJ.MakePlot.samples.QCD import *
from SVJ.MakePlot.samples.signals import *
from SVJ.MakePlot.utilities.services import Histo, Stack, Legend

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
        flatweight = array.array('f',[0]) 
      
        reader.AddVariable("mult", mult)
        reader.AddVariable("axisminor", axisminor)
        reader.AddVariable("girth", girth)
        reader.AddVariable("tau21", tau21)
        reader.AddVariable("tau32", tau32)
        reader.AddVariable("msd", msd)
        reader.AddVariable("deltaphi", deltaphi)
        reader.AddSpectator("pt", pt)
        
        weightsfile = "/mnt/t3nfs01/data01/shome/grauco/SVJAna/CMSSW_8_0_20/src/BDT/weights/TMVAClassification_BDTG.weights.xml"
        reader.BookMVA("BDTG", weightsfile)
        nentries = t.GetEntries()
        c = ROOT.TCanvas()

        hBDT = ROOT.TH1F("BDT", label+" BDT", 100, -1.0, 1.0)
        hBDT.SetTitle(label+ " BDT")
        hBDT.GetXaxis().SetTitle("BDTG response")
        hBDT.GetYaxis().SetTitle("Entries")
        hBDT.SetLineColor(ROOT.kRed)

        for entry in t:
            mult[0] = entry.mult
            girth[0] = entry.girth
            tau21[0] = entry.tau21
            tau32[0] = entry.tau32
            msd[0] = entry.msd
            axisminor[0] = entry.axisminor
            deltaphi[0] = entry.deltaphi
            pt[0] = entry.pt
            flatweight[0] = entry.flatweight
            #print "flatweight: ", flatweight[0]
            BDT = reader.EvaluateMVA("BDTG")
            hBDT.Fill(BDT, flatweight[0])

        hBDT.Draw("histo")
        if(not os.path.isdir("plots")): os.system('mkdir plots')
        c.Print("plots/"+label+"_BDT.pdf")
        return hBDT


def getBDT(path, sample):
    filename = path + "tree_" + sample.label + ".root"
    print "FILENAME: ",filename 
    hBDT = getSingleBDT(filename, sample.label)
    fout =  ROOT.TFile(sample.label + "Out.root", "UPDATE")
    fout.cd()
    hBDT.Write()
    fout.Close()
    return hBDT

    
def getCompositeBDT(path, sample):

    hBDT = ROOT.TH1F("BDT", "BDT", 100, -1.0, 1.0)

    for s in sample.components:
        print s
        h = getBDT(path, s)
        print "Sample: ", s.label
        print "Sigma: ", s.sigma
        print "N of events: ", h.Integral()
        # sf = s.sigma/h.Integral()
        hBDT.Add(h)

    fout =  ROOT.TFile(sample.label + "Out.root", "UPDATE")
    fout.cd()
    hBDT.Write()
    fout.Close()
    c = ROOT.TCanvas()
    hBDT.Draw("HIST")
    c.Print("plots/QCD.pdf")

    return hBDT

    
def getEffs(hBDT, step):
    min = hBDT.GetXaxis().GetBinLowEdge(hBDT.GetXaxis().GetFirst())
    max = hBDT.GetXaxis().GetBinUpEdge(hBDT.GetXaxis().GetLast())

    nBins = hBDT.GetNbinsX()
    x = []
    eff = 0.
    y = []
    x.append(-1.)
    y.append(0.)
    nPoints = int((max-min)/step)
    x_ = min
    while x_ < max:
        x_ = x_+step
        x.append(x_)
        eff = hBDT.Integral(hBDT.GetXaxis().FindBin(x_), nBins)/ hBDT.Integral()
        y.append(eff)

    return y


def getROC(effSig, effBkg):

    c = ROOT.TCanvas("", "", 800, 800)

    rejBkg = [1-i for i in effBkg]
    y  = array.array('f', rejBkg)
    x  = array.array('f', effSig)
    
    leg = Legend((0.60, 0.6, 0.94, 0.94))
    graph = ROOT.TGraph(len(x), x, y )
    c.cd()

    graph.Draw("AP")
    graph.SetTitle("")
    graph.GetYaxis().SetTitle("Background rejection")
    graph.GetXaxis().SetTitle("Signal efficiency")

    fraction = graph.Integral()
    leg.AddEntryGraph(graph, ', area = {0:.3f}'.format(fraction), 'lp')
    leg.Draw()

    c.Update()

    if(not os.path.isdir("plots")): os.system('mkdir plots')
    c.Print("plots/graph.pdf")


path = "/mnt/t3nfs01/data01/shome/grauco/SVJAna/CMSSW_8_0_20/src/BDT/Ntuple/"            


h = getBDT(path,SVJ_mZ3000_mDM20_rinv03_alpha02)

#hb = getCompositeBDT(path, QCD)

hb = getBDT(path, QCD_Pt_80to120)



#fsig =  ROOT.TFile(SVJ_mZ3000_mDM1_rinv03_alpha02.label+"Out.root")
#h = fsig.Get("BDT")
#fsig.Close()
effSig = getEffs(h, 0.001)

#fbkg =  ROOT.TFile(QCD.label+"Out.root")
#hb = fbkg.Get("BDT")
rejBkg = getEffs(hb, 0.001)


getROC(effSig, rejBkg)









