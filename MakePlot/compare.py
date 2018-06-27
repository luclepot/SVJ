#!/bin/env python


import optparse

# Partse command line before ANY action
usage = 'usage: %prog -l lumi'
parser = optparse.OptionParser(usage)

parser.add_option('-l', '--lumi', dest='lumi', type='float', default = '27.2', help='Luminosity')
(opt, args) = parser.parse_args()


import time
import sys
import ROOT
import copy
import commands, os.path
import numpy as n
import tdrstyle, CMS_lumi
from utilities.services import Histo, Stack, Legend
from utilities.toPlot import *
from utilities.settings import *


ROOT.gROOT.Reset();
ROOT.gROOT.SetStyle('Plain')
ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

tdrstyle.setTDRStyle();

c = ROOT.TCanvas("c1","c1",600,600)
c.SetFillColor(0);
c.SetBorderMode(0);
c.SetFrameFillStyle(0);
#c1.SetFrameBorderMode(0);
#c1.SetLeftMargin( L/W );
#c1.SetRightMargin( R/W );
c.SetTopMargin( 1 );
c.SetBottomMargin(0);
c.SetTickx(0);
c.SetTicky(0);
c.cd()

pad = ROOT.TPad("pad", "pad", 0, 0 , 1, 1)
pad.SetBorderMode(0);
pad.SetTickx(0);
pad.SetTicky(0);
pad.Draw()
pad.cd()


path = "/mnt/t3nfs01/data01/shome/grauco/SVJAna/CMSSW_8_0_20/src/SVJ/SVJAnalysis/SkimmedLoose_ntuples/"


signals = [(1000, 20, 0.3, 0.2),(2000, 20, 0.3, 0.2),(3000, 20, 0.3, 0.2), (3000, 1, 0.3, 0.2), (3000, 50, 0.3, 0.2), (3000, 100, 0.3, 0.2), (3000, 20, 0.1, 0.2),(3000, 20, 0.5, 0.2),(3000, 20, 0.7, 0.2),(3000, 20, 0.3, 0.1), (3000, 20, 0.3, 0.5), (3000, 20, 0.3, 1), (4000, 20, 0.3, 0.2)]


def plot(var, samples, label=""):

    v = settings[var]
    stack = Stack(var, v[0])
    leg = Legend((0.40, 0.79, 0.94, 0.94))
    i = 0

    for sname, s in samples.items():
        mZ= s.mZ
        mDark = s.mDark
        rinv = s.rinv
        alpha = s.alpha
        
        filename = path + "tree_SVJ2_mZprime-%d_mDark-%d_rinv-%.1f_alpha-%.1f.root" %(mZ, mDark, rinv, alpha)
        if alpha == 1:     filename = path + "tree_SVJ2_mZprime-%d_mDark-%d_rinv-%.1f_alpha-%d.root" %(mZ, mDark, rinv, alpha)
        
        h = ROOT.TH1F(v[0], v[0],v[1], v[2], v[3])
    
        try:
            f = ROOT.TFile.Open(filename)
        except IOError:
            print "Cannot open ", filename
        else:
            #print "Opening file ",  filename
            t = f.Get("tree")
            nEvts = t.GetEntries()
            for entry  in t:
            
                h.Fill(getattr(t,var))

        h.SetMarkerColor(s.color)
        h.SetLineColor(s.color)
        h.SetLineWidth(2)

        pad.cd()
        if i==0: h.Draw()
        else:h.Draw("SAME")
        i = 1
        if(h.Integral()!=0.):h.Scale(1./h.Integral())
        hist = Histo.fromTH1(h)
        stack.Add(hist)
        leg.AddEntry(hist, s.leglabel, "l")
        c.Update()


    ROOT.gStyle.SetHistTopMargin(0.);
    stack.DrawStack(1, opt = "nostack")
    stack.SetMaximum(stack.GetMaximum("nostack")*1.2)
    stack.DrawStack(1, opt = "nostack")

    CMS_lumi.writeExtraText = 1
    CMS_lumi.extraText = ""
    lumi = 1
    if(lumi<1.):
        lumi = lumi*1000
        unit = " pb^{-1}"
    else: unit = " fb^{-1}"
    
    CMS_lumi.lumi_sqrtS = str(lumi)+ unit +" (13 TeV)" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)
    iPeriod = 0
    iPos = 11
    CMS_lumi.CMS_lumi(pad, iPeriod, iPos)

    
    leg.Draw("SAME")
    
    c.cd()
    c.Update();
    c.RedrawAxis();
    if(not os.path.isdir("plots")): os.system('mkdir plots')
    c.Print("plots/"+var +"_"+ label+".pdf")







samps = {"":samples, "mDM": samples_mDM, "rinv": samples_rinv, "alpha": samples_alpha}    


for label, samples in samps.iteritems():

    plot("girth", samples, label)
    plot("pt", samples, label)
    plot("axismajor", samples, label)
    plot("axisminor", samples, label)
    plot("tau21", samples, label)
    plot("tau32", samples, label)
    plot("mult", samples, label)
    plot("momenthalf", samples, label)
    plot("msd", samples, label)
    plot("deltaphi", samples, label)




