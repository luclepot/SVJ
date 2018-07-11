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
import SVJ.MakePlot.tdrstyle as tdrstyle
import SVJ.MakePlot.CMS_lumi as CMS_lumi
from  SVJ.MakePlot.utilities.services import Histo, Stack, Legend
from  SVJ.MakePlot.utilities.toPlot import *
from  SVJ.MakePlot.utilities.settings import *


ROOT.gROOT.Reset();
ROOT.gROOT.SetStyle('Plain')
ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

tdrstyle.setTDRStyle()

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
    leg = Legend((0.40, 0.65, 0.94, 0.94))
    i = 0
    if(label == "SVJ"):
        print "I am looking for a signal"
        print samples
    for sname, s in samples.items():
        filename = path
        if(s.label.startswith("SVJ")):
            mZ= s.mZ
            mDark = s.mDark
            rinv = s.rinv
            alpha = s.alpha
        
            filename = filename + "tree_SVJ2_mZprime-%d_mDark-%d_rinv-%.1f_alpha-%.1f.root" %(mZ, mDark, rinv, alpha)
            if alpha == 1:     filename = filename + "tree_SVJ2_mZprime-%d_mDark-%d_rinv-%.1f_alpha-%d.root" %(mZ, mDark, rinv, alpha)
        else:
            filename = path + "tree_" + s.label + ".root"

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


        if(label == "SVJ"): print "Integral: ", h.Integral()
        h.SetMarkerColor(s.color)
        h.SetLineColor(s.color)
        h.SetLineWidth(2)

        #pad.cd()
        if i==0: h.Draw()
        else:h.Draw("SAME")
        i = 1
        if(h.Integral()!=0.):h.Scale(1./h.Integral())
        hist = Histo.fromTH1(h)
        stack.Add(hist)
        leg.AddEntry(hist, s.leglabel, "l")
        c.Update()


    ROOT.gStyle.SetHistTopMargin(0.);
    #stack.DrawStack(1, opt = "nostack")
    stack.SetMaximum(stack.GetMaximum("nostack")*1.2)
    #stack.DrawStack(1, opt = "nostack")

    ## CMS_lumi.writeExtraText = 1
    ## CMS_lumi.extraText = ""
    ## lumi = 1
    ## if(lumi<1.):
    ##     lumi = lumi*1000
    ##     unit = " pb^{-1}"
    ## else: unit = " fb^{-1}"
    
    ## CMS_lumi.lumi_sqrtS = str(lumi)+ unit +" (13 TeV)" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)
    ## iPeriod = 0
    ## iPos = 11
    ## CMS_lumi.CMS_lumi(pad, iPeriod, iPos)

    
    #leg.Draw("SAME")
    
    c.cd()
    c.Update();
    c.RedrawAxis();
    if(not os.path.isdir("plots")): os.system('mkdir plots')
    c.Print("plots/"+var +"_"+ label+".pdf")
    return stack, leg


def compareSamples(var, samples_ref, label_ref, samples, label):

    stack_ref, leg_ref = plot(var, samples_ref, label_ref)
    stack, leg = plot(var, samples, label)

    stack.SetMaximum(stack.GetMaximum("nostack")*1.2)
    stack.DrawStack(1, opt = "nostack")
    leg.AddEntry(stack_ref.GetHistogram(), label_ref, "l" )
    print stack_ref.GetHistogram()
    print "is TStack? ", isinstance(stack_ref, Stack)
    stack_ref.DrawStack(1, opt = "nostack, SAME")
    leg.Draw("SAME")
    if(not os.path.isdir("plots")): os.system('mkdir plots')
    c.Print("plots/"+var +".pdf")

    
    




#samps = {"":samples, "mDM": samples_mDM, "rinv": samples_rinv, "alpha": samples_alpha}    

samps = {"QCD":samples_QCD}
samps_ref = {"SVJ":samples_ref}


## for label, samples in samps.iteritems():

##     plot("girth", samples, label)
##     plot("pt", samples, label)
##     plot("axismajor", samples, label)
##     plot("axisminor", samples, label)
##     plot("tau21", samples, label)
##     plot("tau32", samples, label)
##     plot("mult", samples, label)
##     plot("momenthalf", samples, label)
##     plot("msd", samples, label)
##     plot("deltaphi", samples, label)


for label, samples in samps.iteritems():

    compareSamples("girth", samples_ref, "SVJ", samples, label)
    compareSamples("pt", samples_ref, "SVJ", samples, label)
    compareSamples("axismajor", samples_ref, "SVJ", samples, label)
    compareSamples("axisminor", samples_ref, "SVJ", samples, label)
    compareSamples("tau21", samples_ref, "SVJ", samples, label)
    compareSamples("tau32", samples_ref, "SVJ", samples, label)
    compareSamples("mult", samples_ref, "SVJ", samples, label)
    compareSamples("momenthalf", samples_ref, "SVJ", samples, label)
    compareSamples("msd", samples_ref, "SVJ", samples, label)
    compareSamples("deltaphi", samples_ref, "SVJ", samples, label)




