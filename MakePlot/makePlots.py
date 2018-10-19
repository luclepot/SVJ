#!/Bin/env python

#########################################################
############ Macro for plots with ratio pad #############
#########################################################

import optparse

# Partse command line before ANY action
usage = 'usage: %prog -l lumi'
parser = optparse.OptionParser(usage)

parser.add_option('-l', '--lumi', dest='lumi', type='float', default = '1', help='Luminosity')
parser.add_option('-s', '--sys', dest='sys', type='string', default = 'noSys', help='Systematics: noSys, jesUp, jesDown')
parser.add_option('-n', '--normData', dest='normData', type='int', default = '0', help='Normalise to data?')
parser.add_option('-r', '--resdir', dest='resdir', type='string', default = '../SVJAnalysis', help='res directory')
parser.add_option('-d', '--no-data', dest='data', action='store_false', default = False, help='Hide data')
parser.add_option('', '--store', dest='store', action='store_true', default = False, help='Store')
parser.add_option('--focusOn', dest='focus', default = None, help='Focus on a single plot')
parser.add_option('-L', '--LogScale', dest='LogScale', default = True, help='Use Log Scale')

(opt, args) = parser.parse_args()

import time
import sys
import ROOT
import copy
import commands, os.path
import numpy as n

# from plots.services import Histo, Stack, Legend, deltaPhi, Histo1
#from plots.services import deltaPhi
from samples.toPlot import samples
#import plots.common, plots.fullhadronic, plots.semileptonic
import plots.SVJ

import tdrstyle, CMS_lumi

def sumerrors(h):
    return sum([h.GetBinError(ib) for ib in xrange(0,h.GetNbinsX()+2)])

def writeSummary(label, sys, h, lumi):
    if sys=="noSys":
        filename = outtxt + '/' + label + ".txt"
    else:
        filename = outtxt + '/' + label + "_" + sys + ".txt"

    ofile = open( filename, 'w')
    #print "Opening file: ", filename
    ofile.write("Lumi: %.1f\n"%(lumi))
    ofile.write("---> %s\n"%(label))


ROOT.gROOT.Reset();
ROOT.gROOT.SetStyle('Plain')
ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

tdrstyle.setTDRStyle();

settings = {}
store = []

#settings.update(plots.common.settings)
#store += plots.common.store
settings.update(plots.SVJ.settings)                                                                                                                   
store += plots.SVJ.store 

settings.update(plots.SVJ.settings)
store += plots.SVJ.store
             
outhistos = 'output/histos/'
outpdfs = 'output/pdfs/'
outtxt = 'output/txt/'


if opt.focus:
    #settings = { k:settings[k] for k in settings if k == opt.focus}
    if not settings:
        print 'Requested plot \''+opt.focus+'\'not found. Exiting'
        sys.exit(0)


# Create output direcotries
for d in [outhistos, outpdfs, outtxt]:
    if os.path.exists(d): continue
    print 'Creating',d
    os.makedirs(d)

# import sys
# sys.exit(0)

histoSig = {}


if opt.data:
    print 'Generating TDR-ratio plots'
    from plots.services import HistoRatio as Histo
    from plots.services import StackRatio as Stack
    from plots.services import LegendRatio as Legend
else:
    print 'Generating TDR plots'
    from plots.services import Histo, Stack, Legend


# for var, title in vars.iteritems():
for var,(title,scale,rebin, usrrng) in settings.iteritems():

    if(opt.store and not var.startswith("metFinal")): continue
    if not title:
        title = ''

    print "Variable: ",var
    print "Title: ", title

    #THStack for plotting

    stack_bkg = Stack(var,title)
    stack_sig = Stack(var,title)
    stack_data = Stack(var,title)
    stack_sh  = Stack(var,title)
    leg = Legend()
    leg_sh = Legend()
    leg_sign = ROOT.TLegend(0.7,0.7,0.9,0.9)  # 0.7,0.7,0.9,0.9)

    h1 = None

    for s in samples.itervalues():
        nEntries = 0
        # print "+ sample" , s.label
        if(hasattr(s, "components")):
            #samples with components
            histos = []
            notFound = []
            for c in s.components:
                #print "comp ", c.label

                if c.label.startswith("MET"): continue

                elif ((c.label.startswith("SingleMu") or c.label.startswith("SingleEl")) and (("CR3" not in var) and ("CR6" not in var) and ("CR7" not in var) and ("CRtt" not in var) and ("HT_2lep" not in var)) ):
                    #print "Fullhadronic channel, skipping dataset: " + c.label +" for variable " + var
                    continue

                elif ( c.label.startswith("MET") and (("CR3" in var) or ("CR6" in var) or ("CR7" in var) or ("CRtt" in var) or ("HT_2lep" in var)) ):
                    #print "Fullhadronic channel, skipping dataset: " + c.label +" for variable " + var
                    continue

                #print "c.label " , c.label

                if(opt.sys=="noSys"):
                    filename = opt.resdir+'/res/'+c.label+".root"
                elif(
                    (opt.sys=="jesUp" or opt.sys=="jesDown" or opt.sys=="jerUp" or opt.sys=="jerDown" or opt.sys=="pdf_totalUp" or opt.sys=="pdf_totalDown" or opt.sys=="q2Up" or opt.sys=="q2Down" or opt.sys=="pdf_zmUp" or opt.sys=="pdf_zmDown" or opt.sys=="pdf_asUp"  or opt.sys=="pdf_asDown") and (c.label=="JetHT_runG" or  c.label=="JetHT_runF" or c.label=="JetHT_runB" or c.label=="JetHT_runC" or c.label=="JetHT_runD" or c.label=="JetHT_runE" or c.label=="JetHT_runH2" or c.label=="JetHT_runH3")
                    ):
                    filename = opt.resdir+'/res/'+c.label+".root"
                else:
                    filename = opt.resdir+'/res/'+c.label+ "_"+ opt.sys +".root"

    
                #cutFlow histo saved only in nominal root file
#                filename_nEvt = opt.resdir+'/res_alph/'+c.label+"_" + opt.channel +".root"
#                infile_nEvt = ROOT.TFile.Open(filename_nEvt)
                infile_nEvt = ROOT.TFile.Open(filename)
                #print filename    
                #print infile_nEvt.Get("h_cutFlow").GetBinContent(1)
                
                if(c.label.startswith("QCDTT")): nEntries=1
                else: nEntries = infile_nEvt.Get("h_cutFlow").GetBinContent(1)

                infile = ROOT.TFile.Open(filename)
                # htmp = infile.Get(var)
                hin = infile.Get(var)
                if not isinstance(hin, ROOT.TH1):
                    notFound.append( (c.label,filename) )
                    # raise RuntimeError('Failed to load histogram %s from %s' % (var,filename))
                    continue

                htmp = hin.Clone()
                # print '--> Sums before scaling: w2:',htmp.GetSumw2().GetSize(),' I(h):',htmp.Integral(), ' E(h):', sumerrors(htmp)

                # Ensure the error structure is activated before doing anything else
                htmp.Sumw2()
                # Applu lumi scale if not data
                #if( nEntries != 0 and not(c.label.startswith("JetHT") or c.label.startswith("QCDTT"):
                if( nEntries != 0 and not(c.label.startswith("JetHT") )):
                    htmp.Scale((1./nEntries) * c.sigma * 1000.* float(opt.lumi) )
        
                # If a cutflow print a nice summary!
                if(var == "h_cutFlow"): writeSummary(c.label, opt.sys, htmp, opt.lumi)

                if(htmp.Integral()!=0):
                    htmp.SetMarkerStyle(20)
                    htmp.SetMarkerSize(1.2)
                    htmp.SetLineWidth(2)
                histos.append(htmp)

            if notFound:
                raise RuntimeError('Failed to retrieve %s' % notFound)

            # Sum components histos
            h1 = histos[0]
            [h1.Add(hi) for hi in histos[1:]]

            # h = Histo1(h1)
            # print 'h1 has sumw2', h1.GetSumw2().GetSize()

            h = Histo.fromTH1(h1)

        else:
            #sample does not have components
            #print "sample label ", s.label
            if opt.sys=="noSys":
                filename = opt.resdir+'/res/'+s.label+".root"
            else:
                filename = opt.resdir+'/res/'+s.label+"_" + opt.sys +".root"

            #cutFlow histo saved only in nominal root file
#            filename_nEvt = opt.resdir+'/res_alph /'+s.label+"_" + opt.channel +".root"
#            infile_nEvt = ROOT.TFile.Open(filename_nEvt)
            infile_nEvt = ROOT.TFile.Open(filename)
            #print infile_nEvt.Get("h_cutFlow").GetBinContent(1)
            if(s.label.startswith("QCDTT")): nEntries=1
            else: nEntries =  infile_nEvt.Get("h_cutFlow").GetBinContent(1)

            infile = ROOT.TFile.Open(filename)
            hin = infile.Get(var)
            if not isinstance(hin, ROOT.TH1):
                raise RuntimeError('Failed to load histogram %s from %s' % (filename, var))

            htmp = hin.Clone()


            # Ensure the error structure is activated before doing anything else
            htmp.Sumw2()

            htmp.SetMarkerStyle(20)
            htmp.SetMarkerSize(1.2)

            if(s.label.startswith("SVJ")):
               htmp.SetLineWidth(4)
             #  print "sample is", s.label
             #  print "before scaling", htmp.Integral()
            # Create the Histo wrapper
            h = Histo.fromTH1(htmp)

            scaleFactor = 1.

            #if (nEntries != 0  and not( s.label.startswith("JetHT") or s.label.startswith("QCDTT"))):
            if( nEntries != 0 and not(s.label.startswith("Data")  )):
                scaleFactor = 1./nEntries * s.sigma * 1000.* float(opt.lumi)
                htmp.Scale(scaleFactor)
                
            h.Scale(scaleFactor)

        ### Apply rebin
        # if(var in rebin.keys()):
        if rebin:
            if(not(s.label.startswith("QCDTT"))): h.Rebin(rebin)
  
        #if(s.label.startswith("Data")):

            #print "before shrinking"
            #if(var == "h_bprimemass_SRhm"): 
            #    print h.Integral()
                #print h.Integral(0,51)
                
          ### Re-range
        if usrrng is not None:
             uFirst, uLast = usrrng[0], usrrng[1]
             h.Shrink(uFirst, uLast)
      
        #if(s.label.startswith("Data")):
            
           # print "after shrinking"
            #if(var == "h_bprimemass_SRhm"): print h.Integral()
        ### Set style and create stack plots
        h.SetStyle(s.color, s.style, s.fill)

        # Writing outfile: only 8 bin for met  Plot, overflow bin included
        if opt.sys=="noSys":
            outfilename = "%s/%s.root" % (outhistos,s.label)
        else:
            outfilename = "%s/%s_%s.root" % (outhistos,s.label, opt.sys)

        outfile = ROOT.TFile(outfilename, "UPDATE")

        outfile.cd()
        if var in store:
            h.GetHisto().SetName(var)
            h.GetHisto().Write()
            #print s.label, " ", h.GetHisto().Integral()
        outfile.Close()

        #adding histo in bkg, sign and data categories
        if (s.label.startswith("SVJ")):

            stack_sig.Add(h)
            label = s.leglabel if scale == 1. else '%s x %d' % (s.leglabel,scale) #x%d
            leg_sign.SetMargin(0.15);
            leg_sign.AddEntry(h.GetHisto(), label , "l")
            h.Scale(scale) # make stack plots of bkground and signal x10 (for few variables)
        elif (s.label.startswith("Data")):
            stack_data.Add(h)
            if opt.data:
                leg.AddEntry(h, s.leglabel, "e0p,x0")
        elif ( not(s.label.startswith("SVJ") or s.label.startswith("Data"))) :
            #print s.label

            stack_bkg.Add(h)
            leg.AddEntry(h, s.leglabel, "f")
            i=0
            if(s.label=="QCD" ):
                #h_leg  = h.Clone("h_leg")
                h_leg = copy.deepcopy(h)
                h_leg.GetHisto().SetFillStyle(3154)
                h_leg.GetHisto().SetFillColor(ROOT.kGray)
                leg.AddEntry(h_leg, "#splitline{Stat uncertainties}{in the MC simulation}", "f")
                
            t = ROOT.TText(0.5,.75,"Stat uncertainties in the MC simulation");
            t.SetTextColor(ROOT.kBlack);
            t.SetTextSize(40);
            t.Draw();

            #box = ROOT.TBox(0.7,0.7,0.75,0.75);
            #box.SetFillStyle(3154);
            #box.SetLineColor(ROOT.kBlack);
            #box.SetLineWidth(2);
            #box.Draw();

        ### Make a summary
        if(var == "h_cutFlow"): writeSummary(s.label, opt.sys, h, opt.lumi)

        # make stack plot of histos normalized to unity

        h_norm = copy.deepcopy(h)
        h_norm.Normalize()
        leg_sh.AddEntry(h_norm, s.leglabel, "l")
        stack_sh.Add(h_norm)

        box = ROOT.TBox(0.7,0.7,0.75,0.75);
        box.SetFillStyle(3154);
        box.SetLineColor(ROOT.kBlack);
        box.SetLineWidth(2);
        #box.Draw();


    ### End of samples loop

    if opt.data:
        H=600
        W=700
    else:
        H=600
        W=700

    # H_ref = 600
    # W_ref = 700

    T = 0.08*H
    B = 0.12*H
    L = 0.12*W
    R = 0.08*W

    tdrstyle.setTDRStyle();

    # Updatethe legend's stype
    leg_sign.SetNColumns(2)
    leg_sign.SetFillColor(0)
    leg_sign.SetFillStyle(0)
    leg_sign.SetTextFont(42)
    leg_sign.SetBorderSize(0)


    box = ROOT.TBox(0.7,0.7,0.75,0.75);
    box.SetFillStyle(3154);
    box.SetLineColor(ROOT.kBlack);
    box.SetLineWidth(2);
    #box.Draw();

    ### Adjust ranges and create/save plots: some settings are defined in tdrstyle.py and service_ratio.py
    c1 = ROOT.TCanvas("c1","c1",50,50,W,H)
    c1.SetFillColor(0);
    c1.SetBorderMode(0);
    c1.SetFrameFillStyle(0);
    c1.SetFrameBorderMode(0);
    c1.SetLeftMargin( L/W );
    c1.SetRightMargin( R/W );
    c1.SetTopMargin( 1 );
    c1.SetBottomMargin(0);
    c1.SetTickx(0);
    c1.SetTicky(0);
    c1.cd()

    if opt.data:
        pad1= ROOT.TPad("pad1", "pad1", 0, 0.30 , 1, 1)
        pad1.SetTopMargin(0.1)
        pad1.SetBottomMargin(0)
        pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.05)


        # print 'x1ndc',leg_sign.GetX1NDC(), leg_sign.GetX1()
        leg_sign.SetTextSize(0.06)

        leg_sign.SetX1(.27)#27
        leg_sign.SetY1(.47)#59
        leg_sign.SetX2(.94)#91
        leg_sign.SetY2(.72)

    else:
        pad1= ROOT.TPad("pad1", "pad1", 0, 0 , 1, 1)

        leg_sign.SetTextSize(0.032)

        leg_sign.SetX1(.40)
        leg_sign.SetY1(.69)
        leg_sign.SetX2(.94)
        leg_sign.SetY2(.84)

  

    box = ROOT.TBox(0.7,0.7,0.75,0.75);
    box.SetFillStyle(3154);
    box.SetLineColor(ROOT.kBlack);
    box.SetLineWidth(2);
    #box.Draw();    

    pad1.SetBorderMode(0);

    if opt.LogScale==True:
        pad1.SetLogy()
    #pad1.SetLogy()
    pad1.SetTickx(0);
    pad1.SetTicky(0);
    pad1.Draw()
    pad1.cd()

    # Normalizing to data
    if(opt.normData>0):
        h_data_ = stack_data._hs.GetStack().Last()
        nData = h_data_.Integral()
        h_bkg_ = stack_bkg._hs.GetStack().Last()
        nBkg = h_bkg_.Integral()
        if nBkg>0: sf_norm = nData/nBkg
        else: sf_norm = 1

        for i in xrange(stack_bkg._hs.GetNhists()):
            h_i=stack_bkg._hs.GetHists().At(i)
            h_i.Scale(sf_norm)
            #print h_i.GetBinContent(1)
            
        #stack_bkg._hs.Modified()
        #stack_data.GetStack().At(stack_data.GetNhists()-1)
 #       print " NORMALIZING TO DATA"
        h_bkg_.Scale(sf_norm)

    #Set Range user for a selection of plot: use same format for other variables if needed
    if usrrng is not None:
        stack_bkg.SetRangeUser( usrrng[0], usrrng[1] )
        stack_sig.SetRangeUser( usrrng[0], usrrng[1] )
        stack_data.SetRangeUser( usrrng[0], usrrng[1] )
        stack_sh.SetRangeUser( usrrng[0], usrrng[1] )

    h_data = stack_data.GetLast()
    h_bkg = stack_bkg.GetLast()

    maximum = max([stack_bkg.GetMaximum(),stack_data.GetMaximum(), h_data.GetMaximum(), stack_sig.GetMaximum()])
    minimum = min([stack_bkg.GetMinimum(),stack_data.GetMinimum(), h_data.GetMinimum(), stack_sig.GetMinimum()])

    #Maximum for log scale
    if opt.LogScale==True :
        stack_bkg.SetMaximum(maximum*( 500000 if opt.data else 10000000000)) 
    else:
        print 'log scale not set'
        stack_bkg.SetMaximum(maximum*1.5)
        
        #stack_bkg.SetMaximum(maximum*( 5000 if opt.data else 10000))
   # stack_bkg.SetMaximum(100000000000)
    if(minimum > 0):
        stack_bkg.SetMinimum(0.0004*minimum)
        #stack_bkg.SetMinimum(1)
    else:
        #stack_bkg.SetMinimum(0.01)
        stack_bkg.SetMinimum(0.1)  
        
    #Drawing THStacks

    box = ROOT.TBox(0.7,0.7,0.75,0.75);
    box.SetFillStyle(3154);
    box.SetLineColor(ROOT.kBlack);
    box.SetLineWidth(2);
    #box.Draw();


    stack_bkg.DrawStack(lumi = opt.lumi)
    #stack_sig.DrawStack(lumi = opt.lumi, opt ="nostack")
    stack_sig.DrawStack(lumi = opt.lumi, opt = "nostack, SAME")
    stack_sig.Draw("axis same"); 

    ROOT.gStyle.SetHatchesSpacing(2)
    ROOT.gStyle.SetHatchesLineWidth(2)

    h_err = h_bkg.Clone("h_err")
    h_err.SetLineWidth(100)
    h_err.SetFillStyle(3154)
    h_err.SetMarkerSize(0)
    h_err.SetFillColor(ROOT.kGray+2)
    h_err.Draw("e2same")


    if opt.data:
        # h_data.Sumw2()
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerSize(1.2)
        h_data.Draw("e0SAMEp,x0")
        #h_data.Draw("PE0same,X0")  
        stack_bkg.GetHistogram().GetXaxis().SetLabelOffset(0.90)
        stack_bkg.GetHistogram().GetXaxis().SetLabelSize(1.00)
    leg.Draw("SAME")
    leg_sign.Draw("same")

    #***
    if opt.data:
        c1.cd()
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.03, 1, 0.29)
        pad2.SetTopMargin(0.05)#4
        pad2.SetBottomMargin(0.36)
        pad2.SetLeftMargin(0.12)
        pad2.SetRightMargin(0.05)

        c1.cd()
        pad2.Draw()
        pad2.cd()

        ratio  = h_data.Clone("ratio")
        ratio.SetLineColor(ROOT.kBlack)
        ratio.SetMinimum(0.1)
        ratio.SetMaximum(3.0)
        ratio.Sumw2()
        ratio.SetStats(0)

        if usrrng is not None:
            ratio.GetXaxis().SetRangeUser( usrrng[0], usrrng[1] )

        denom = h_bkg.Clone("denom")
        denom.Sumw2()
        
        if(denom.Integral() !=0):
            ratio.Divide(denom)
            ratio.SetMarkerStyle(20)
            ratio.SetMarkerSize(1.2)
            ratio.Draw("e0SAMEp,x0")
            ratio.SetTitle("")


        h_bkg_err = h_bkg.Clone("h_err")
        h_bkg_err.Reset()
        h_bkg_err.Sumw2()

        for i in range(0,h_bkg.GetNbinsX()):
            h_bkg_err.SetBinContent(i,1)
            if(h_bkg.GetBinContent(i)):
                h_bkg_err.SetBinError(i, (h_bkg.GetBinError(i)/h_bkg.GetBinContent(i)))
            else:
                h_bkg_err.SetBinError(i, 0)
        h_bkg_err = h_bkg.Clone("h_bkg_err")
        h_bkg_err.Sumw2()
        h_bkg_err.Divide(h_bkg);
        h_bkg_err.Draw("E2same");
        #h_bkg_err.SetMaximum(2.);
        #h_bkg_err.SetMinimum(0);
        h_bkg_err.SetLineWidth(100)
        h_bkg_err.SetFillStyle(3154)
        h_bkg_err.SetMarkerSize(0)
        h_bkg_err.SetFillColor(ROOT.kGray+2)

        f1 = ROOT.TF1("myfunc","[0]",-100000,10000);
        f1.SetLineColor(ROOT.kBlack)
        f1.SetLineStyle(ROOT.kDashed)
        f1.SetParameter(0,1);
        f1.Draw("same")

        #some settings for histo are defined in tdrstyle.py and service_ratio.py
        ratio.Draw("e0SAMEp,x0")
        ratio.Sumw2()
        ratio.GetYaxis().SetTitle("Data / Bkg")
        ratio.GetYaxis().SetNdivisions(503)

        ratio.GetXaxis().SetLabelFont(42);
        ratio.GetYaxis().SetLabelFont(42);
        ratio.GetXaxis().SetTitleFont(42);
        ratio.GetYaxis().SetTitleFont(42);

        ratio.GetXaxis().SetTitleOffset(0.96);
        ratio.GetYaxis().SetTitleOffset(0.30);#0.31

        ratio.GetXaxis().SetLabelSize(0.18);#15
        ratio.GetYaxis().SetLabelSize(0.15);#12
        ratio.GetXaxis().SetTitleSize(0.18);#18
        ratio.GetYaxis().SetTitleSize(0.18);#16

        ratio.GetYaxis().SetRangeUser(0.5,1.5);#-2.3,3.7
        
        if(var=="h_nsubj"): 
            ratio.GetYaxis().SetRangeUser(0.9,1.1);
            ratio.GetXaxis().SetTitleOffset(0.95);
            ratio.GetXaxis().SetLabelSize(0.25);#15  
            
        ratio.GetXaxis().SetTitle(title)
        ratio.GetXaxis().SetLabelOffset(0.04) #0.06
        ratio.GetYaxis().SetLabelOffset(0.015)

        if(var=="h_nsubj"): 
            ratio.GetXaxis().SetLabelOffset(0.015);

    CMS_lumi.writeExtraText = 1
    CMS_lumi.extraText = "Work in Progress"
    lumi = opt.lumi
    if(lumi<1.):
        lumi = lumi*1000
        unit = " pb^{-1}"
    else: unit = " fb^{-1}"

    box = ROOT.TBox(0.7,0.7,0.75,0.75);
    box.SetFillStyle(3154);
    box.SetLineColor(ROOT.kBlack);
    box.SetLineWidth(2);
    box.Draw();
    

    #CMS_lumi.lumi_sqrtS = str(lumi)+ unit +" (13 TeV)" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)
    CMS_lumi.lumi_sqrtS = "(13 TeV)"

    iPeriod = 0
    iPos = 11

    # writing the lumi information and the CMS "logo"
    # Ratio Check HERE
    CMS_lumi.CMS_lumi(pad1, iPeriod, iPos)

    #gPad.RedrawAxis();

    c1.cd()
    c1.Update();
    c1.RedrawAxis();
    #c1.GetFrame().Draw();

    pdfbasename = var
 
    syst_label = '' if opt.sys == 'noSys' else ('_'+opt.sys)
    data_label = '' if opt.data else '_nodata'

    pdfname = outpdfs+'/'+pdfbasename+syst_label+data_label+'.pdf'
    rootname = outpdfs+'/'+pdfbasename+syst_label+data_label+'.root'
    pngname = outpdfs+'/'+pdfbasename+syst_label+data_label+'.png'

    if opt.sys == 'noSys':
        c1.Print(pdfname)
        c1.Print(rootname)
        c1.Print(pngname)

    c1.Clear() 
