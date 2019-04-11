### Decorrelate Q2 syst among the processes

import sys
import os, commands
import shutil
import ROOT
import copy
from samples.toPlot import samples
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)
import optparse


usage = 'usage: %prog -d histosDir'
parser = optparse.OptionParser(usage)

parser.add_option('-d', '--dir', dest='path', type='string', default = 'output/histos_sys_2017/', help='Histos folder')
parser.add_option('', '--debug', dest='debug', type='int', default = '1', help='Increment level of verbosity for debugging purposes')

(opt, args) = parser.parse_args()

channels = ["cat"]

varsFh = {"h_Mt_presel":""}

def raiseError(histo, var, filename): 
    if not isinstance(histo, ROOT.TH1):
        raise RuntimeError('Failed to load histogram  %s from file  %s' % ( var, filename))

                
def applySys():
    syst = "Trigger"
    for s in samples.itervalues():
        for ch in channels:        
            nom = opt.path + s.label + "_"+ ch + ".root"
            f_nom = ROOT.TFile(nom)
        
            sysUp = opt.path + s.label + "_" + ch + "_" + syst + "Up.root"
            sysDown =opt.path + s.label + "_" + ch + "_" + syst + "Down.root"
            
            shutil.copyfile(nom, sysUp)
            shutil.copyfile(nom, sysDown)
            
            if(ch =="cat"):
                histos_up = []
                histos_down = []
                
                for  v, region in varsFh.iteritems():
                
                    h = f_nom.Get(v)
                    raiseError(h, v, nom)
                    
                    nh_up = copy.deepcopy(h)
                    nh_down = copy.deepcopy(h)
                    
                    nh_up.Scale(1.02)
                    histos_up.append(nh_up)
                    nh_down.Scale(0.98)
                    histos_down.append(nh_down)
                    
                nf_up = ROOT.TFile(sysUp, "UPDATE")
                nf_up.cd()
                [h.Write() for h in histos_up]
                nf_up.Close()
                
                nf_down = ROOT.TFile(sysDown, "UPDATE")
                nf_down.cd()
                [h.Write() for h in histos_down]
                nf_down.Close()
                
applySys()
