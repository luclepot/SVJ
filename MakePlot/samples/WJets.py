######################################
#
# Annapaola de Cosa, January 2015
#
######################################

from utils import *

WJetsToLNu_HT100to200 = sample()
WJetsToLNu_HT100to200.files = outlist (d,"WJetsToLNu_HT-100to200")
WJetsToLNu_HT100to200.skimEff = 1.21
WJetsToLNu_HT100to200.sigma = 1627.45
WJetsToLNu_HT100to200.jpref = jetLabel 
WJetsToLNu_HT100to200.jp = jetLabel
WJetsToLNu_HT100to200.color = ROOT.kYellow -7
WJetsToLNu_HT100to200.style = 1
WJetsToLNu_HT100to200.fill = 1001
WJetsToLNu_HT100to200.leglabel = "WJetsHT"
WJetsToLNu_HT100to200.label = "WJetsToLNu_HT100to200"

WJetsToLNu_HT200to400 = sample()
WJetsToLNu_HT200to400.files = outlist (d,"WJetsToLNu_HT-200to400")
WJetsToLNu_HT200to400.skimEff = 1.21
WJetsToLNu_HT200to400.sigma = 435.24
WJetsToLNu_HT200to400.jpref = jetLabel 
WJetsToLNu_HT200to400.jp = jetLabel
WJetsToLNu_HT200to400.color = ROOT.kYellow -7
WJetsToLNu_HT200to400.style = 1
WJetsToLNu_HT200to400.fill = 1001
WJetsToLNu_HT200to400.leglabel = "WJetsHT"
WJetsToLNu_HT200to400.label = "WJetsToLNu_HT200to400"

WJetsToLNu_HT400to600 = sample()
WJetsToLNu_HT400to600.files = outlist (d,"WJetsToLNu_HT-400to600")
WJetsToLNu_HT400to600.skimEff = 1.21
WJetsToLNu_HT400to600.sigma = 59.18
WJetsToLNu_HT400to600.jpref = jetLabel 
WJetsToLNu_HT400to600.jp = jetLabel
WJetsToLNu_HT400to600.color = ROOT.kYellow -7
WJetsToLNu_HT400to600.style = 1
WJetsToLNu_HT400to600.fill = 1001
WJetsToLNu_HT400to600.leglabel = "WJetsHT"
WJetsToLNu_HT400to600.label = "WJetsToLNu_HT400to600"

WJetsToLNu_HT600to800 = sample()
WJetsToLNu_HT600to800.files = outlist (d,"WJetsToLNu_HT-600to800")
WJetsToLNu_HT600to800.skimEff = 1.21
WJetsToLNu_HT600to800.sigma = 14.58
WJetsToLNu_HT600to800.jpref = jetLabel 
WJetsToLNu_HT600to800.jp = jetLabel
WJetsToLNu_HT600to800.color = ROOT.kYellow -7
WJetsToLNu_HT600to800.style = 1
WJetsToLNu_HT600to800.fill = 1001
WJetsToLNu_HT600to800.leglabel = "WJetsHT"
WJetsToLNu_HT600to800.label = "WJetsToLNu_HT600to800"

WJetsToLNu_HT800to1200 = sample()
WJetsToLNu_HT800to1200.files = outlist (d,"WJetsToLNu_HT-800to1200")
WJetsToLNu_HT800to1200.skimEff = 1.21
WJetsToLNu_HT800to1200.sigma = 6.66
WJetsToLNu_HT800to1200.jpref = jetLabel 
WJetsToLNu_HT800to1200.jp = jetLabel
WJetsToLNu_HT800to1200.color = ROOT.kYellow -7
WJetsToLNu_HT800to1200.style = 1
WJetsToLNu_HT800to1200.fill = 1001
WJetsToLNu_HT800to1200.leglabel = "WJetsHT"
WJetsToLNu_HT800to1200.label = "WJetsToLNu_HT800to1200"

WJetsToLNu_HT1200to2500 = sample()
WJetsToLNu_HT1200to2500.files = outlist (d,"WJetsToLNu_HT-1200to2500")
WJetsToLNu_HT1200to2500.skimEff = 1.21
WJetsToLNu_HT1200to2500.sigma = 1.608
WJetsToLNu_HT1200to2500.jpref = jetLabel 
WJetsToLNu_HT1200to2500.jp = jetLabel
WJetsToLNu_HT1200to2500.color = ROOT.kYellow -7
WJetsToLNu_HT1200to2500.style = 1
WJetsToLNu_HT1200to2500.fill = 1001
WJetsToLNu_HT1200to2500.leglabel = "WJetsHT"
WJetsToLNu_HT1200to2500.label = "WJetsToLNu_HT1200to2500"

WJetsToLNu_HT2500toInf = sample()
WJetsToLNu_HT2500toInf.files = outlist (d,"WJetsToLNu_HT-2500toInf")
WJetsToLNu_HT2500toInf.skimEff = 1.21
WJetsToLNu_HT2500toInf.sigma = 0.03891
WJetsToLNu_HT2500toInf.jpref = jetLabel 
WJetsToLNu_HT2500toInf.jp = jetLabel
WJetsToLNu_HT2500toInf.color = ROOT.kYellow -7
WJetsToLNu_HT2500toInf.style = 1
WJetsToLNu_HT2500toInf.fill = 1001
WJetsToLNu_HT2500toInf.leglabel = "WJetsHT"
WJetsToLNu_HT2500toInf.label = "WJetsToLNu_HT2500toInf"

WJets = sample()
WJets.color = 881
WJets.style = 1
WJets.fill = 1001
WJets.leglabel = "W + jets"
WJets.label = "WJets"
WJets.components = [WJetsToLNu_HT100to200, WJetsToLNu_HT200to400, WJetsToLNu_HT400to600, WJetsToLNu_HT600to800, WJetsToLNu_HT800to1200,  WJetsToLNu_HT1200to2500, WJetsToLNu_HT2500toInf]
#WJets.components = [WJetsToLNu_HT100to200]





