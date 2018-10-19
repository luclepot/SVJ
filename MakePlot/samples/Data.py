######################################
#
# Deborah Pinna, August 2015
#
######################################


from utils import *

JetHT_runB = sample()
JetHT_runB.files = outlist (d,"JetHT_runB")
JetHT_runB.skimEff = 1
JetHT_runB.sigma = 1
JetHT_runB.color = ROOT.kBlack
JetHT_runB.jpref = jetLabel 
JetHT_runB.jp = jetLabel
JetHT_runB.style = 1
JetHT_runB.fill = 1001
JetHT_runB.leglabel = "JetHT_runB"
JetHT_runB.label = "JetHT_runB"

JetHT_runG = sample()
JetHT_runG.files = outlist (d,"JetHT_runG")
JetHT_runG.skimEff = 1
JetHT_runG.sigma = 1
JetHT_runG.color = ROOT.kBlack
JetHT_runG.jpref = jetLabel
JetHT_runG.jp = jetLabel
JetHT_runG.style = 1
JetHT_runG.fill = 1001
JetHT_runG.leglabel = "JetHT_runG"
JetHT_runG.label = "JetHT_runG"

JetHT_runC = sample()
JetHT_runC.files = outlist (d,"JetHT_runC")
JetHT_runC.skimEff = 1
JetHT_runC.sigma = 1
JetHT_runC.color = ROOT.kBlack
JetHT_runC.jpref = jetLabel
JetHT_runC.jp = jetLabel
JetHT_runC.style = 1
JetHT_runC.fill = 1001
JetHT_runC.leglabel = "JetHT_runC"
JetHT_runC.label = "JetHT_runC"

JetHT_runD = sample()
JetHT_runD.files = outlist (d,"JetHT_runD")
JetHT_runD.skimEff = 1
JetHT_runD.sigma = 1
JetHT_runD.color = ROOT.kBlack
JetHT_runD.jpref = jetLabel
JetHT_runD.jp = jetLabel
JetHT_runD.style = 1
JetHT_runD.fill = 1001
JetHT_runD.leglabel = "JetHT_runD"
JetHT_runD.label = "JetHT_runD"

JetHT_runE = sample()
JetHT_runE.files = outlist (d,"JetHT_runE")
JetHT_runE.skimEff = 1
JetHT_runE.sigma = 1
JetHT_runE.color = ROOT.kBlack
JetHT_runE.jpref = jetLabel
JetHT_runE.jp = jetLabel
JetHT_runE.style = 1
JetHT_runE.fill = 1001
JetHT_runE.leglabel = "JetHT_runE"
JetHT_runE.label = "JetHT_runE"

JetHT_runF = sample()
JetHT_runF.files = outlist (d,"JetHT_runF")
JetHT_runF.skimEff = 1
JetHT_runF.sigma = 1
JetHT_runF.color = ROOT.kBlack
JetHT_runF.jpref = jetLabel
JetHT_runF.jp = jetLabel
JetHT_runF.style = 1
JetHT_runF.fill = 1001
JetHT_runF.leglabel = "JetHT_runF"
JetHT_runF.label = "JetHT_runF"

JetHTg = sample()
JetHTg.files = outlist (d,"JetHTg")
JetHTg.skimEff = 1
JetHTg.sigma = 1
JetHTg.color = ROOT.kBlack
JetHTg.jpref = jetLabel
JetHTg.jp = jetLabel
JetHTg.style = 1
JetHTg.fill = 1001
JetHTg.leglabel = "JetHTg"
JetHTg.label = "JetHTg"

JetHT_runH2 = sample()
JetHT_runH2.files = outlist (d,"JetHT_runH2")
JetHT_runH2.skimEff = 1
JetHT_runH2.sigma = 1
JetHT_runH2.color = ROOT.kBlack
JetHT_runH2.jpref = jetLabel
JetHT_runH2.jp = jetLabel
JetHT_runH2.style = 1
JetHT_runH2.fill = 1001
JetHT_runH2.leglabel = "JetHT_runH2"
JetHT_runH2.label = "JetHT_runH2"


JetHT_runH3 = sample()
JetHT_runH3.files = outlist (d,"JetHT_runH3")
JetHT_runH3.skimEff = 1
JetHT_runH3.sigma = 1
JetHT_runH3.color = ROOT.kBlack
JetHT_runH3.jpref = jetLabel
JetHT_runH3.jp = jetLabel
JetHT_runH3.style = 1
JetHT_runH3.fill = 1001
JetHT_runH3.leglabel = "JetHT_runH3"
JetHT_runH3.label = "JetHT_runH3"


Data = sample()
Data.color = ROOT.kBlack
Data.style = 1
Data.fill = 1001
Data.leglabel = "Data"
Data.label = "Data"
Data.components = [JetHT_runB]
#Data.components = [JetHT_runB, JetHT_runC, JetHT_runD, JetHT_runE, JetHT_runF, JetHT_runH2 , JetHT_runH3, JetHT_runG ]


