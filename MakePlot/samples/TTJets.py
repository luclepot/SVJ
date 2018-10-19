#from SVJ.MakePlot.samples.sample import *
from utils import *

TTJets = sample()
TTJets.files = outlist (d,"TTJets")
TTJets.sigma = 831.76
TTJets.color = 801
TTJets.style = 1
TTJets.fill = 1001
TTJets.leglabel = "t#bar{t}"
TTJets.label = "TTJets"
