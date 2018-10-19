import collections
from SVJ.MakePlot.samples.signals import *
from SVJ.MakePlot.samples.QCD import *

samples_ref = collections.OrderedDict()
samples = collections.OrderedDict()
samples_mDM = collections.OrderedDict()
samples_rinv = collections.OrderedDict()
samples_alpha = collections.OrderedDict()
samples_QCD = collections.OrderedDict()
samples_all = collections.OrderedDict()

### Reference sample ###
samples_ref["SVJ_mZ1000_mDM20_rinv03_alpha02"] = SVJ_mZ1000_mDM20_rinv03_alpha02

### MZ variations ###
samples["SVJ_mZ1000_mDM20_rinv03_alpha02"] = SVJ_mZ1000_mDM20_rinv03_alpha02
samples["SVJ_mZ2000_mDM20_rinv03_alpha02"] = SVJ_mZ2000_mDM20_rinv03_alpha02
samples["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples["SVJ_mZ4000_mDM20_rinv03_alpha02"] = SVJ_mZ4000_mDM20_rinv03_alpha02

### mDM variations ###
samples_mDM["SVJ_mZ3000_mDM1_rinv03_alpha02"] = SVJ_mZ3000_mDM1_rinv03_alpha02
samples_mDM["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_mDM["SVJ_mZ3000_mDM50_rinv03_alpha02"] = SVJ_mZ3000_mDM50_rinv03_alpha02
samples_mDM["SVJ_mZ3000_mDM100_rinv03_alpha02"] = SVJ_mZ3000_mDM100_rinv03_alpha02

### rinv variations ###
samples_rinv["SVJ_mZ3000_mDM20_rinv01_alpha02"] = SVJ_mZ3000_mDM20_rinv01_alpha02
samples_rinv["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_rinv["SVJ_mZ3000_mDM20_rinv05_alpha02"] = SVJ_mZ3000_mDM20_rinv05_alpha02
samples_rinv["SVJ_mZ3000_mDM20_rinv07_alpha02"] = SVJ_mZ3000_mDM20_rinv07_alpha02

### alpha variations ###
samples_alpha["SVJ_mZ3000_mDM20_rinv03_alpha01"] = SVJ_mZ3000_mDM20_rinv03_alpha01
samples_alpha["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_alpha["SVJ_mZ3000_mDM20_rinv03_alpha05"] = SVJ_mZ3000_mDM20_rinv03_alpha05
samples_alpha["SVJ_mZ3000_mDM20_rinv03_alpha1"] = SVJ_mZ3000_mDM20_rinv03_alpha1

### samples all ###
samples_all["SVJ_mZ1000_mDM20_rinv03_alpha02"] = SVJ_mZ1000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ2000_mDM20_rinv03_alpha02"] = SVJ_mZ2000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ4000_mDM20_rinv03_alpha02"] = SVJ_mZ4000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM1_rinv03_alpha02"] = SVJ_mZ3000_mDM1_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM50_rinv03_alpha02"] = SVJ_mZ3000_mDM50_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM100_rinv03_alpha02"] = SVJ_mZ3000_mDM100_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv01_alpha02"] = SVJ_mZ3000_mDM20_rinv01_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv05_alpha02"] = SVJ_mZ3000_mDM20_rinv05_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv07_alpha02"] = SVJ_mZ3000_mDM20_rinv07_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha01"] = SVJ_mZ3000_mDM20_rinv03_alpha01
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha02"] = SVJ_mZ3000_mDM20_rinv03_alpha02
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha05"] = SVJ_mZ3000_mDM20_rinv03_alpha05
samples_all["SVJ_mZ3000_mDM20_rinv03_alpha1"] = SVJ_mZ3000_mDM20_rinv03_alpha1

### QCD samples ###
samples_QCD["QCD_Pt_80to120"]=QCD_Pt_80to120
samples_QCD["QCD_Pt_120to170"]=QCD_Pt_120to170
samples_QCD["QCD_Pt_170to300"]=QCD_Pt_170to300
samples_QCD["QCD_Pt_300to470"]=QCD_Pt_300to470
samples_QCD["QCD_Pt_470to600"]=QCD_Pt_470to600
samples_QCD["QCD_Pt_600to800"]=QCD_Pt_600to800
samples_QCD["QCD_Pt_800to1000"]=QCD_Pt_800to1000
samples_QCD["QCD_Pt_1000to1400"]=QCD_Pt_1000to1400
samples_QCD["QCD_Pt_1400to1800"]=QCD_Pt_1400to1800
samples_QCD["QCD_Pt_1800to2400"]=QCD_Pt_1800to2400
samples_QCD["QCD_Pt_2400to3200"]=QCD_Pt_2400to3200
samples_QCD["QCD_Pt_3200toInf"]=QCD_Pt_3200toInf
