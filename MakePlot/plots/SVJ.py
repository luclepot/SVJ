met_range = (0,2000, True)

# title,    scale,  rebin, usrrng
settings = {
    'h_cutFlow'             : ('', 10, None, None),
    
    'h_AK8jetsmult_nosel' : ('Jets multiplicity', 1, None, None),
    'h_looseMuonsmult_nosel' : ('Muons multiplicity', 1, None, None),
    'h_looseElectronsmult_nosel': ('Electrons multiplicity', 1, None, None),
    'h_METPt_nosel': ('MET p_{T} (GeV)', 1, None, None),
    'h_Ht_nosel': ('H_{T} (GeV)', 1, None, None),
    'h_AK8jetPt_nosel': ('Jets p_{T} (GeV)', 1, None, None),
    'h_AK8jetPhi_nosel' : ('Jets #Phi', 1, None, None),
    'h_AK8jetEta_nosel' : ('Jets #eta', 1, None, None),
    'h_AK8jetE_lead_nosel' : ('Leading jet E (GeV)', 1, None, None),
    'h_AK8jetPt_lead_nosel' : ('Leading jet p_{T} (GeV)', 1, None, None),
    'h_AK8jetPhi_lead_nosel' : ('Leading jet #Phi', 1, None, None),
    'h_AK8jetEta_lead_nosel' : ('Leading jet #eta', 1, None, None),
    'h_AK8jetE_sublead_nosel' : ('Subleading jet E (GeV)', 1, None, None),
    'h_AK8jetPt_sublead_nosel' : ('Subleading jet p_{T} (GeV)', 1, None, None),
    'h_AK8jetPhi_sublead_nosel' : ('Subleading jet #Phi', 1, None, None),
    'h_AK8jetEta_sublead_nosel' : ('Subleading jet #eta', 1, None, None),
    'h_AK8jetdP_nosel' : ('#Delta#Phi', 1, None, None),
    'h_AK8jetdR_nosel' : ('#Delta R', 1, None, None),
    'h_AK8jetdE_nosel' : ('#Delta#eta', 1, None, None),
    'h_Mmc_nosel' : ('m_{MC} (GeV)', 1, None, None),
    'h_Mjj_nosel' : ('m_{jj} (GeV)', 1, None, None),
    'h_Mt_nosel' : ('m_{T} (GeV)', 1, None, None),
    'h_dPhimin_nosel' : ('min #Delta#Phi', 1, None, None),
    'h_dPhi2_nosel' : ('#Delta#Phi(MET, subleading jet)', 1, None, None),
    'h_dPhi1_nosel' : ('#Delta#Phi(MET, leading jet)', 1, None, None),
    'h_transverseratio_nosel' : ('MET/m_{T}', 1,  None, None),
    
    'h_dEta' : ('#Delta#eta', 1, None, None),
    'h_dPhimin' : ('min #Delta#Phi', 1, None, None),
    'h_transverseratio' : ('MET/m_{T}', 1, None, None),
    'h_Mt':('m_{T} (GeV)', 1, 1, None) ,
    'h_Mmc' : ('m_{MC} (GeV)', 1, None, None),
    'h_Mjj' : ('m_{jj} (GeV)', 1, None, None),
    'h_METPt' : ('MET p_{T} (GeV)', 1, None, None),
    'h_dPhi' : ('#Delta#Phi', 1, None, None),
}

store = [
    'h_dEta',
    'h_dPhimin',
    'h_transverseratio',
    'h_Mt',
    'h_Mmc',
    'h_Mjj',
    'h_METPt',
    'h_dPhi',
    ]
