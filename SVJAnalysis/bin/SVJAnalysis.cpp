#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h>
#include "DataFormats/Math/interface/deltaPhi.h"

using std::sin;
using std::cos;
using std::sqrt; 
using std::abs; 

// count leptons according to eta/pt restriction

namespace Vetos {
    bool LeptonVeto(TLorentzMock& lepton) {
        return fabs(lepton.Pt()) > 10 && fabs(lepton.Eta()) < 2.4;
    }
 
    bool IsolationVeto(double &iso) {
        return iso >= 0.4;
    }

    bool JetEtaVeto(TLorentzVector& jet) {
        return abs(jet.Eta()) < 2.5;
    }

    bool JetDeltaEtaVeto(TLorentzVector& jet1, TLorentzVector& jet2) {
        return abs(jet1.Eta() - jet2.Eta()) < 1.5;
    }
}

size_t leptonCount(vector<TLorentzMock>* leptons, vector<double>* isos) {
    size_t n = 0;
    for (size_t i = 0; i < leptons->size(); ++i)
        if (Vetos::LeptonVeto(leptons->at(i)) && Vetos::IsolationVeto(isos->at(i))) 
            n++;
    return n;
}

struct BooleanCuts {
    bool leptonsCount = false;
    bool jetsCount = false; 
    bool jetsEta = false;
    bool jetsDeltaEta = false;
    bool ptRatio = false;
    bool jetsPt = false; 
    bool jetsDijet = false; 
};

int main(int argc, char **argv) {
    // declare core object and enable debug
    SVJFinder core(argv);
    core.Debug(true);

    // make file collection and chain
    core.MakeFileCollection();
    core.MakeChain();

    // add componenets for jets (tlorentz)
    vector<TLorentzVector>* Jets = core.AddLorentz("Jet", {"Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"});
    
    // electrons/muons (mock tlorentz)
    vector<TLorentzMock>* Electrons = core.AddLorentzMock("Electron", {"Electron.PT","Electron.Eta"});
    vector<TLorentzMock>* Muons = core.AddLorentzMock("Muon", {"MuonLoose.PT","MuonLoose.Eta"});

    // extra vectorized parameters for electrons and muons    
    vector<double>* MuonIsolation = core.AddVectorVar("MuonIsolation", "MuonLoose.IsolationVarRhoCorr");
    vector<double>* ElectronEhadOverEem = core.AddVectorVar("ElectronEhadOverEem", "Electron.EhadOverEem");
    vector<double>* ElectronIsolation = core.AddVectorVar("ElectronIsolation", "Electron.IsolationVarRhoCorr"); 
    
    // single parameters
    double* metFull_Pt = core.AddVar("metMET", "MissingET.MET");
    double* metFull_Phi = core.AddVar("metPhi", "MissingET.Phi");
    double* jetSize = core.AddVar("Jet_size", "Jet_size");


    // needed variables for loop
    vector<bool> cutsRef; 
    TLorentzVector Vjj;
    int nMuons, nElectrons; 

    // disable debug
    core.Debug(false);

    // loop over the first nEntries (debug) 
    size_t nEntries = 100;
    
    // start loop timer
    core.start();


    for (size_t entry = 0; entry < nEntries; ++entry) {

        BooleanCuts cuts;
        
        core.GetEntry(entry);

        // lepton vetos (automatically called in leptonCount)
        nMuons = leptonCount(Muons, MuonIsolation);
        nElectrons = leptonCount(Electrons, ElectronIsolation);

        // add lepton cut, need to have zero
        cuts.leptonsCount = (nElectrons + nMuons) < 1; 

        // require more than 1 jet
        cuts.jetsCount = (Jets->size() > 1); 

        if (cuts.jetsCount) {
            double JetsDR = (Jets->at(0)).DeltaR(Jets->at(1));

            // leading jet eta difference 
            double dEta= fabs(Jets->at(0).Eta() - Jets->at(1).Eta());
            double dPhi= fabs(reco::deltaPhi(Jets->at(0).Phi(), Jets->at(1).Phi())); 
            
            // leading jet phi difference w/ missing et phi
            double dPhi_j0_met = fabs(reco::deltaPhi(Jets->at(0).Phi(), *metFull_Phi));
            double dPhi_j1_met = fabs(reco::deltaPhi(Jets->at(1).Phi(), *metFull_Phi));

            Vjj = Jets->at(0) + Jets->at(1);

            double metFull_Py = (*metFull_Pt)*sin(*metFull_Phi);
            double metFull_Px = (*metFull_Pt)*cos(*metFull_Phi);
            double Mjj = Vjj.M();
            double Mjj2 = Mjj*Mjj;
            double ptjj = Vjj.Pt();
            double ptjj2 = ptjj*ptjj;
            double ptMet = Vjj.Px()*(metFull_Px + Vjj.Py()*metFull_Py);

            double MT2 = sqrt(Mjj2 + 2*(sqrt(Mjj2 + ptjj2)*(*metFull_Pt) - ptMet));

            // leading jet etas both meet eta veto
            cuts.jetsEta = Vetos::JetEtaVeto(Jets->at(0)) && Vetos::JetEtaVeto(Jets->at(1));
            
            // leading jets meet delta eta veto
            cuts.jetsDeltaEta = Vetos::JetDeltaEtaVeto(Jets->at(0), Jets->at(1));

            // ratio between calculated mt2 of dijet system and missing momentum is not negligible
            cuts.ptRatio = (*metFull_Pt)/MT2 > 0.15;

            // 
            cuts.jetsPt = (Jets->at(0)).Pt() > 200 && (Jets->at(1)).Pt()>200;


            // preselection_muonveto = nMuons < 1 && preselection_muonLooseveto;
            // preselection_jetspt = 
            // preselection_dijet = preselection_jetseta && preselection_jetsID && preselection_jetspt;
            
        }
    }

    core.Debug(true); 
    core.end();
    core.logt();

    return 0;
}
