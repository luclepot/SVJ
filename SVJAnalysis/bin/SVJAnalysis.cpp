#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h>
#include "DataFormats/Math/interface/deltaPhi.h"

using std::sin;
using std::cos;
using std::sqrt; 
using std::abs; 

using namespace Cuts;

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

    bool JetPtVeto(TLorentzVector& jet) {
        return jet.Pt() > 200.;
    }
}

size_t leptonCount(vector<TLorentzMock>* leptons, vector<double>* isos) {
    size_t n = 0;
    for (size_t i = 0; i < leptons->size(); ++i)
        if (Vetos::LeptonVeto(leptons->at(i)) && Vetos::IsolationVeto(isos->at(i))) 
            n++;
    return n;
}


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

        // init
        core.InitCuts(); 
        core.GetEntry(entry);

        // add lepton cut, need to have zero
        core.Cut(
            (leptonCount(Muons, MuonIsolation) + leptonCount(Electrons, ElectronIsolation)) < 1,
            leptonCounts
            );

        // require more than 1 jet
        core.Cut(
            Jets->size() > 1,
            jetCounts
            ); 

        if (core.Cut(jetCounts)) {
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
            core.Cut(
                Vetos::JetEtaVeto(Jets->at(0)) && Vetos::JetEtaVeto(Jets->at(1)), 
                jetEtas
                );
            
            // leading jets meet delta eta veto
            core.Cut(
                Vetos::JetDeltaEtaVeto(Jets->at(0), Jets->at(1)),
                jetDeltaEtas
                );

            // ratio between calculated mt2 of dijet system and missing momentum is not negligible
            core.Cut(
                (*metFull_Pt) / MT2 > 0.15,
                metRatio
                );

            // require both leading jets to have transverse momentum greater than 200
            core.Cut(
                Vetos::JetPtVeto(Jets->at(0)) && Vetos::JetPtVeto(Jets->at(1)),
                jetPt
                );

            // conglomerate cut, whether jet is a dijet
            core.Cut(
                core.Cut(jetEtas) && core.Cut(jetPt),
                jetDiJet
                );

            // magnitude of MET squared > 1500 
            core.Cut(
                MT2 > 1500,
                metValue
                );

            core.Cut(
                core.CutsRange(0, int(preselection - 1)),
                preselection
                );            
        }
        core.PrintCuts(); 
    }

    core.Debug(true); 
    core.end();
    core.logt();

    return 0;
}
