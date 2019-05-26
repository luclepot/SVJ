#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h>
#include "DataFormats/Math/interface/deltaPhi.h"

using std::sin;
using std::cos;
using std::sqrt; 
using std::abs; 

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

double calculateMT2(TLorentzVector& Vjj, double metFull_Pt, double metFull_Phi) {
    double Mjj2 = Vjj.M()*Vjj.M(), ptjj2 = Vjj.Pt()*Vjj.Pt();
    return sqrt(Mjj2 + 2*(sqrt(Mjj2 + ptjj2)*(metFull_Pt) - Vjj.Px()*((metFull_Pt)*cos(metFull_Phi) + Vjj.Py()*(metFull_Pt)*sin(metFull_Phi))));
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
    // vector<double>* ElectronEhadOverEem = core.AddVectorVar("ElectronEhadOverEem", "Electron.EhadOverEem");
    vector<double>* ElectronIsolation = core.AddVectorVar("ElectronIsolation", "Electron.IsolationVarRhoCorr"); 
    
    // single parameters
    double* metFull_Pt = core.AddVar("metMET", "MissingET.MET");
    double* metFull_Phi = core.AddVar("metPhi", "MissingET.Phi");
    // double* jetSize = core.AddVar("Jet_size", "Jet_size");


    // needed variables for loop

    // disable debug
    core.Debug(false);

    // loop over the first nEntries (debug) 
    size_t nEntries = 100;
    TLorentzVector Vjj;
    double MT2;
    
    // add histogram tracking
    core.AddHist(Hists::dEta, "h_dEta", "#Delta#eta(j0,j1)", 100, 0, 10);
    core.AddHist(Hists::dPhi, "h_dPhi", "#Delta#Phi(j0,j1)", 100, 0, 5);
    core.AddHist(Hists::tRatio,  "h_transverseratio", "MET/M_{T}", 100, 0, 1);
    core.AddHist(Hists::met2, "h_Mt", "m_{T}", 750, 0, 7500);
    core.AddHist(Hists::mjj, "h_Mjj", "m_{JJ}", 750, 0, 7500);
    core.AddHist(Hists::metPt, "h_METPt", "MET_{p_{T}}", 100, 0, 2000);
    
    
    // start loop timer
    core.start();
    for (size_t entry = 0; entry < nEntries; ++entry) {

        // init
        core.InitCuts(); 
        core.GetEntry(entry);

        // require zero leptons which pass cuts
        core.Cut(
            (leptonCount(Muons, MuonIsolation) + leptonCount(Electrons, ElectronIsolation)) < 1,
            Cuts::leptonCounts
            );

        // require more than 1 jet
        core.Cut(
            Jets->size() > 1,
            Cuts::jetCounts
            );


        // rest of cuts, dependent on jetcount
        if (core.Cut(Cuts::jetCounts)) {

            // double JetsDR = (Jets->at(0)).DeltaR(Jets->at(1));
            
            // leading jet phi difference w/ missing et phi
            // double dPhi_j0_met = fabs(reco::deltaPhi(Jets->at(0).Phi(), *metFull_Phi));
            // double dPhi_j1_met = fabs(reco::deltaPhi(Jets->at(1).Phi(), *metFull_Phi));

            Vjj = Jets->at(0) + Jets->at(1);
            MT2 = calculateMT2(Vjj, *metFull_Pt, *metFull_Phi);

            // leading jet etas both meet eta veto
            core.Cut(
                Vetos::JetEtaVeto(Jets->at(0)) && Vetos::JetEtaVeto(Jets->at(1)), 
                Cuts::jetEtas
                );
            
            // leading jets meet delta eta veto
            core.Cut(
                Vetos::JetDeltaEtaVeto(Jets->at(0), Jets->at(1)),
                Cuts::jetDeltaEtas
                );

            // ratio between calculated mt2 of dijet system and missing momentum is not negligible
            core.Cut(
                (*metFull_Pt) / MT2 > 0.15,
                Cuts::metRatio
                );

            // require both leading jets to have transverse momentum greater than 200
            core.Cut(
                Vetos::JetPtVeto(Jets->at(0)) && Vetos::JetPtVeto(Jets->at(1)),
                Cuts::jetPt
                );

            // conglomerate cut, whether jet is a dijet
            core.Cut(
                core.Cut(Cuts::jetEtas) && core.Cut(Cuts::jetPt),
                Cuts::jetDiJet
                );

            // magnitude of MET squared > 1500 
            core.Cut(
                MT2 > 1500,
                Cuts::metValue
                );

            // preselection cut
            core.Cut(
                core.CutsRange(0, int(Cuts::preselection)),
                Cuts::preselection
                );
            
            // tighter MET ratio
            core.Cut(
                (*metFull_Pt) / MT2 > 0.25,
                Cuts::metRatioTight
                );
                 
            // final selection cut
            core.Cut(
                core.Cut(Cuts::preselection) && core.Cut(Cuts::metRatioTight),
                Cuts::selection
            ); 

            // save histograms, if passing
            if (core.Cut(Cuts::selection)) {
                core.Fill(Hists::dEta, fabs(Jets->at(0).Eta() - Jets->at(1).Eta())); 
                core.Fill(Hists::dPhi, fabs(reco::deltaPhi(Jets->at(0).Phi(), Jets->at(1).Phi())));
                core.Fill(Hists::tRatio, (*metFull_Pt) / MT2);
                core.Fill(Hists::mjj, Vjj.M());
                core.Fill(Hists::met2, MT2);
                core.Fill(Hists::metPt, *metFull_Pt);

                // fill histogram with
                // dEta
                // dPhi
                // metFull_Pt/MT2
                // MT2
                // Mjj
                // metFull_Pt
                core.PrintCuts();
            }
        }

    }
    core.Debug(true);
    core.end();
    core.logt();
    core.WriteHists(); 

    return 0;
}
