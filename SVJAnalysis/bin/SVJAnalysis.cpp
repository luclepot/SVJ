#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h>

// remove leptons according to eta/pt restriction
size_t leptonVeto(vector<TLorentzMock>* lepton) {
    size_t n = 0;
    for (size_t i = 0; i < lepton->size(); ++i)
        if (std::fabs(lepton->at(i).Pt()) > 10.0 && std::fabs(lepton->at(i).Eta()) < 2.4)
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

    // disable debug
    core.Debug(false);

    // loop over the first nEntries (debug) 
    size_t nEntries = 10000;
    
    // start loop timer
    core.start();
    
    for (size_t entry = 0; entry < nEntries; ++entry) {
        core.GetEntry(entry);
        int nMuons=0, nElectrons=0;

        nMuons = leptonVeto(Muons);
        nElectrons = leptonVeto(Electrons);

        if (nMuons || nElectrons || Muons->size() || Electrons->size())
            cout << entry << ": " << nElectrons << "/" << Electrons->size() << ", " << nMuons << "/" << Muons->size() << endl;
    }

    core.Debug(true); 
    core.end();
    core.logt();

    return 0;
}
