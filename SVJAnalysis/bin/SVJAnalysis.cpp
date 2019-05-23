#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h>

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
    double* metMET = core.AddVar("metMET", "MissingET.MET");
    double* metPhi = core.AddVar("metPhi", "MissingET.Phi");
    double* jsize = core.AddVar("Jet_size", "Jet_size");

    // disable debug
    // core.Debug(false);

    size_t nEntries = 100;
    for (size_t entry = 0; entry < nEntries; ++entry) {
        core.GetEntry(entry);
    }

    return 0;
}