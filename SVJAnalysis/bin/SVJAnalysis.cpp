#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h> 


int main(int argc, char **argv) {
    // declare object
    SVJFinder core(argv, true);
    
    // make file collection and chain
    core.MakeFileCollection();
    TChain * chain = core.MakeChain();

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

    core.GetEntry(17);

    core.Current(); 
    // for (int i = 0; i < 100; ++i)
    // print(Jets);
    // print(); 
    // print(Electrons);
    // print(Muons);
    // print();
    // print(MuonIsolation);
    // print(ElectronIsolation);
    // print(ElectronEhadOverEem);
    // print();
    // print(metMET);
    // print(metPhi);
    // print();
    // print(metMET);
    // print(metPhi);
    // print(jsize);

    // for (int en = 0; en < size; ++en) {

    // }

    // pvec(*ElectronIsolation); 
    // pvec(*ElectronEhadOverEem);
    // pvec(*MuonIsolation);
    // cout << "first" << endl;
    // cout << "second" << endl;

    // core.Debug(true);

    // // analysis loop
    // int size = 4;
    
    return 0;
}