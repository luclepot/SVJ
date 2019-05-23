#include "TLorentzMock.h"
#include "SVJFinder.h"
#include <math.h> 

// #include <typeinfo>
// #include "DataFormats/Math/interface/LorentzVector.h"
// #include "SVJ/SVJAnalysis/interface/Weights.h"
// #include "SVJ/SVJAnalysis/interface/MT2Utility.h"
// #include "SVJ/SVJAnalysis/interface/mt2w_bisect.h"
// #include "SVJ/SVJAnalysis/interface/mt2bl_bisect.h"
// #include "SVJ/SVJAnalysis/interface/Mt2Com_bisect.h"
// #include "SVJ/SVJAnalysis/interface/kFactors.h"
// #include "SVJ/SVJAnalysis/interface/TriggerFuncCorrector.h"
// #include "DataFormats/Math/interface/deltaR.h"
// #include <assert.h>
// #include "TMVA/Tools.h"
// #include "TMVA/Reader.h"
// #include "DataFormats/Math/interface/deltaPhi.h"
// #include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
// #include "TH1.h"
// #include "TH1F.h"
// #include "TH2F.h"

// using SVJFinder;
// using SVJFinder::vectorTypes;
// using TLorentzMock::TLorentzMock; 

template<typename t>
void pvec(vector<t> v, bool end=true){
    cout << "printin"; 
    cout << "{ ";
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << "},   "; 
    if (end)
        cout << endl;
}


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