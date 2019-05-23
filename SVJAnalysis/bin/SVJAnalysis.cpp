#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLorentzVector.h"
#include <vector>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <sstream>
#include <string>
#include "TFileCollection.h"
#include "THashList.h"
#include "TBenchmark.h"
#include <iostream>
#include <typeinfo>
#include <sstream> 
#include <utility>

// #include "TMVA/Tools.h"
// #include "TMVA/Reader.h"

// #include "DataFormats/Math/interface/deltaPhi.h"
// #include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "SVJ/SVJAnalysis/interface/Weights.h"
#include "SVJ/SVJAnalysis/interface/MT2Utility.h"
#include "SVJ/SVJAnalysis/interface/mt2w_bisect.h"
#include "SVJ/SVJAnalysis/interface/mt2bl_bisect.h"
#include "SVJ/SVJAnalysis/interface/Mt2Com_bisect.h"
// #include "DataFormats/Math/interface/deltaR.h"
#include "SVJ/SVJAnalysis/interface/kFactors.h"
#include "SVJ/SVJAnalysis/interface/TriggerFuncCorrector.h"
#include <map>
#include <cassert>

using std::string;
using std::endl;
using std::cout;
using std::vector;
using std::pair; 

class TLorentzMock{
    public:
        TLorentzMock() = delete; 

        TLorentzMock(Float_t Pt_, Float_t Eta_) {
            this->Eta_ = Eta_;
            this->Pt_ = Pt_; 
        }

        TLorentzMock(Float_t Pt_, Float_t Eta_, Float_t Isolation_) {
            this->Eta_ = Eta_;
            this->Pt_ = Pt_; 
            this->Isolation_ = Isolation_;
        }
        
        TLorentzMock(Float_t Pt_, Float_t Eta_, Float_t Isolation_, Float_t EhadOverEem_) {
            this->Eta_ = Eta_;
            this->Pt_ = Pt_; 
            this->Isolation_ = Isolation_;
            this->EhadOverEem_ = EhadOverEem_; 
        }

        Float_t Eta() {
            return this->Eta_; 
        }
        Float_t Pt() {
            return this->Pt_; 
        }
        Float_t Isolation() {
            return this->Isolation_;
        }
        Float_t EhadOverEem() {
            return this->EhadOverEem_; 
        }
        
    private:
        Float_t Eta_, Pt_, Isolation_, EhadOverEem_;
};

vector<TLorentzVector> getTLorentzVectorsPtEtaPhiM(vector<TLeaf*> &v) {
    int n = v[0]->GetLen(); 
    vector<TLorentzVector> ret;
    for (int i = 0; i < n; ++i) {
        ret.push_back(TLorentzVector());
        ret[i].SetPtEtaPhiM(
            v[0]->GetValue(i),
            v[1]->GetValue(i),
            v[2]->GetValue(i),
            v[3]->GetValue(i)
        );
    }
    return ret;
}

vector<TLorentzMock> getTLorentzMockVectors(vector<TLeaf*> &v) {
    int n = v[0]->GetLen();
    int size = v.size();
    vector<TLorentzMock> ret;
    for(int i = 0; i < n; ++i) {
        switch (size) {
            case 2: {
                ret.push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i)));
                break;
            }
            case 3: {
                ret.push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i), v[2]->GetValue(i)));
                break;
            }
            case 4: {
                ret.push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i), v[2]->GetValue(i), v[3]->GetValue(i)));
                break;
            }
        }
    }
    return ret; 
}

enum class vectorType {
    Lorentz,
    Mock,
    Map
};

class SVJFinder {
    public:
        // constructor, requires argv as input
        SVJFinder(char **argv, bool _debug=false) {
            cout << endl;
            log("-----------------------------------");
            log(":          SVJAnalysis            :");
            log("-----------------------------------");
            this->init_vars(argv);
            log("SVJ object created");
            log();
            debug = _debug; 
        }

        // generates tfile collection and returns a pointer to it
        TFileCollection *MakeFileCollection() {
            log("Loading File Collection from " + path);
            if (fc)
                delete fc;
            fc = new TFileCollection(sample.c_str(), sample.c_str(), path.c_str());
            log("Success: Loaded " + std::to_string(fc->GetNFiles())  + " file(s).");
            log();
            return fc;
        }

        // generates a chain an returns a pointer to it
        TChain *MakeChain() {
            log("Creating file chain with tree type '" + treename + "'...");
            if (chain)  
                delete chain;
            chain = new TChain(TString(treename));
            chain->AddFileInfoList(fc->GetList());
            nEvents = (Int_t)chain->GetEntries();
            log("Success");
            log();
            return chain;
        }

        // general function to add LEAF components as a part of a tlorentz vector
        vector<TLorentzVector>* AddLorentz(string vectorName, vector<string> components) {
            assert(components.size() == 4);
            AddCompsBase(vectorName, components);

            size_t i = LorentzVectors.size();
            subIndex.push_back(std::make_pair(i, vectorType::Lorentz));

            LorentzVectors.push_back(vector<TLorentzVector>());
            logr("Success");
            return &LorentzVectors[i];
        }

        // general function to add LEAF components as a part of a mock tlorentz vector
        vector<TLorentzMock>* AddLorentzMock(string vectorName, vector<string> components) {
            assert(components.size() > 1 && components.size() < 5);
            AddCompsBase(vectorName, components);
            
            size_t i = MockVectors.size();
            subIndex.push_back(std::make_pair(i, vectorType::Mock));

            MockVectors.push_back(vector<TLorentzMock>());
            logr("Success");
            return &MockVectors[i];
        }

        // general function to add LEAF components as a part of a basic vector of vectors
        vector<vector<double>>* AddComps(string vectorName, vector<string> components) {
            AddCompsBase(vectorName, components);
            
            size_t i = MapVectors.size();
            subIndex.push_back(std::make_pair(i, vectorType::Map));
            MapVectors.push_back(vector<vector<double>>());
            logr("Success");
            return &MapVectors[i];
        }

        // general function to add a singular leaf component
        double* AddVar(string varName, string component) {
            logp("Adding 1 component to var " + varName + "...  ");
            size_t i = varLeafs.size();
            varIndex[varName] = i;
            varLeafs.push_back(chain->FindLeaf(TString(component)));
            varValues.push_back(-1);
            logr("Success");
            return &varValues[i];
        }

        int GetEntry(int entry = 0) {
            assert(entry < chain->GetEntries());
            logp("Getting entry " + to_string(entry) + "...  ");
            chain->GetEntry(entry);
            for (size_t i = 0; i < subIndex.size(); ++i) {
                switch(subIndex[i].second) {
                    case vectorType::Lorentz: {
                        SetLorentz(i, subIndex[i].first);
                        break;
                    }
                    case vectorType::Mock: {
                        SetMock(i, subIndex[i].first);
                        break;
                    }
                    case vectorType::Map: {
                        SetMap(i, subIndex[i].first);
                        break;
                    }
                }
            }

            for (size_t i = 0; i < varLeafs.size(); ++i) {
                SetVar(i);
            }

            logr("Success");
            return 1; 
        }

        Int_t GetEntries() {
            return nEvents;  
        }

        void Debug(bool debugSwitch) {
            debug = debugSwitch;
        }


        // general init vars
        string sample, path, outdir, treename;
        Int_t nEvents;

        bool debug=true;
                       
    private:

        void SetLorentz(size_t leafIndex, size_t lvIndex) {
            vector<TLeaf*> & v = compVectors[leafIndex];
            vector<TLorentzVector> & ret = LorentzVectors[lvIndex];
            ret.clear();

            size_t n = v[0]->GetLen(); 
            for (size_t i = 0; i < n; ++i) {
                ret.push_back(TLorentzVector());
                ret[i].SetPtEtaPhiM(
                    v[0]->GetValue(i),
                    v[1]->GetValue(i),
                    v[2]->GetValue(i),
                    v[3]->GetValue(i)
                );
            }
        }

        void SetMock(size_t leafIndex, size_t mvIndex) {
            vector<TLeaf*> & v = compVectors[leafIndex];
            vector<TLorentzMock> & ret = MockVectors[mvIndex];            
            ret.clear();

            size_t n = v[0]->GetLen(), size = v.size();
            for(size_t i = 0; i < n; ++i) {
                switch(size) {
                    case 2: {
                        ret.push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i)));
                        break;
                    }
                    case 3: {
                        ret.push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i), v[2]->GetValue(i)));
                        break;
                    }
                    case 4: {
                        ret.push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i), v[2]->GetValue(i), v[3]->GetValue(i)));
                        break;
                    }
                    default: {
                        throw "Invalid number arguments for MockTLorentz vector (" + to_string(size) + ")";
                    }                   
                }
            }
        }

        void SetMap(size_t leafIndex, size_t mIndex) {
            vector<TLeaf*> & v = compVectors[leafIndex];

            // cout << mIndex << endl;
            // cout << leafIndex << endl;
            // cout << MapVectors.size() << endl; 
            vector<vector<double>> & ret = MapVectors[mIndex];
            size_t n = v[0]->GetLen();

            ret.clear(); 
            ret.resize(n);
            
            for (size_t i = 0; i < n; ++i) {
                ret[i].clear();
                ret[i].reserve(v.size());
                for (size_t j = 0; j < v.size(); ++j) {
                    ret[i].push_back(v[j]->GetValue(i));
                }
            }
        }

        void SetVar(size_t leafIndex) {
            varValues[leafIndex] = varLeafs[leafIndex]->GetValue(0);
        }

        void AddCompsBase(string& vectorName, vector<string>& components) {
            if(compIndex.find(vectorName) != compIndex.end())
                throw "Vector variable '" + vectorName + "' already exists!"; 
            size_t index = compIndex.size();
            logp("Adding " + to_string(components.size()) + " components to vector " + vectorName + "...  "); 
            compVectors.push_back(vector<TLeaf*>());
            compNames.push_back(vector<string>());

            for (size_t i = 0; i < components.size(); ++i) {
                compVectors[index].push_back(chain->FindLeaf(components[i].c_str()));
                compNames[index].push_back(lastWord(components[i]));
            }
            compIndex[vectorName] = index;
        }

        void log() {
            if (debug)
                cout << LOG_PREFIX << endl; 
        }
        
        template<typename t>
        void log(t s) {
            if (debug) {
                cout << LOG_PREFIX;
                lograw(s);
                cout << endl;
            }
        }

        template<typename t>
        void logp(t s) {
            if (debug) {
                cout << LOG_PREFIX;
                lograw(s);
            }
        }

        template<typename t>
        void logr(t s) {
            if (debug) {
                lograw(s);
                cout << endl; 
            }
        }

        template<typename t>
        void warning(t s) {
            debug = true;
            log("WARNING :: " + to_string(s));
            debug = false;
        }

        template<typename t>
        void lograw(t s) {
            cout << s; 
        }

        void init_vars(char **argv) {
            log("Starting");

            sample = argv[1];
            log(string("sample: " + sample)); 

            path = argv[2];
            log(string("File list to open: " + path));

            outdir = argv[6];
            log(string("Output directory: " + outdir)); 

            treename = argv[8];
            log(string("Tree name: " + treename));
        }

        vector<string> split(string s, char delimiter = '.') {
            std::replace(s.begin(), s.end(), delimiter, ' ');
            vector<string> ret;
            stringstream ss(s);
            string temp;
            while(ss >> temp)
                ret.push_back(temp);
            return ret;
        }

        string lastWord(string s, char delimiter = '.') {
            return split(s, delimiter).back(); 
        }

        TFileCollection *fc=nullptr;
        TChain *chain=nullptr;


        const string LOG_PREFIX = "SVJAnalysis :: ";
        std::map<vectorType, std::string> componentTypeStrings = {
            {vectorType::Lorentz, "TLorentzVector"},
            {vectorType::Mock, "MockTLorentzVector"},
            {vectorType::Map, "Map"}
        };

        // single variable data
        std::map<string, size_t> varIndex;
        vector<TLeaf *> varLeafs;
        vector<double> varValues;

        // vector component data
        std::map<string, size_t> compIndex;
        vector<pair<size_t, vectorType>> subIndex;

        vector<vector<TLeaf*>> compVectors;
        vector<vector<string>> compNames;

        vector< vector< TLorentzVector > > LorentzVectors;
        vector< vector< TLorentzMock > > MockVectors;
        vector<vector<vector<double>>> MapVectors;

};

template<typename t>
void pvec(vector<t> v){
    cout << "{ ";
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << "}" << endl;
}

int main(int argc, char **argv) {
    // declare object
    SVJFinder core(argv, true);
    
    // make file collection and chain
    core.MakeFileCollection();
    TChain * chain = core.MakeChain();

    // add componenets for jets, electrons, muons, and missing et
    // - declare
    vector<TLorentzVector> *Jets;
    vector<TLorentzMock> *Electrons, *Muons;
    vector<vector<double>> *ElectronParams, *MuonParams;
    double *metMET, *metPhi, *jsize; 
    // - then initalize
    Jets = core.AddLorentz("Jet", {"Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"});
    Electrons = core.AddLorentzMock("Electron", {"Electron.PT","Electron.Eta"});
    Muons = core.AddLorentzMock("Muon", {"MuonLoose.PT","MuonLoose.Eta"});
    ElectronParams = core.AddComps("ElectronParams", {"Electron.IsolationVarRhoCorr","Electron.EhadOverEem"}); 
    MuonParams = core.AddComps("MuonParams", {"MuonLoose.IsolationVarRhoCorr"});
    
    metMET = core.AddVar("metMET", "MissingET.MET");
    metPhi = core.AddVar("metPhi", "MissingET.Phi");
    jsize = core.AddVar("Jet_size", "Jet_size");

    // turn off debug outputss
    core.Debug(false);

    // analysis loop
    int skim = 10;
    for (int en = 0; en < skim; ++en) {
        core.GetEntry(en);
    }
    // cout << *jsize << endl;

    // for (int i = 0; i < 3; ++i) {
    // }

    return 0;
}