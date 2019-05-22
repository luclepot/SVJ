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
        SVJFinder(char **argv) {
            cout << endl;
            log("-----------------------------------");
            log(":          SVJAnalysis            :");
            log("-----------------------------------");
            this->init_vars(argv);
            log("SVJ object created");
            log();
        }

        TFileCollection *MakeFileCollection() {
            log("Loading File Collection from " + path);
            if (fc)
                delete fc;
            fc = new TFileCollection(sample.c_str(), sample.c_str(), path.c_str());
            log("Success: Loaded " + std::to_string(fc->GetNFiles())  + " file(s).");
            log();
            return fc;
        }

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

        // general function to add LEAF components as a part of a (recognized) structure
        void addComps(string vectorName, vector<string> components, vectorType type=vectorType::Map) {
            if (type == vectorType::Lorentz)
                assert(components.size() == 4);
            else if (type == vectorType::Mock)
                assert(components.size() > 1 && components.size() < 5);
            logp("Adding " + to_string(components.size()) + " components to vector " + vectorName + " with type " + componentTypeStrings[type] + "...  ");
            componentVectors[vectorName] = vector<TLeaf*>();
            componentTypes[vectorName] = type;
            componentNames[vectorName] = vector<string>();
            for (size_t i = 0; i < components.size(); ++i) {
                componentVectors[vectorName].push_back(chain->FindLeaf(components[i].c_str()));
                componentNames[vectorName].push_back(lastWord(components[i]));
            }
            logr("Success"); 
        }

        void addComp(string varName, string component) {
            logp("Adding 1 component to var " + varName + "...  ");
            componentVars[varName] = chain->FindLeaf(TString(component));
            logr("Success");
        }

        std::map<string, TLeaf *> componentVars;
        std::map<string, vector<TLeaf*>> componentVectors;
        std::map<string, vectorType> componentTypes;
        std::map<string, vector<string>> componentNames;

        TFileCollection *fc=nullptr;
        TChain *chain=nullptr;

    private:
        void log() {
            cout << LOG_PREFIX << endl; 
        }
        
        template<typename t>
        void log(t s) {
            cout << LOG_PREFIX;
            lograw(s);
            cout << endl;
        }

        template<typename t>
        void logp(t s) {
            cout << LOG_PREFIX;
            lograw(s);
        }

        template<typename t>
        void logr(t s) {
            lograw(s);
            cout << endl; 
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

        string sample, path, outdir, treename;
        Int_t nEvents;

        const string LOG_PREFIX = "SVJAnalysis :: ";
        std::map<vectorType, std::string> componentTypeStrings = {
            {vectorType::Lorentz, "TLorentzVector"},
            {vectorType::Mock, "MockTLorentzVector"},
            {vectorType::Map, "Map"}
        };

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
};

int main(int argc, char **argv) {
    SVJFinder core(argv);
    TFileCollection* fc = core.MakeFileCollection();
    TChain* chain = core.MakeChain();
    core.addComps("Jet", {"Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"}, vectorType::Lorentz);
    core.addComps("Electron", {"Electron.PT","Electron.Eta"}, vectorType::Mock);
    core.addComps("ElectronParams", {"Electron.IsolationVarRhoCorr","Electron.EhadOverEem"}, vectorType::Map); 
    core.addComps("Muon", {"MuonLoose.PT","MuonLoose.Eta"}, vectorType::Mock);
    core.addComps("MuonParams", {"MuonLoose.IsolationVarRhoCorr"}, vectorType::Map);
    core.addComps("MissingET", {"MissingET.MET", "MissingET.Phi"}, vectorType::Map);
    return 0;
}