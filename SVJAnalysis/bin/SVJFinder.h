#include "TChain.h"
#include "TLeaf.h"
#include "TLorentzVector.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <string>
#include "TFileCollection.h"
#include "THashList.h"
#include "TBenchmark.h"
#include <iostream>
#include <sstream> 
#include <utility>
#include <map>
#include <cassert>

using std::string;
using std::endl;
using std::cout;
using std::vector;
using std::pair; 
using std::to_string;
using std::stringstream; 

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

        // destructor for dynamically allocated data
        ~SVJFinder() {
            DelVector(varValues);
            DelVector(vectorVarValues);
            DelVector(LorentzVectors);
            DelVector(MockVectors);
            DelVector(MapVectors);
            // DelVector(varLeaves);
            // DelVector(vectorVarLeaves);
            // for (vector<TLeaf*> vec : compVectors) 
            //     DelVector(vec);
        }

        // sets up tfile collection and returns a pointer to it
        TFileCollection *MakeFileCollection() {
            log("Loading File Collection from " + path);
            if (fc)
                delete fc;
            fc = new TFileCollection(sample.c_str(), sample.c_str(), path.c_str());
            log("Success: Loaded " + std::to_string(fc->GetNFiles())  + " file(s).");
            log();
            return fc;
        }

        // sets up tchain and returns a pointer to it
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

        // creates, assigns, and returns tlorentz vector pointer to be updated on GetEntry
        vector<TLorentzVector>* AddLorentz(string vectorName, vector<string> components) {
            assert(components.size() == 4);
            AddCompsBase(vectorName, components);
            size_t i = LorentzVectors.size();
            subIndex.push_back(std::make_pair(i, vectorType::Lorentz));
            vector<TLorentzVector>* ret = new vector<TLorentzVector>; 
            LorentzVectors.push_back(ret);
            logr("Success");
            return ret;
        }

        // creates, assigns, and returns mock tlorentz vector pointer to be updated on GetEntry
        vector<TLorentzMock>* AddLorentzMock(string vectorName, vector<string> components) {
            assert(components.size() > 1 && components.size() < 5);
            AddCompsBase(vectorName, components);
            size_t i = MockVectors.size();
            subIndex.push_back(std::make_pair(i, vectorType::Mock));
            vector<TLorentzMock>* ret = new vector<TLorentzMock>; 
            MockVectors.push_back(ret);
            logr("Success");
            return ret;
        }

        // creates, assigns, and returns general double vector pointer to be updated on GetEntry
        vector<vector<double>>* AddComps(string vectorName, vector<string> components) {
            AddCompsBase(vectorName, components);
            size_t i = MapVectors.size();
            subIndex.push_back(std::make_pair(i, vectorType::Map));
            vector<vector<double>>* ret = new vector<vector<double>>;
            MapVectors.push_back(ret);
            logr("Success");
            return ret;
        }

        // creates, assigns, and returns a vectorized single variable pointer to be updates on GetEntry
        vector<double>* AddVectorVar(string vectorVarName, string component) {
            logp("Adding 1 component to vector var " + vectorVarName + "...  ");
            int i = int(vectorVarValues.size());
            vectorVarIndex[vectorVarName] = i;
            vectorVarLeaves.push_back(chain->FindLeaf(TString(component)));
            vector<double>* ret = new vector<double>;
            vectorVarValues.push_back(ret);
            logr("Success");
            // log(vectorVarIndex.size());
            // log(vectorVarValues.back().size());
            // log(i);
            return ret;
        }

        // creates, assigns, and returns a singular double variable pointer to update on GetEntry 
        double* AddVar(string varName, string component) {
            logp("Adding 1 component to var " + varName + "...  ");
            size_t i = varLeaves.size();
            varIndex[varName] = i;
            double* ret = new double;
            varLeaves.push_back(chain->FindLeaf(TString(component)));
            varValues.push_back(ret);
            logr("Success");
            return ret;
        }

        // get the ith entry of the TChain
        void GetEntry(int entry = 0) {
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

            for (size_t i = 0; i < varValues.size(); ++i) {
                SetVar(i);
            }

            for (size_t i = 0; i < vectorVarValues.size(); ++i) {
                SetVectorVar(i);
            }
            // cout << vectorVarValues.size() << endl;
            // for (size_t i = 0; i < vectorVarValues.size(); ++i) {
            //     cout << i << " | "; 
            //     for (size_t j = 0; j < vectorVarValues[i].size(); ++j) {
            //         cout << j << ": " << vectorVarValues[i][j] << ", ";
            //     }
            //     cout << endl; 
            // }
            logr("Success");
        }

        // get the number of entries in the TChain
        Int_t GetEntries() {
            return nEvents;  
        }

        // Turn on or off debug logging with this switch
        void Debug(bool debugSwitch) {
            debug = debugSwitch;
        }

        // prints a summary of the current entry
        void Current() {
            log();
            if (varIndex.size() > 0) {
                log();
                print("SINGLE VARIABLES:");
            }
            for (auto it = varIndex.begin(); it != varIndex.end(); it++) {
                print(it->first, 1);
                print(varValues[it->second], 2);
            }
            if (vectorVarIndex.size() > 0) {
                log();
                print("VECTOR VARIABLES:");
            }
            for (auto it = vectorVarIndex.begin(); it != vectorVarIndex.end(); it++) {
                print(it->first, 1);
                print(varValues[it->second], 2);
            }
            if (MapVectors.size() > 0) {
                log();
                print("MAP VECTORS:");
            }
            for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
                if (subIndex[it->second].second == vectorType::Map) {
                    print(it->first, 1);
                    print(MapVectors[subIndex[it->second].first], 2);
                }
            }
            if (MockVectors.size() > 0) {
                log();
                print("MOCK VECTORS:");
            }
            for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
                if (subIndex[it->second].second == vectorType::Mock) {
                    print(it->first, 1);
                    print(MockVectors[subIndex[it->second].first], 2);
                }
            }
            if (LorentzVectors.size() > 0) {
                log();
                print("TLORENTZ VECTORS:");
            }
            for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
                if (subIndex[it->second].second == vectorType::Lorentz) {
                    print(it->first, 1);
                    print(LorentzVectors[subIndex[it->second].first], 2);
                }
            }
            log(); 
            log();
        }

        // general init vars, parsed from argv
        string sample, path, outdir, treename;
        // number of events
        Int_t nEvents;
        // internal debug switch
        bool debug=true;
                       
    private:

        template<typename t>
        void DelVector(vector<t*> &v) {
            for (size_t i = 0; i < v.size(); ++i) {
                delete v[i];
                v[i] = nullptr;
            }
        }

        void SetLorentz(size_t leafIndex, size_t lvIndex) {
            vector<TLeaf*> & v = compVectors[leafIndex];
            vector<TLorentzVector> * ret = LorentzVectors[lvIndex];
            ret->clear();

            size_t n = v[0]->GetLen(); 
            for (size_t i = 0; i < n; ++i) {
                ret->push_back(TLorentzVector());
                ret->at(i).SetPtEtaPhiM(
                    v[0]->GetValue(i),
                    v[1]->GetValue(i),
                    v[2]->GetValue(i),
                    v[3]->GetValue(i)
                );
            }
        }

        void SetMock(size_t leafIndex, size_t mvIndex) {
            vector<TLeaf*> & v = compVectors[leafIndex];
            vector<TLorentzMock>* ret = MockVectors[mvIndex];            
            ret->clear();

            size_t n = v[0]->GetLen(), size = v.size();
            for(size_t i = 0; i < n; ++i) {
                switch(size) {
                    case 2: {
                        ret->push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i)));
                        break;
                    }
                    case 3: {
                        ret->push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i), v[2]->GetValue(i)));
                        break;
                    }
                    case 4: {
                        ret->push_back(TLorentzMock(v[0]->GetValue(i), v[1]->GetValue(i), v[2]->GetValue(i), v[3]->GetValue(i)));
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
            vector<vector<double>>* ret = MapVectors[mIndex];
            size_t n = v[0]->GetLen();

            ret->clear(); 
            ret->resize(n);
            
            for (size_t i = 0; i < n; ++i) {
                ret->at(i).clear();
                ret->at(i).reserve(v.size());
                for (size_t j = 0; j < v.size(); ++j) {
                    ret->at(i).push_back(v[j]->GetValue(i));
                }
            }
        }

        void SetVar(size_t leafIndex) {
            *varValues[leafIndex] = varLeaves[leafIndex]->GetValue(0);
        }

        void SetVectorVar(size_t leafIndex) {
            vectorVarValues[leafIndex]->clear();
            for (int i = 0; i < vectorVarLeaves[leafIndex]->GetLen(); ++i) {
                vectorVarValues[leafIndex]->push_back(vectorVarLeaves[leafIndex]->GetValue(i));
            }
            // log(leafIndex);
            // log(vectorVarLeaves[leafIndex]->GetLen());
            // log(vectorVarValues[leafIndex].size());
            // log();
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

        void indent(int level){
            cout << LOG_PREFIX << string(level*3, ' ');
        }

        void print(string s, int level=0) {
            indent(level);
            cout << s << endl;
        }

        void print(double* var, int level=0) {
            indent(level); cout << *var << endl;
        }

        void print(vector<double>* var, int level=0) {
            indent(level);
            cout << "{ ";
            for (size_t i = 0; i < var->size() - 1; ++i) {
                cout << var->at(i) << ", ";
            }
            cout << var->back() << " }";
            cout << endl;
        }

        void print(vector<vector<double>>* var, int level=0) {
            for (size_t i = 0; i < var->size(); ++i) {
                print(&var[i], level);
            }
        }

        void print(vector<TLorentzMock>* var, int level=0) {
            for (size_t i = 0; i < var->size(); ++i) {
                auto elt = var->at(i);
                indent(level); cout << "(Pt,Eta)=(" << elt.Pt() << "," << elt.Eta() << "}" << endl;
            }
        }

        void print(vector<TLorentzVector>* var, int level=0) {
            for (size_t i = 0; i < var->size(); ++i) {
                auto elt = var->at(i);
                indent(level);
                elt.Print();
            }
        }

        void print() {
            indent(0);
            cout << endl; 
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
        vector<TLeaf *> varLeaves;
        vector<double*> varValues;

        // vector variable data
        std::map<string, size_t> vectorVarIndex;
        vector<TLeaf *> vectorVarLeaves;
        vector<vector<double>*> vectorVarValues;

        // vector component data
        std::map<string, size_t> compIndex;
        vector<pair<size_t, vectorType>> subIndex;

        vector<vector<TLeaf*>> compVectors;
        vector<vector<string>> compNames;

        vector< vector< TLorentzVector >*> LorentzVectors;
        vector< vector< TLorentzMock >*> MockVectors;
        vector<vector<vector<double>>*> MapVectors;

};
