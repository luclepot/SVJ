#include "TTree.h"
#include "TLeaf.h"
#include "TFile.h"
#include <string>
#include <iostream>
#include <fstream> 
#include <vector>

using std::string;
using std::vector;


class ParallelTreeChain{
    public:
        ParallelTreeChain() {

        }

        ~ParallelTreeChain() {
            for (size_t i = 0; i < files.size(); ++i) {
                files[i]->Close();
                files[i] = nullptr; 
            }

            for (size_t i = 0; i < trees.size(); ++i) {
                trees[i] = nullptr; 
            }
        }

        vector<TLeaf*> FindLeaf(string & spec) {
            vector<TLeaf*> v;
            for (size_t i = 0; i < ntrees; ++i) 
                v.push_back(trees[i]->FindLeaf(spec.c_str()));
            return v;
        }

        vector<TLeaf*> FindLeaf(const char* spec) {
            vector<TLeaf*> v;
            for (size_t i = 0; i < ntrees; ++i) 
                v.push_back(trees[i]->FindLeaf(spec));
            return v;
        }

        int GetEntry(int entry) {
            if (entry > entries)
                return -1;
            getN(entry);
            trees[currentTree]->GetEntry(currentEntry);
            return currentTree;
        }

        void getN(int entry){
            int tn = 0;
            while (entry >= 0) 
                entry -= sizes[tn++];
            currentTree = tn - 1;
            currentEntry = entry + sizes[tn - 1]; 
        }

        size_t size() {
            return ntrees;
        }

        Int_t GetEntries() { 
            return entries; 
        }

        void GetTrees(string filename) {
            GetTreeNames(filename);
            entries = 0; 
            for (size_t i = 0; i < treenames.size(); ++i) {
                files.push_back(new TFile(treenames[i].c_str()));
                trees.push_back((TTree*)files[i]->Get("Delphes"));
                sizes.push_back(trees[i]->GetEntries());
                trees[i]->GetEntry(0);
                entries += sizes[i];
            }   
            ntrees = trees.size(); 
        }

        void GetTreeNames(string filename) {
            std::ifstream file(filename.c_str());
            string s;
            while (getline(file, s))
                if (s.size() > 0) 
                    treenames.push_back(s);
        }

        int currentEntry, currentTree; 
        size_t ntrees;
        Int_t entries; 
        vector<string> treenames;
        vector<TTree*> trees;
        vector<TFile*> files;
        vector<size_t> sizes; 
}; 
