#include <string>
#include <iostream>
#include <vector>
#include <algorithm> 

using namespace std;

const string newname = "/eos/project/d/dshep/TOPCLASS/DijetAnomaly/ZprimeDark_2000GeV_13TeV_PU40/ZprimeDark_2000GeV_13TeV_PU40_1_ap.root";
const string oldname = "/eos/project/d/dshep/TOPCLASS/DijetAnomaly/ZprimeDark_2000GeV_13TeV_PU40/ZprimeDark_2000GeV_13TeV_PU40_0.root";

TCanvas *PlotTwo(TTree* tree1, TTree* tree2, string name, int bins, string f1label, string f2label) {

    float min1 = tree1->GetMinimum(name.c_str());
    float min2 = tree2->GetMinimum(name.c_str()); 

    float max1 = tree1->GetMaximum(name.c_str());
    float max2 = tree1->GetMaximum(name.c_str());

    float min = std::min(min1, min2);
    float max = std::max(max1, max2); 

    TH1F * h1 = new TH1F("h1", "h1", bins, min, max);
    TH1F * h2 = new TH1F("h2", "h2", bins, min, max);

    tree1->Draw((name + ">>h1").c_str(), "Jet.PT>0", "goff");
    tree2->Draw((name + ">>h2").c_str(), "Jet.PT>0", "goff");
    
    h1->Scale(1./h1->Integral());
    h2->Scale(1./h2->Integral());

    // h1->SetMarkerSize(0)/

    TCanvas *cst = new TCanvas("cst");
    auto legend = new TLegend(0.6, 0.7, .95, .92);

    THStack *hs = new THStack("hs", name.c_str()); 

    // h1->SetMarkerStyle(21);
    h1->SetLineColor(kBlue);
    h1->SetLineStyle(1); 

    // h2->SetMarkerStyle(21);
    h2->SetLineColor(kRed);
    h2->SetLineStyle(1); 


    hs->Add(h1, "HIST");
    hs->Add(h2, "HIST");

    cst->cd();
    hs->Draw("nostack"); 

    // legend->SetHeader("","C"); 
    legend->AddEntry(h1, f1label.c_str(), "f"); 
    legend->AddEntry(h2, f2label.c_str(), "f"); 
    legend->Draw(); 

    return cst; 
} 

void compare(string f1name, string f2name, vector<string> names, string f1label="old signal", string f2label="new signal", string treename="Delphes", int bins=100) {

    TFile * f1 = new TFile(f1name.c_str());
    TFile * f2 = new TFile(f2name.c_str());

    TTree * tree1 = (TTree*)f1->Get(treename.c_str());
    TTree * tree2 = (TTree*)f2->Get(treename.c_str());

    for (size_t i = 0; i < names.size(); ++i) {
        string name = names[i]; 
        TImage *img = TImage::Create();
        TCanvas* canvas = PlotTwo(tree1, tree2, name, bins, f1label, f2label); 
        img->FromPad(canvas);
        string outpath = name + ".png"; 
        cout << "saving image to file '" << outpath << "'" << endl;
        img->WriteImage(outpath.c_str()); 
        delete img;
    }
}