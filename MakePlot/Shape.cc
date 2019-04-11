#include "TH1.h"
#include "TMath.h"
#include "TF1.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TFile.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TString.h"
#include "TLatex.h"
#include "TVirtualFitter.h"
#include "TRandom.h"
#include "TGraphErrors.h"

#define PATH "output/histos2017/"

std::vector<double> excludedbins(TH1F *histo) {
  std::vector<double> excluded;
  for(int i=1; i<=histo->GetNbinsX(); i++){
    if(histo->GetBinError(i)>0.5){
      excluded.push_back(histo->GetBinLowEdge(i));
    }
  }
  return excluded;
}

Double_t binwidth(TH1F *histo) {
  return histo->GetBinWidth(1);
}

Bool_t reject;
std::vector<double> excl;
Double_t width; 

Double_t fline(Double_t *x, Double_t *par)
{

  Bool_t reject_bin = kFALSE;
  //cout << "size in function: " << excl.size() << endl;
  for(int i=0; i<excl.size(); i++){
    
    //cout << excl.at(i) << " " << x[0] << endl;
    //cout << "maggiore di " << (int)(x[0]>excl.at(i)) << " minore di " << (int)(x[0]<(excl.at(i)+width)) << endl;
    //cout << excl.at(i)<< " " << x[0] << " " << ((excl.at(i)+width)) << endl;
    //cout << "width: " << width << endl;
    if(x[0]>excl.at(i) && x[0]<(excl.at(i)+width)){
      reject_bin = kTRUE;
      //cout << "it passes the if" << endl;
    } 
    //cout << excl.at(i) << " " << reject_bin << endl;
  }
  
  if(reject && reject_bin){
  //if(reject && x[0] > 3000){
    TF1::RejectPoint();
    return 0;
  }
  //return par[0] + par[1]*x[0] + par[2]*pow(x[0],2);
  //return par[0] + par[1]*x[0];
  return exp(par[0]+par[1]*x[0]) + (par[2] + par[3] * x[0]);
  //return exp(par[0]+par[1]*x[0]);
}

void Shape(const char* histo_num, const char* histo_den){
  TH1::SetDefaultSumw2();
  gROOT->ForceStyle();

  TFile *qcd = new TFile((PATH+string("QCD.root")).c_str());
  TFile *tt = new TFile((PATH+string("TT.root")).c_str());
  TFile *wjets = new TFile((PATH+string("WJets.root")).c_str());
  TFile *zjets = new TFile((PATH+string("ZJets.root")).c_str());
  //TFile *data = new TFile((PATH+string("Data.root")).c_str());

  TH1F *h_num_qcd = (TH1F*)qcd->Get(histo_num);
  TH1F *h_num_tt = (TH1F*)tt->Get(histo_num);
  TH1F *h_num_wjets = (TH1F*)wjets->Get(histo_num);
  TH1F *h_num_zjets = (TH1F*)zjets->Get(histo_num);

  TH1F *h_den_qcd = (TH1F*)qcd->Get(histo_den);
  TH1F *h_den_tt = (TH1F*)tt->Get(histo_den);
  TH1F *h_den_wjets = (TH1F*)wjets->Get(histo_den);
  TH1F *h_den_zjets = (TH1F*)zjets->Get(histo_den);

  //TH1F *h_den_data = (TH1F*)data->Get(histo_den);

  THStack *hs_num = new THStack("hs_num","Stacked 1D histograms");
  hs_num->Add(h_num_tt);
  hs_num->Add(h_num_qcd);
  hs_num->Add(h_num_zjets);
  hs_num->Add(h_num_wjets);

  THStack *hs_den = new THStack("hs_den","Stacked 1D histograms");
  //hs_den->Add(h_den_data);
  hs_den->Add(h_den_tt);
  hs_den->Add(h_den_qcd);
  hs_den->Add(h_den_zjets);
  hs_den->Add(h_den_wjets);

  TH1F *h_numerator = (TH1F*) hs_num->GetStack()->Last();
  TH1F *h_denumerator = (TH1F*) hs_den->GetStack()->Last();

  TCanvas *c1 = new TCanvas("c1","c1",0,0,600,600);
  TPad *pad1 = new TPad("pad1", "pad1", 0, 0.35, 1, 1.0);
  pad1->SetBottomMargin(0); 
  pad1->SetGridx();         
  pad1->SetGridy();         
  pad1->Draw();             
  pad1->cd();         

  gStyle->SetOptStat(0);
  gPad->SetLogy();
  
  h_denumerator->SetLineColor(kBlue);
  h_denumerator->SetLineWidth(2);
  h_denumerator->SetMarkerColor(kBlue);
  h_denumerator->SetMarkerStyle(21);
  h_denumerator->SetTitle("");
  h_denumerator->GetYaxis()->SetTitle("Normalized number of events");
  h_denumerator->GetXaxis()->SetTitle("m_{T} (GeV)");
  h_denumerator->GetXaxis()->SetRangeUser(1500,3900);
  h_denumerator->DrawNormalized("ep");

  h_numerator->SetLineColor(kGreen+2);
  h_numerator->SetLineWidth(2);
  h_numerator->SetMarkerStyle(21);
  h_numerator->SetMarkerColor(kGreen+2);
  h_numerator->GetXaxis()->SetRangeUser(1500,3900);
  h_numerator->DrawNormalized("ep same");

  //h_denumerator->SetMaximum(1.5*TMath::Max(h_numerator->GetMaximum(), h_denumerator->GetMaximum()));
  //h_denumerator->SetMinimum(TMath::Min(h_numerator->GetMinimum(), h_denumerator->GetMinimum()));

  leg = new TLegend(0.65,0.65,0.85,0.85);
  leg->SetFillColor(0);
  leg->SetBorderSize(0);
  leg->AddEntry(h_numerator,histo_num,"l");
  leg->AddEntry(h_denumerator,histo_den,"l");
  leg->Draw();

  c1->Update();
  c1->cd();          

  TPad *pad2 = new TPad("pad2", "pad2", 0, 0.05, 1, 0.3);
  pad2->SetTopMargin(0);
  pad2->SetBottomMargin(0.2);
  pad2->SetGridx(); 
  pad2->SetGridy(); 
  pad2->Draw();
  pad2->cd();       
  gStyle->SetOptFit(1);

  h_numerator->Divide(h_denumerator);
  TH1F *h_ratio = (TH1F*)h_numerator->Clone("h_ratio");
  h_ratio->GetXaxis()->SetRangeUser(1500,3900);

  h_ratio->SetLineColor(kBlack);
  h_ratio->SetStats(0);     
  h_ratio->SetMarkerStyle(20);
  h_ratio->SetMarkerColor(kBlack);
  //h_ratio->SetMinimum(std::max(h_ratio->GetMinimum(), -100.)); 
  //h_ratio->SetMaximum(std::min(h_ratio->GetMaximum(), 100.)); 
  h_ratio->SetMinimum(0.001); 
  h_ratio->SetMaximum(0.41); 
  h_ratio->Draw("ep same");
    
  //h_ratio->Fit("f1", "R", "", 1500, 3900);
  //TF1 *f1 = new TF1("f1","[0] +[1]*x",0,5);


  TF1 *f1 = new TF1("f1",fline,1500,3900,4);  
  //cout << width << " " << excl.size() << endl;
  //TF1 *fl = new TF1("fl",fline,0,5,2)
  //std::vector<double> excl = excludedbins(h_ratio);
  excl = excludedbins(h_ratio);
  width = binwidth(h_ratio);
  //cout << "size outside function: " << excl.size() << endl;
  reject = kFALSE; //kTRUE;
  h_ratio->Fit("f1", "R");
   
  gStyle->SetOptFit(1);
  f1->SetLineColor(kRed);

  c1->Update();

  gStyle->SetOptFit(1);
  h_ratio->Draw("ep same");
  h_ratio->SetTitle("");
  h_ratio->GetXaxis()->SetTitle("m_{T} (GeV)");
  h_ratio->GetYaxis()->SetTitle("TF");
  h_ratio->GetYaxis()->SetTitleOffset(0.3);
  h_ratio->GetXaxis()->SetTitleSize(0.1);
  h_ratio->GetYaxis()->SetTitleSize(0.1);
  h_ratio->GetXaxis()->SetLabelSize(0.1);
  h_ratio->GetYaxis()->SetLabelSize(0.1);

  TLatex *t = new TLatex();
  t->SetNDC();
  t->SetTextAlign(22);
  t->SetTextSize(0.08);
  //t->SetTextFont(63);
  //t->SetTextSizePixels(22);
  t->DrawLatex(0.3,0.9,Form("#Chi^{2}/NDF: %f", f1->GetChisquare()/f1->GetNDF() ));
  //cout << f1->GetNDF() << endl;
  //cout << f1->GetChisquare() << endl;
  //t->DrawLatex(0.3,0.75,Form("y = %f x + %f", f1->GetParameter(1), f1->GetParameter(0) ));
  t->DrawLatex(0.3,0.75,Form("y = exp(%f x + %f) + (%f + %f x)", f1->GetParameter(1), f1->GetParameter(0), f1->GetParameter(2), f1->GetParameter(3)));
  //t->DrawLatex(0.3,0.75,Form("y = %f + %f x", f1->GetParameter(0), f1->GetParameter(1)));

  c1->Update();   

  gStyle->SetOptFit(1);
  
  c1->SaveAs("Fit_v8/2017/"+(TString)histo_num+"_"+histo_den+"_exppol1.pdf");
  c1->SaveAs("Fit_v8/2017/"+(TString)histo_num+"_"+histo_den+"_exppol1.png");
  c1->SaveAs("Fit_v8/2017/"+(TString)histo_num+"_"+histo_den+"_exppol1.root");
 
}
