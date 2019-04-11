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

#define PATH "output/histos_nopu_2016/"

double crystalball_function(double x, double alpha, double n, double sigma, double mean) {
    
  if (sigma < 0.)     return 0.;
  double z = (x - mean)/sigma; 
  if (alpha < 0) z = -z; 
  double abs_alpha = std::abs(alpha);

  if (z  > - abs_alpha)
    return std::exp(- 0.5 * z * z);
  else {

    double nDivAlpha = n/abs_alpha;
    double AA =  std::exp(-0.5*abs_alpha*abs_alpha);
    double B = nDivAlpha -abs_alpha;
    double arg = nDivAlpha/(B-z);
    return AA * std::pow(arg,n);
  }
}


std::vector<double> excludedbins(TH1F *histo) {
  std::vector<double> excluded;
  for(int i=1; i<=histo->GetNbinsX(); i++){
    if(histo->GetBinError(i)/histo->GetBinContent(i)>0.30){
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

Double_t f_bkg(Double_t *x, Double_t *par)
{
  Bool_t reject_bin = kFALSE;
  for(int i=0; i<excl.size(); i++){
    if(x[0]>excl.at(i) && x[0]<(excl.at(i)+width)){
      reject_bin = kTRUE;
    }
  }
  if(reject && reject_bin){
    TF1::RejectPoint();
    return 0;
  }

  //return exp(par[0]+ (x[0]/(13*1000))*par[1] + pow(x[0]/(13*1000),2)*par[2]);
  return (par[0]*pow((1-(x[0]/(13*1000))),par[1])/pow(x[0]/(13*1000),(par[2]+par[3]*log(x[0]/(13*1000))+ par[4] * pow(log(x[0]/(13*1000)),2))));
}

Double_t f_sig(Double_t *x, Double_t *par)
{
  Bool_t reject_bin = kFALSE;
  for(int i=0; i<excl.size(); i++){
    if(x[0]>excl.at(i) && x[0]<(excl.at(i)+width)){
      reject_bin = kTRUE;
    }
  }
  if(reject && reject_bin){
    TF1::RejectPoint();
    return 0;
  }

  static Double_t pi = 3.1415926535; 

  const Double_t xx =x[0]; 

  const Double_t width = (xx > par[0]) ? par[1] : par[2];
  const Double_t arg    = pow(((xx-par[0])/width),2);
  const Double_t ampl  = par[3];

  //return ampl*exp(-arg/2) + par[4] + xx*(par[5] + xx * par[6]); 
  //return (par[0]*crystalball_function(x[0], par[3], par[4], par[2], par[1]));
  return par[0] * exp(-0.5* (pow( ((x[0]-par[1])/par[2]),2)));
}


void Fit(const char* histo, const char* sample, int mZ = 0){

  gROOT->ForceStyle();

  std::string sample_str = sample;

  TH1F *h;
  int range_min = -1, range_max = -1;

  if(sample_str.compare("background")==0){  
    TFile *qcd = new TFile((PATH+string("QCD.root")).c_str());
    TFile *tt = new TFile((PATH+string("TT.root")).c_str());
    TFile *wjets = new TFile((PATH+string("WJets.root")).c_str());
    TFile *zjets = new TFile((PATH+string("ZJets.root")).c_str());

    TH1F *h_qcd = (TH1F*)qcd->Get(histo);
    TH1F *h_tt = (TH1F*)tt->Get(histo);
    TH1F *h_wjets = (TH1F*)wjets->Get(histo);
    TH1F *h_zjets = (TH1F*)zjets->Get(histo);
   
    THStack *hs = new THStack("hs","Stacked 1D histograms");
    hs->Add(h_tt);
    hs->Add(h_qcd);
    hs->Add(h_zjets);
    hs->Add(h_wjets);
    h = (TH1F*) hs->GetStack()->Last();  

    range_min = 1500;
    range_max = 3900;
  } else{
    TFile *signal = new TFile((PATH+sample_str+string(".root")).c_str());
    cout << "opening file: " << PATH+sample_str+string(".root") << endl;
    h = (TH1F*)signal->Get(histo);

    range_min = 0;
    range_max = 5000;

  }

  TF1 *f1;
  
  if(sample_str.compare("background")==0){
    f1 = new TF1("f1",f_bkg,range_min,range_max,5);

    excl = excludedbins(h);
    width = binwidth(h);
    reject = kFALSE;
  
    f1->SetParameter(0,0.001);
    f1->SetParameter(1,8);
    f1->SetParameter(2,4);
    f1->SetParameter(3,0.1);
    f1->FixParameter(4,0.);                                                                                                    
  }else{
    f1 = new TF1("f1",f_sig,range_min,range_max,3);
    f1->SetParameter(0,1);
    f1->SetParameter(1,mZ);
    f1->SetParameter(2,100);

    //f1->SetParameter(1, mZ);
    //f1->SetParameter(2, 1000);
    //f1->SetParameter(3, 100);
    //f1->SetParameter(4, 1);
    //f1->SetParameter(0, 1);
    //(x, [Alpha], [N], [Sigma], [Mean])
    //(par[0]*crystalball_function(x[0], par[3], par[4], par[2], par[1]));

    //par[0] * exp(-0.5* (pow( ((x[0]-par[1])/par[2]),2)));

  }

  //parameter 1
  //f1->SetParameter(0,0.001);
  //f1->FixParameter(1,0);
  //f1->SetParameter(2,10);                                                                                                       //f1->FixParameter(3,0.);                                                                                                       //f1->FixParameter(4,0.);                                                                                                                     
  //parameter 2
  //f1->SetParameter(0,0.001);
  //f1->SetParameter(1,8);
  //f1->SetParameter(2,4);                                                                                                      
  //f1->FixParameter(3,0.);                                                                                                       //f1->FixParameter(4,0.);                                                                                                                    
  //parameter 3
             
  //parameter 4
  //f1->SetParameter(0,0.001);
  //f1->SetParameter(1,8);
  //f1->SetParameter(2,4);
  //f1->SetParameter(3,0.1);
  //f1->SetParameter(4,0.0);
 
  TCanvas *c1 = new TCanvas("c1","c1",0,0,600,600);
  TPad *pad1 = new TPad("pad1", "pad1", 0, 0.35, 1, 1.0);
  pad1->SetBottomMargin(0); 
  pad1->SetGridx();         
  pad1->SetGridy();         
  pad1->Draw();             
  pad1->cd();               

  ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(10000); 
  ROOT::Math::MinimizerOptions::SetDefaultTolerance(1); 

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);
  //gPad->SetLogy();
  
  h->SetLineColor(kBlack);
  h->SetMarkerColor(kBlack);
  h->SetMarkerStyle(20);
  h->SetMarkerSize(1);

  h->SetTitle("");
  h->GetYaxis()->SetTitle("Number of events");
  h->GetXaxis()->SetTitle("m_{T} (GeV)");
  h->GetXaxis()->SetRangeUser(range_min,range_max);
  h->Draw("ep");

  h->Fit("f1", "R");
  cout << f1->GetParameter(0) << " " << f1->GetParameter(1) << " " << f1->GetParameter(2) << " " << f1->GetParameter(3) << " " << endl;

  f1->SetLineColor(kRed);
  
  /*Create a histogram to hold the confidence intervals*/
  /*TH1D *hint = new TH1D("hint",
			"Fitted gaussian with .95 conf.band", 100, 1400, 4200);
  (TVirtualFitter::GetFitter())->GetConfidenceIntervals(hint);
  hint->SetStats(kFALSE);
  hint->SetFillColor(kRed-9);
  hint->Draw("e3 same");
  h->Draw("ep same");
  */
  
  h->SetMaximum(1.5*h->GetMaximum());

  leg = new TLegend(0.12,0.8,0.58,0.88);
  leg->SetFillColor(0);
  leg->SetBorderSize(0);
  leg->AddEntry(h, sample ,"ep");
  leg->AddEntry(f1, "Fit", "l");
  leg->Draw();

  TLatex *t = new TLatex();
  t->SetNDC();
  t->SetTextAlign(22);
  t->SetTextFont(63);
  t->SetTextSizePixels(22);
  
  std::string histo_str = histo;

  if(histo_str.compare("h_Mt_BDT2")==0) {
    t->DrawLatex(0.75, 0.6,"BDT2 category");
  } else if(histo_str.compare("h_Mt_BDT1")==0){
    t->DrawLatex(0.75, 0.6,"BDT1 category");
  } else if(histo_str.compare("h_Mt_BDT0")==0){
    t->DrawLatex(0.75, 0.6,"BDT0 category");
  } else if(histo_str.compare("h_Mt")==0){
    t->DrawLatex(0.75, 0.6 ,"cut-based category");
  }

  c1->Update();
  c1->cd();          

  TPad *pad2 = new TPad("pad2", "pad2", 0, 0.05, 1, 0.3);
  pad2->SetTopMargin(0);
  pad2->SetBottomMargin(0.2);
  pad2->SetGridx(); 
  pad2->SetGridy(); 
  pad2->Draw();
  pad2->cd();					
  pad2->Clear();
  c1->Update();
  c1->Modified();

  TH1F *h_ratio = (TH1F*)h->Clone("h_ratio");
  h_ratio->SetDirectory(0);
  h_ratio->GetXaxis()->SetRangeUser(range_min,range_max);
  
  for(int i=1; i<=h_ratio->GetNbinsX(); i++){
    if(sample_str.compare("background")==0){
      h_ratio->SetBinContent(i, h->GetBinContent(i)/(f1->GetParameter(0)*pow((1-(h->GetXaxis()->GetBinCenter(i)/(13*1000))),f1->GetParameter(1))/(pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),(f1->GetParameter(2)+f1->GetParameter(3)*log(h->GetXaxis()->GetBinCenter(i)/(13*1000))  + f1->GetParameter(4)*pow(log( h->GetXaxis()->GetBinCenter(i)/(13*1000)),2) ) ))));

      h_ratio->SetBinError(i, h->GetBinError(i) / (f1->GetParameter(0)*pow((1-(h->GetXaxis()->GetBinCenter(i)/(13*1000))),f1->GetParameter(1))/(pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),(f1->GetParameter(2)+f1->GetParameter(3)*log(h->GetXaxis()->GetBinCenter(i)/(13*1000))  + f1->GetParameter(4)*pow(log( h->GetXaxis()->GetBinCenter(i)/(13*1000)),2) ) ))));  
    } else{

      h_ratio->SetBinContent(i, h->GetBinContent(i)/ (f1->GetParameter(0) * exp(-0.5* (pow( ((h->GetXaxis()->GetBinCenter(i)-f1->GetParameter(1))/f1->GetParameter(2)),2)))) );

      h_ratio->SetBinError(i, h->GetBinError(i)/ (f1->GetParameter(0) * exp(-0.5* (pow( ((h->GetXaxis()->GetBinCenter(i)-f1->GetParameter(1))/f1->GetParameter(2)),2)))) );
    }
    //h_ratio->SetBinError(i, ( (f1->GetParameter(0)*pow((1-(h->GetXaxis()->GetBinCenter(i)/(13*1000))),f1->GetParameter(1))/(pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),(f1->GetParameter(2)+f1->GetParameter(3)*log(h->GetXaxis()->GetBinCenter(i)/(13*1000))  + f1->GetParameter(4)*pow(log( h->GetXaxis()->GetBinCenter(i)/(13*1000)),2) ) ))) - h->GetBinContent(i) )/h->GetBinError(i) );

    //exp(par[0]+ (x[0]/(13*1000))*par[1] + pow(x[0]/(13*1000),2)*par[2]);

    //h_ratio->SetBinContent(i, h->GetBinContent(i)/(exp(f1->GetParameter(0) + f1->GetParameter(1)*pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),1) + f1->GetParameter(2)*pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),2) )));

    //h_ratio->SetBinError(i, h->GetBinError(i)/( h->GetBinContent(i)/(exp(f1->GetParameter(0) + f1->GetParameter(1)*pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),1) + f1->GetParameter(2)*pow(h->GetXaxis()->GetBinCenter(i)/(13*1000),2) ))  ));
  }

  h_ratio->GetFunction("f1")->SetBit(TF1::kNotDraw);
  h_ratio->GetXaxis()->SetRangeUser(range_min,range_max);
  h_ratio->SetLineColor(kBlack);
  h_ratio->SetMinimum(0.1); 
  h_ratio->SetMaximum(2.1);
  h_ratio->Sumw2();
  h_ratio->SetStats(0);     
  h_ratio->SetMarkerStyle(21);
  c1->Update();
  c1->Modified();
  h_ratio->Draw("ep");
  c1->Update();
  c1->Modified();

  h_ratio->GetXaxis()->SetTitle("m_{T} (GeV)");
  h_ratio->GetYaxis()->SetTitle("Ratio");
  h_ratio->GetYaxis()->SetTitleOffset(0.3);
  h_ratio->GetXaxis()->SetTitleSize(0.1);
  h_ratio->GetYaxis()->SetTitleSize(0.1);
  h_ratio->GetXaxis()->SetLabelSize(0.1);
  h_ratio->GetYaxis()->SetLabelSize(0.1);

  c1->Update(); 

  c1->SaveAs("Fit_v12/2016/"+(TString)histo+"_"+(TString)sample+".pdf");
  c1->SaveAs("Fit_v12/2016/"+(TString)histo+"_"+(TString)sample+".png");
  c1->SaveAs("Fit_v12/2016/"+(TString)histo+"_"+(TString)sample+".root");
 
}
