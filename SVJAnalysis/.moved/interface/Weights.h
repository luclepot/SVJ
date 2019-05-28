#include "TFile.h"
#include "TH2.h"
#include <string>

class Weights{

 public:
  Weights( TFile * file , std::string name ){
    histoFile = file;
    wHisto = (TH2F*)histoFile->Get(name.c_str());
  }

  float getEff(float eta, float pt){ 
    int i(-1), j(-1);
    findBin( eta, pt, i, j);
    /*
    int i = wHisto->GetXaxis()->FindBin(eta);
    int j = wHisto->GetYaxis()->FindBin(pt);
    if(i == 0) 
      i = 1;
    if(j == wHisto->GetYaxis()->GetNbins() + 1) 
      j =  wHisto->GetYaxis()->GetNbins();
    */
    float eff = wHisto->GetBinContent(i,j) ;
    return eff;
  }
 
  float getErr(float eta, float pt)
  { 
    std::cout << "getting bin" << std::endl;
    int i(-1), j(-1);
    findBin( eta, pt, i, j);
    /*
  int i = wHisto->GetXaxis()->FindBin(eta);
  int j = wHisto->GetYaxis()->FindBin(pt);
    if(i == 0) 
      i = 1;
    if(j == wHisto->GetYaxis()->GetNbins() + 1) 
      j =  wHisto->GetYaxis()->GetNbins();
    */
    float err = wHisto->GetBinError(i,j) ;
    return err;
  }
 
  void findBin(float eta, float pt, int & i, int & j){ 
    i = wHisto->GetXaxis()->FindBin(eta);
    j = wHisto->GetYaxis()->FindBin(pt);
    if(i == 0) 
      i = 1;
    if(j == wHisto->GetYaxis()->GetNbins() + 1) 
      j =  wHisto->GetYaxis()->GetNbins();
  }

 private:
  TFile* histoFile;
  TH2F* wHisto;

};

