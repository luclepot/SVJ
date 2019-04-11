#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TBranch.h"
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
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "SVJ/SVJAnalysis/interface/Weights.h"
#include "SVJ/SVJAnalysis/interface/MT2Utility.h"
#include "SVJ/SVJAnalysis/interface/mt2w_bisect.h"
#include "SVJ/SVJAnalysis/interface/mt2bl_bisect.h"
#include "SVJ/SVJAnalysis/interface/Mt2Com_bisect.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "SVJ/SVJAnalysis/interface/kFactors.h"

using namespace std;

typedef vector<double> vdouble;
typedef vector<float> vfloat;
typedef vector<int> vint;
typedef vector<bool> vbool;
typedef vector<string> vstring;

enum weightedSysts { NOSYST=0, PUUP=1, PUDOWN=2,MAXSYSTS=3};
enum theoSysts {SCALEUP=101,SCALEDOWN=102, NNPDF1=100, NNPDF2=102};
int wLimit =150;

struct systWeights{
  void initHistogramsSysts(TH1F** histo, TString name, TString, int, float, float);
  void createFilesSysts(TFile ** allFiles, TString basename, TString opt="RECREATE");
  void fillHistogramsSysts(TH1F** histo, float v, float W, double *wcats= NULL,bool verbose=false);
  void fillHistogramsSysts(TH1F** histo, float v, float W,  float *systWeights, int nFirstSysts=(int)MAXSYSTS, double *wcats=NULL, bool verbose=false);
  void writeHistogramsSysts(TH1F** histo, TFile ** allFiles );
  void writeSingleHistogramSysts(TH1F* histo,TFile ** allMyFiles);
  void setMax(int max);
  void setMaxNonPDF(int max);
  void setSystValue(string name, double value, bool mult=false);
  void setSystValue(int systPlace, double value, bool mult=false);
  void setPDFWeights(float * wpdfs, double * xsections, int numPDFs, float wzero=1.0, bool mult=true);
  void setQ2Weights(float q2up, float q2down, float wzero=1.0,bool mult=true);
  void setTWeight(float tweight, float wtotsample=1.0,bool mult=true);
  void setVHFWeight(int vhf,bool mult=true, double shiftval=0.65);
  void setPDFValue(int numPDF, double value);
  double getPDFValue(int numPDF);
  void setWeight(string name, double value, bool mult=false);
  void setWeight(int systPlace, double value, bool mult=false);
  void prepareDefault(bool addDefault, bool addPDF, bool addQ2, bool addTopPt, bool addVHF, bool addTTSplit, int numPDF=102);
  void addSyst(string name);
  void addSystNonPDF(string name);
  void setWCats(double *wcats);
  
  void addkFact(string name);
  void setkFact(string name,float kfact_nom, float kfact_up,float kfact_down,  bool mult=true);

  void copySysts(systWeights sys);
  void calcPDFHisto(TH1F** histos, TH1F* singleHisto, double scalefactor=1.0, int c = 0);
  void setOnlyNominal(bool useOnlyNominal=false);
  bool onlyNominal;
  bool addPDF, addQ2, addTopPt, addVHF, addTTSplit;
  int maxSysts, maxSystsNonPDF;
  int nPDF;
  int nCategories;
  float weightedSysts[150];
  double wCats[10];
  string weightedNames[150];
  string categoriesNames[10];
};

void callme(){
  std::cout<<" NaN value"<<std::endl;
}

int main(int argc, char **argv) {

  TBenchmark bench;
  
  std::cout<<"Let's start"<<endl;
 
  string sample(argv[1]) ;
  std::cout<<"sample: "<<sample<<endl;

  string path(argv[2]);
  std::cout<<"File list to open: "<<path<<endl;
    
  string sys(argv[3]);
  std::cout<<"systematics: "<<sys<<endl;

  string sync(argv[4]);
  std::cout<<"synchro: "<<sync<<endl;

  string isData(argv[5]);
  std::cout<<"isData: "<<isData<<endl;

  std::string outdir(argv[6]);
  std::cout<<"Output directory: "<<outdir<<endl;

  TString path_ = path ; 
  std::cout<<"File to open: "<<path_<<endl;
  
  std::cout << "Loading file collection from " << path << std::endl;
  TFileCollection fc(sample.c_str(),sample.c_str(),path.c_str());
  std::cout << "Files found : " << fc.GetNFiles() << std::endl;

  TLorentzVector Rematch(TLorentzVector gp, std::vector<TLorentzVector> jet, float dR);
  float check_match(TLorentzVector gp1, TLorentzVector gp2, std::vector<TLorentzVector> jet, float dR);

  TH1F * initproduct(TH1F * h_A,TH1F* h_B, int rebinA = 1, int rebinB=1,double integral = -1.);
  TH1F * makeproduct(TH1F * h_A,TH1F* h_B, int rebinA = 1, int rebinB=1,double integral = -1.);
  TH1F * makeproduct(TH2F * h);

  TString syststr = "";
  string syststrname = "";

  if (sys != "noSys"){syststr= "_"+ sys; syststrname= "_"+ sys;}
  
  string reportDir = outdir+"/txt";
  string reportName = reportDir+"/SelectedEvents_"+sample+syststrname+".txt";

  const char* noTTlabel = "TT";
  const char* TTJetslabel = "TTJets";
  const char* TTJets_DiLeptlabel = "TTJets_DiLept";
  const char* TTJets_DiLept_genMET150label = "TTJets_DiLept_genMET150";
  const char* TTJets_SingleLeptFromTlabel = "TTJets_SingleLeptFromT";
  const char* TTJets_SingleLeptFromT_genMET150label = "TTJets_SingleLeptFromT_genMET150";
  const char* TTJets_SingleLeptFromTbarlabel = "TTJets_SingleLeptFromTbar";
  const char* TTJets_SingleLeptFromTbar_genMET150label = "TTJets_SingleLeptFromTbar_genMET150";
  const char* TTJets_HT600to800label = "TTJets_HT600to800";
  const char* TTJets_HT800to1200label = "TTJets_HT800to1200";
  const char* TTJets_HT1200to2500label = "TTJets_HT1200to2500";
  const char* TTJets_HT2500toInflabel = "TTJets_HT2500toInf";
  
  bool tt_stitching_TTJets=false, tt_stitching_TTJets_DiLept=false, tt_stitching_TTJets_DiLept_genMET150=false, tt_stitching_TTJets_SingleLeptFromT=false, tt_stitching_TTJets_SingleLeptFromT_genMET150=false, tt_stitching_TTJets_SingleLeptFromTbar=false, tt_stitching_TTJets_SingleLeptFromTbar_genMET150=false, tt_stitching_TTJets_HT600to800=false, tt_stitching_TTJets_HT800to1200=false, tt_stitching_TTJets_HT1200to2500=false, tt_stitching_TTJets_HT2500toInf=false, tt_stitching_noTT=false;

  if( !strncmp(sample.c_str(), TTJetslabel , strlen(TTJetslabel))) {tt_stitching_TTJets=true;}
  if( !strncmp(sample.c_str(), TTJets_DiLeptlabel , strlen(TTJets_DiLeptlabel))) {tt_stitching_TTJets_DiLept=true;}
  if( !strncmp(sample.c_str(), TTJets_DiLept_genMET150label , strlen(TTJets_DiLept_genMET150label))) {tt_stitching_TTJets_DiLept_genMET150=true;}
  if( !strncmp(sample.c_str(), TTJets_SingleLeptFromTlabel , strlen(TTJets_SingleLeptFromTlabel))) {tt_stitching_TTJets_SingleLeptFromT=true;}
  if( !strncmp(sample.c_str(), TTJets_SingleLeptFromT_genMET150label , strlen(TTJets_SingleLeptFromT_genMET150label))) {tt_stitching_TTJets_SingleLeptFromT_genMET150=true;}
  if( !strncmp(sample.c_str(), TTJets_SingleLeptFromTbarlabel , strlen(TTJets_SingleLeptFromTbarlabel))) {tt_stitching_TTJets_SingleLeptFromTbar=true;}
  if( !strncmp(sample.c_str(), TTJets_SingleLeptFromTbar_genMET150label , strlen(TTJets_SingleLeptFromTbar_genMET150label))) {tt_stitching_TTJets_SingleLeptFromTbar_genMET150=true;}
  if( !strncmp(sample.c_str(), TTJets_HT600to800label , strlen(TTJets_HT600to800label))) {tt_stitching_TTJets_HT600to800=true;}
  if( !strncmp(sample.c_str(), TTJets_HT800to1200label , strlen(TTJets_HT800to1200label))) {tt_stitching_TTJets_HT800to1200=true;}
  if( !strncmp(sample.c_str(), TTJets_HT1200to2500label , strlen(TTJets_HT1200to2500label))) {tt_stitching_TTJets_HT1200to2500=true;}
  if( !strncmp(sample.c_str(), TTJets_HT2500toInflabel , strlen(TTJets_HT2500toInflabel))) {tt_stitching_TTJets_HT2500toInf=true;}
  if( strncmp(sample.c_str(), noTTlabel, 2)!=0 ) {tt_stitching_noTT=true;}  
  
  const char* data2016label = "Run2016";
  const char* data2017label = "Run2017";

  bool isData2017 = false;
  bool isData2016 = false;   
      
  if( !strncmp(sample.c_str(), data2016label , strlen(data2016label))) {isData2016=true;}
  if( !strncmp(sample.c_str(), data2017label , strlen(data2017label))) {isData2017=true;}

  TString weightedSystsNames (weightedSysts sy);
  void initHistogramsSysts(TH1F* histo[(int)MAXSYSTS],TString, TString, int, float, float , bool useOnlyNominal=false);
  void createFilesSysts(  TFile * allFiles[(int)MAXSYSTS], TString basename, bool useOnlyNominal=false, TString opt="RECREATE");
  void fillHistogramsSysts(TH1F* histo[(int)MAXSYSTS], float v, float W, float systWeight[(int)MAXSYSTS] , bool useOnlyNominal=false);
  void writeHistogramsSysts(TH1F* histo[(int)MAXSYSTS], TFile * allFiles[(int)MAXSYSTS] , bool useOnlyNominal=false);
  void writeSingleHistogramSysts(TH1F* histo,TFile * allMyFiles[(int)MAXSYSTS],bool useOnlyNominal=false);

  systWeights systZero, systSVJ;
  int maxSysts=0;
  bool addPDF=false,addQ2=false, addTopPt=false,addVHF=false/*,addWZNLO=false*/, addTTSplit=false;

  // change in true
  addPDF=false;
  addQ2=false;

  int nPDF=102;
  if(isData=="DATA"){addPDF=false, addQ2=false;}
  systZero.prepareDefault(true, addQ2, addPDF, addTopPt,addVHF,addTTSplit);
  
  maxSysts= systZero.maxSysts;

  ofstream fileout;
  fileout.open(reportName.c_str(),ios::in | ios::out | ios::trunc);
  fileout<<"RunNumber EvtNumber Lumi "<<std::endl;

  float systWeightsSVJ[maxSysts];
  float w_pdfs[nPDF], w, w_pu, w_q2up,w_q2down, w_zero;

  TFile * allMyFiles[maxSysts];
  TString outfile = outdir + "/res/"+sample + ".root";
  TString outfile_hotspot = outdir + "/res_hotspot/"+sample + ".root";
  TFile fout(outfile, "recreate");
  cout << "output file name: " <<  outfile << endl;
  std::cout<<"File to open: "<<path_<<endl;
  /*
  TString treePath = "TreeMaker2/PreSelection";
  TString treePathNEvents = "TreeMaker2/PreSelection";
  if(sys=="jesUp") treePath = "TreeMaker2/ttDM__jes__up";
  else if(sys=="jesDown") treePath = "TreeMaker2/ttDM__jes__down";
  else if(sys=="jerUp") treePath = "TreeMaker2/ttDM__jer__up";
  else if(sys=="jerDown")treePath = "TreeMaker2/ttDM__jer__down";
  */
  
  TString treePath = "tree";
  TString treePathNEvents = "tree";

  //TString treePath = "TreeMaker2/PreSelection";
  //TString treePathNEvents = "TreeMaker2/PreSelection";


  /* 
  //to uncomment if tree name for jes and jer up and down different from nominal
  if(sys=="jesUp") treePath = "tree/ttDM__jes__up";
  else if(sys=="jesDown") treePath = "tree/ttDM__jes__down";
  else if(sys=="jerUp") treePath = "tree/ttDM__jer__up";
  else if(sys=="jerDown")treePath = "tree/ttDM__jer__down";
  */

  bench.Start("NEvents");
  //  TChain chainNEvents(treePathNEvents);
  TChain chainNEvents(treePath);
  chainNEvents.AddFileInfoList(fc.GetList());
  Int_t nEvents = (Int_t)chainNEvents.GetEntries();
  bench.Stop("NEvents");
  bench.Print("NEvents");

  TH1D totalWeightTop("w_top_total","Top pt reweighting: overall sample weight",2000,0,2.0);
  //chainNEvents.Project("w_top_total","Event_T_Weight","Event_T_Weight!=1.00");
  double topWeight=totalWeightTop.GetMean();
  cout << "totaltopweight is "<< topWeight<<endl;
  if(topWeight==0)topWeight=1;

  //bench.Start("NEventsPrePres");
  TChain chain(treePath);
  chain.AddFileInfoList(fc.GetList());

  Int_t nEventsPrePres = -1;
  TH1F * h_nEventProc = new TH1F("h_nEventProc", "h_nEventProc", 1, 0, 1);
  h_nEventProc = (TH1F*)chain.GetFile()->Get("nEventProc");
  nEventsPrePres = h_nEventProc->Integral();
  
  //nEventsPrePres = (Int_t)chain.GetEntries();
  
  std::cout<<"--> --> Number of Events: "<<nEvents<< " after preselection "<< nEventsPrePres << endl;
  bench.Stop("NEventsPrePres");
  bench.Print("NEventsPrePres");
  
  std::string samplestr(sample.c_str());
  
  //Q2 and PDF splitting                                                                                                                                                                                                    
  double q2SplittedWeight=1.;
  if(addQ2){
    if((samplestr).find("SVJ") != std::string::npos){
      TH1D splittedWeightQ2("w_q2_splitted","Q2 splitting: overall sample weight",2000,0,2.0);
      //chain.Project("w_q2_splitted","Event_LHEWeight4/Event_LHEWeight0");
      chain.Project("w_q2_splitted","PDFweights[4]");
      q2SplittedWeight=splittedWeightQ2.GetMean();
    }
    cout << "q2SplittedWeight is "<< q2SplittedWeight<<endl;
  }
  
  double PDFsplittedWeight[nPDF];

  if(addPDF){
      for (int i = 1 ; i <= nPDF ; ++i) 
	{
	  PDFsplittedWeight[i]=1.;
	  
	    if((samplestr).find("SVJ") != std::string::npos){
	    stringstream pdfss;
	    //pdfss<<(i+8);
	    pdfss<<(i);
	    string pstr=(pdfss.str());
	    TH1D splittedWeightPDF("w_pdf_splitted","PDF splitting: overall sample weight",2000,0,2.0);
	    //chain.Project("w_pdf_splitted",(("Event_LHEWeight"+pstr+"/Event_LHEWeight0").c_str())); 
	    chain.Project("w_pdf_splitted",(("PDFweights["+pstr + "] / PDFweights[0]").c_str())); 
	    PDFsplittedWeight[i]=splittedWeightPDF.GetMean();
	    cout << "PDFSplittedWeight " << i << " is " << PDFsplittedWeight[i]<<endl;
	  }
	  
    }
  }

  /* Configuring BDT */
  Float_t bdt_mult, bdt_axisminor, bdt_girth, bdt_tau21, bdt_tau32, bdt_msd, bdt_deltaphi, bdt_pt, bdt_eta, bdt_mt;
  //  Float_t mult, axisminor, girth, tau21, tau32, msd, deltaphi, pt, eta, mt; 
  TMVA::Reader reader( "!Color:!Silent" );
  reader.AddVariable( "mult", &bdt_mult ); 
  reader.AddVariable( "axisminor", &bdt_axisminor );
  reader.AddVariable( "girth", &bdt_girth );
  reader.AddVariable( "tau21", &bdt_tau21 );
  reader.AddVariable( "tau32", &bdt_tau32 );
  reader.AddVariable( "msd", &bdt_msd );
  reader.AddVariable( "deltaphi", &bdt_deltaphi );
  reader.AddSpectator( "spec1 := pt", &bdt_pt);
  reader.AddSpectator( "spec1 := eta", &bdt_eta);
  reader.AddSpectator( "spec1 := mt", &bdt_mt);

  const std::string cmssw_base = getenv("CMSSW_BASE");
  //const std::string weightsfile = cmssw_base + std::string("/src/SVJ/SVJAnalysis/data/TMVAClassification_BDTG.weights.xml");
  const std::string weightsfile = cmssw_base + std::string("/src/SVJ/SVJAnalysis/mZ3000/TMVAClassification_BDTG.weights.xml");
  reader.BookMVA("BDTG", weightsfile.c_str() );

  double metFull_Pt=0., metFull_Phi=0.;//, metFull_Px=0., metFull_Py=0.;

  int sizeMax_gen=50000;
  std::vector<int>* triggerPassPtr(0x0);

  std::vector<TLorentzVector>* genPartsPtr(0x0);
  int genPart_pdgId[sizeMax_gen], genPart_Status[sizeMax_gen];

  std::vector<TLorentzVector>* jetsAK8CHSPtr(0x0);
  std::vector<TLorentzVector>* MuonsPtr(0x0); 
  std::vector<TLorentzVector>* ElectronsPtr(0x0);

  std::vector<TLorentzVector>* GenElectronsPtr(0x0);
  std::vector<TLorentzVector>* GenMuonsPtr(0x0);
  std::vector<TLorentzVector>* GenTausPtr(0x0);

  double madHT=0, GenMET=0;

  std::vector<double>* JetsAK8_NHFPtr(0x0);
  std::vector<double>* JetsAK8_CHFPtr(0x0);

  double Ht(0.), MT(0.);
  int nMuons=-1, nElectrons=-1;
  ULong64_t EventNumber(0.);
  
  bool BadChargedCandidateFilter = 0, BadPFMuonFilter = 0;
  int EcalDeadCellTriggerPrimitiveFilter = 0,   HBHEIsoNoiseFilter = 0,  HBHENoiseFilter = 0,  globalTightHalo2016Filter = 0, NVtx = 0;

  std::vector<int>* multPtr(0x0);
  std::vector<double>* axisminorPtr(0x0);
  std::vector<double>* girthPtr(0x0);
  std::vector<double>* tau1Ptr(0x0); 
  std::vector<double>* tau2Ptr(0x0); 
  std::vector<double>* tau3Ptr(0x0); 
  std::vector<double>* msdPtr(0x0);
  std::vector<bool>* jetsIDPtr(0x0);
  std::vector<double>* muMiniIsoPtr(0x0);
  std::vector<double>* PDFweightsPtr(0x0);
 
  double deltaphi1, deltaphi2;
  double DeltaPhiMin;
  
  //float mult, axisminor, girth, tau21, tau32, msd, deltaphi;

  int NumInteractions =0;
  double puSysDown, puSysUp, puWeight;
  chain.SetBranchAddress("NumInteractions", &NumInteractions);
  chain.SetBranchAddress("puSysDown", &puSysDown);
  chain.SetBranchAddress("puSysUp", &puSysUp);
  chain.SetBranchAddress("puWeight", &puWeight);

  chain.SetBranchAddress("TriggerPass", &triggerPassPtr);

  chain.SetBranchAddress("EvtNum", &EventNumber);
  chain.SetBranchAddress("HT", &Ht);
  chain.SetBranchAddress("MT_AK8", &MT);

  chain.SetBranchAddress("MET",&metFull_Pt);
  chain.SetBranchAddress("METPhi",&metFull_Phi);
  //chain.SetBranchAddress("MET",&metFull_Px);
  //chain.SetBranchAddress("MET",&metFull_Py);
  
  if(isData=="MC"){
    chain.SetBranchAddress("GenParticles", &genPartsPtr);
    chain.SetBranchAddress("GenParticles_PdgId",&genPart_pdgId);
    chain.SetBranchAddress("GenParticles_Status",&genPart_Status);
  }
  
  chain.SetBranchAddress("JetsAK8",&jetsAK8CHSPtr);
  chain.SetBranchAddress("Muons",&MuonsPtr);
  chain.SetBranchAddress("Muons_MiniIso",&muMiniIsoPtr);
  chain.SetBranchAddress("Electrons",&ElectronsPtr);

  chain.SetBranchAddress("JetsAK8_neutralHadronEnergyFraction", &JetsAK8_NHFPtr);
  chain.SetBranchAddress("JetsAK8_chargedHadronEnergyFraction", &JetsAK8_CHFPtr);

  chain.SetBranchAddress("NMuons",&nMuons);
  chain.SetBranchAddress("NElectrons",&nElectrons);

  chain.SetBranchAddress("BadChargedCandidateFilter", &BadChargedCandidateFilter);
  chain.SetBranchAddress("BadPFMuonFilter", &BadPFMuonFilter);
  chain.SetBranchAddress("EcalDeadCellTriggerPrimitiveFilter", &EcalDeadCellTriggerPrimitiveFilter);
  chain.SetBranchAddress("HBHEIsoNoiseFilter", &HBHEIsoNoiseFilter);
  chain.SetBranchAddress("HBHENoiseFilter", &HBHENoiseFilter);  
  chain.SetBranchAddress("globalTightHalo2016Filter", &globalTightHalo2016Filter);
  chain.SetBranchAddress("NVtx", &NVtx);
  
  chain.SetBranchAddress("DeltaPhiMin_AK8", &DeltaPhiMin);
  chain.SetBranchAddress("JetsAK8_ID", &jetsIDPtr);
  chain.SetBranchAddress("JetsAK8_multiplicity", &multPtr);
  chain.SetBranchAddress("JetsAK8_axisminor", &axisminorPtr);
  chain.SetBranchAddress("JetsAK8_girth", &girthPtr);
  chain.SetBranchAddress("JetsAK8_NsubjettinessTau1", &tau1Ptr);
  chain.SetBranchAddress("JetsAK8_NsubjettinessTau2", &tau2Ptr);
  chain.SetBranchAddress("JetsAK8_NsubjettinessTau3", &tau3Ptr);

  chain.SetBranchAddress("JetsAK8_softDropMass", &msdPtr);
  chain.SetBranchAddress("DeltaPhi1_AK8", &deltaphi1);
  chain.SetBranchAddress("DeltaPhi2_AK8", &deltaphi2);

  chain.SetBranchAddress("GenMET", &GenMET);
  chain.SetBranchAddress("madHT", &madHT);
  chain.SetBranchAddress("GenElectrons",&GenElectronsPtr);
  chain.SetBranchAddress("GenMuons",&GenMuonsPtr);
  chain.SetBranchAddress("GenTaus",&GenTausPtr);

  if(isData=="MC") chain.SetBranchAddress("PDFweights", &PDFweightsPtr);

  //if(isData=="MC") chain.SetBranchAddress("PDFWeights", &w_zero);
  
  /********************************************************************************/
  /**************                    Histogram booking              ***************/
  /********************************************************************************/

  float p=1;

  float x=0.;
  float NE=0.;

  if(strcmp (sample.c_str(),"QCDHT100To200") == 0) NE=(float)(16399902.); //347700 351300                                                                                     
  else if(strcmp (sample.c_str(),"QCDHT200To300") == 0) NE=(float)((18398764.)); //32100 31630                                                                                   
  else if(strcmp (sample.c_str(),"QCDHT300To500") == 0) NE=(float)((18398764.)); //32100 31630                                                                                   
  else if(strcmp (sample.c_str(),"QCDHT500To700") == 0) NE=(float)((18398764.)); //32100 31630                                                                                   
  else if(strcmp (sample.c_str(),"QCDHT700To1000") == 0) NE=(float)((15289380.)); //                                                                                            
  else if(strcmp (sample.c_str(),"QCDHT1500To2000") == 0) NE=(float)((3970819.));
  else if(strcmp (sample.c_str(),"QCDHT1000To1500") == 0) NE=(float)((4767100.));
  else if(strcmp (sample.c_str(),"QCDHT2000ToInf") == 0) NE=(float)((1913485.));

  //else if(strcmp (sample.c_str(),"WJetsHT100to200") == 0) NE=(float)(());
  else if(strcmp (sample.c_str(),"WJetsHT200to400") == 0) NE=(float)((19735128.));
  else if(strcmp (sample.c_str(),"WJetsHT400to600") == 0) NE=(float)((5677949.));
  else if(strcmp (sample.c_str(),"WJetsHT600to800") == 0) NE=(float)((14560421.));
  else if(strcmp (sample.c_str(),"WJetsHT800to1200") == 0) NE=(float)((1944423.));
  else if(strcmp (sample.c_str(),"WJetsHT1200to2500") == 0) NE=(float)((5455497.));
  else if(strcmp (sample.c_str(),"WJetsHT2500toInf") == 0) NE=(float)((2384260.));

  //else if(strcmp (sample.c_str(),"ZJetsHT100to200") == 0) NE=(float)(());
  else if(strcmp (sample.c_str(),"ZJetsHT200to400") == 0) NE=(float)((4532071.));
  else if(strcmp (sample.c_str(),"ZJetsHT400to600") == 0) NE=(float)((1020309.));
  else if(strcmp (sample.c_str(),"ZJetsHT600to800") == 0) NE=(float)((5600226.));
  else if(strcmp (sample.c_str(),"ZJetsHT800to1200") == 0) NE=(float)((1944423.));
  else if(strcmp (sample.c_str(),"ZJetsHT1200to2500") == 0) NE=(float)((296666.));
  else if(strcmp (sample.c_str(),"ZJetsHT2500toInf") == 0) NE=(float)((405752.));

  else if(strcmp (sample.c_str(),"TTJets") == 0) NE=(float)((103191488.));
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ1000rinv03") == 0) NE=(float)(12500.);
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ2000rinv03") == 0) NE=(float)(12500.);
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ3000rinv03") == 0) NE=(float)(12500.);
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ4000rinv03") == 0) NE=(float)(12003.);
  x = (float)(1000./NE);

  if(strcmp (sample.c_str(),"QCDHT300To500") == 0) p=(float)(x * (float)(347700)); //347700 351300                                                                                     
  else if(strcmp (sample.c_str(),"QCDHT500To700") == 0) p=(float)(x * (float)(32100)); //32100 31630                                                                                        
  else if(strcmp (sample.c_str(),"QCDHT700To1000") == 0) p=(float)(x * (float)(6831)); //                                                                                            
  else if(strcmp (sample.c_str(),"QCDHT1000To1500") == 0) p=(float)(x * (float)(1207));
  else if(strcmp (sample.c_str(),"QCDHT1500To2000") == 0) p=(float)(x * (float)(119.9));
  else if(strcmp (sample.c_str(),"QCDHT2000ToInf") == 0) p=(float)(x * (float)(25.24));
  //else if(strcmp (sample.c_str(),"WJetsHT100to200") == 0) p=(float)(1.21 *x * (float)(1345));
  else if(strcmp (sample.c_str(),"WJetsHT200to400") == 0) p=(float)(1.21 *x * (float)(359.7));
  else if(strcmp (sample.c_str(),"WJetsHT400to600") == 0) p=(float)(1.21 *x * (float)(48.91));
  else if(strcmp (sample.c_str(),"WJetsHT600to800") == 0) p=(float)(1.21 *x * (float)(12.05));
  else if(strcmp (sample.c_str(),"WJetsHT800to1200") == 0) p=(float)(1.21 *x * (float)(5.501));
  else if(strcmp (sample.c_str(),"WJetsHT1200to2500") == 0) p=(float)(1.21 *x * (float)(1.329));
  else if(strcmp (sample.c_str(),"WJetsHT2500toInf") == 0) p=(float)(1.21 * x * (float)(0.03216));
  //else if(strcmp (sample.c_str(),"ZJetsHT100to200") == 0) p=(float)(1.23 * x * (float)(280.35));
  else if(strcmp (sample.c_str(),"ZJetsHT200to400") == 0) p=(float)(1.23 * x * (float)(77.67));
  else if(strcmp (sample.c_str(),"ZJetsHT400to600") == 0) p=(float)(1.23 * x * (float)(10.73));
  else if(strcmp (sample.c_str(),"ZJetsHT600to800") == 0) p=(float)(1.23 * x * (float)(2.559));
  else if(strcmp (sample.c_str(),"ZJetsHT800to1200") == 0) p=(float)(1.23 * x * (float)(1.1796));
  else if(strcmp (sample.c_str(),"ZJetsHT1200to2500") == 0) p=(float)(1.23 * x * (float)(0.28833));
  else if(strcmp (sample.c_str(),"ZJetsHT2500toInf") == 0) p=(float)(1.23 * x * (float)(0.006945));
  else if(strcmp (sample.c_str(),"TT") == 0) p=(float)(x * (float)(831.76));
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ1000rinv03") == 0) p=(float)(x * (float)(1.));
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ2000rinv03") == 0) p=(float)(x * (float)(1.));
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ3000rinv03") == 0) p=(float)(x * (float)(1.));
  else if(strcmp (sample.c_str(),"SVJalphaD01mZ4000rinv03") == 0) p=(float)(x * (float)(1.));
  cout << "weight is " << p << endl;
  
  TH2F *h_AK8jet_etaphi_lead = new TH2F("h_AK8jet_etaphi_lead", "h_AK8jet_etaphi_lead", 24, -2.4, +2.4, 35, -3.5, +3.5);
  TH2F *h_AK8jet_etaphi_sublead = new TH2F("h_AK8jet_etaphi_sublead", "h_AK8jet_etaphi_sublead", 24, -2.4, +2.4, 35, -3.5, +3.5);
  TH2F *h_AK8jet_etaphi_lead_newID = new TH2F("h_AK8jet_etaphi_lead_newID", "h_AK8jet_etaphi_lead_newID", 24, -2.4, +2.4, 35, -3.5, +3.5);
  TH2F *h_AK8jet_etaphi_sublead_newID = new TH2F("h_AK8jet_etaphi_sublead_newID", "h_AK8jet_etaphi_sublead_newID", 24, -2.4, +2.4, 35, -3.5, +3.5);

  TH1F *h_cutFlow  = new TH1F("h_cutFlow","cutFlow",13,-0.5,12.5);
  //TH1F *h_cutFlow  [maxSysts] ;systZero.initHistogramsSysts(h_cutFlow, "h_cutFlow","cutFlow",13,-0.5,12.5);
  TH1F *h_nPV   [maxSysts] ;systZero.initHistogramsSysts(h_nPV,"h_nPV","nPV",70,0,70);
  TH1F *h_nPV_w   [maxSysts] ;systZero.initHistogramsSysts(h_nPV_w,"h_nPV_w","nPV",70,0,70);

  //Plots with no selectiona applied
  TH1F *h_AK8jetsmult_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetsmult_presel, "h_AK8jetsmult_presel", "AK8 jets multiplicity", 10, -0.5, 9.5);  
  TH1F *h_Muonsmult_presel [maxSysts] ;systZero.initHistogramsSysts(h_Muonsmult_presel,"h_Muonsmult_presel", " muons multiplicity", 10, -0.5, 9.5);
  TH1F *h_Electronsmult_presel [maxSysts] ;systZero.initHistogramsSysts(h_Electronsmult_presel,"h_Electronsmult_presel", " electrons multiplicity", 10, -0.5, 9.5);
  TH1F *h_METPt_presel [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel,"h_METPt_presel", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Ht_presel [maxSysts] ;systZero.initHistogramsSysts(h_Ht_presel,"h_Ht_presel", "{H_{T}}", 100, 0, 5000);
  TH1F *h_AK8jetPt_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetPt_presel,"h_AK8jetPt_presel", "gen-level HV semi-visible jet p_{T}", 60, 0, 3000);
  TH1F *h_AK8jetPhi_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetPhi_presel,"h_AK8jetPhi_presel", "gen-level HV semi-visible jet #Phi", 100, -5, 5);  
  TH1F *h_AK8jetEta_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetEta_presel,"h_AK8jetEta_presel", "gen-level HV semi-visible jet #eta", 100, -6, 6);
  TH1F *h_AK8jetE_lead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetE_lead_presel,"h_AK8jetE_lead_presel", "AK8 jet E", 100, 0, 3000);
  TH1F *h_AK8jetPt_lead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetPt_lead_presel,"h_AK8jetPt_lead_presel", "AK8 jet p_{T}", 60, 0, 3000);
  TH1F *h_AK8jetPhi_lead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetPhi_lead_presel,"h_AK8jetPhi_lead_presel", "AK8 jet #Phi", 100, -5, 5);
  TH1F *h_AK8jetEta_lead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetEta_lead_presel,"h_AK8jetEta_lead_presel", "AK8 jet #eta", 100, -6, 6);
  TH1F *h_AK8jetE_sublead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetE_sublead_presel,"h_AK8jetE_sublead_presel", "AK8 jet E", 100, 0, 3000);
  TH1F *h_AK8jetPt_sublead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetPt_sublead_presel,"h_AK8jetPt_sublead_presel", "AK8 jet p_{T}", 60, 0, 3000);
  TH1F *h_AK8jetPhi_sublead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetPhi_sublead_presel,"h_AK8jetPhi_sublead_presel", "AK8 jet #Phi", 100, -5, 5);
  TH1F *h_AK8jetEta_sublead_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetEta_sublead_presel,"h_AK8jetEta_sublead_presel", "AK8 jet #eta", 100, -6, 6);
  TH1F *h_AK8jetdR_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetdR_presel,"h_AK8jetdR_presel", "#DeltaR(j0,j1)", 100, 0, 5);
  TH1F *h_AK8jetdP_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetdP_presel,"h_AK8jetdP_presel", "#Delta#Phi(j0,j1)", 100, 0, 5);
  TH1F *h_AK8jetdE_presel [maxSysts] ;systZero.initHistogramsSysts(h_AK8jetdE_presel,"h_AK8jetdE_presel", "#Delta#Eta(j0,j1)", 100, 0, 5);
  TH1F *h_Mmc_presel [maxSysts] ;systZero.initHistogramsSysts(h_Mmc_presel,"h_Mmc_presel", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj_presel [maxSysts] ;systZero.initHistogramsSysts(h_Mjj_presel,"h_Mjj_presel", "m_{JJ}", 750, 0, 7500);
  TH1F *h_Mt_presel [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel,"h_Mt_presel", "m_{T}", 750, 0, 7500);

  TH1F *h_Mt_presel_1 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_1,"h_Mt_presel_1", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_1 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_1,"h_METPt_presel_1", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Mt_presel_2 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_2,"h_Mt_presel_2", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_2 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_2,"h_METPt_presel_2", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Mt_presel_3 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_3,"h_Mt_presel_3", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_3 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_3,"h_METPt_presel_3", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Mt_presel_4 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_4,"h_Mt_presel_4", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_4 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_4,"h_METPt_presel_4", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Mt_presel_5 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_5,"h_Mt_presel_5", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_5 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_5,"h_METPt_presel_5", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Mt_presel_6 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_6,"h_Mt_presel_6", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_6 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_6,"h_METPt_presel_6", "MET_{p_{T}}", 100, 0, 5000);
  TH1F *h_Mt_presel_7 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_presel_7,"h_Mt_presel_7", "m_{T}", 750, 0, 7500);
  TH1F *h_METPt_presel_7 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_presel_7,"h_METPt_presel_7", "MET_{p_{T}}", 100, 0, 5000);

  TH1F *h_dPhimin_presel [maxSysts] ;systZero.initHistogramsSysts(h_dPhimin_presel,"h_dPhimin_presel", "min(#Delta#Phi_{1},#Delta#Phi_{2})", 100, 0, 3.5);
  TH1F *h_dPhi1_presel [maxSysts] ;systZero.initHistogramsSysts(h_dPhi1_presel,"h_dPhi1_presel", "min(MET,j1)", 100, 0, 3.5);
  TH1F *h_dPhi2_presel [maxSysts] ;systZero.initHistogramsSysts(h_dPhi2_presel,"h_dPhi2_presel", "min(MET,j2)", 100, 0, 3.5);
  TH1F *h_transverseratio_presel [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio_presel,"h_transverseratio_presel", "MET/M_{T}", 100, 0, 1);
  
  //Plots after full-selection
  TH1F *h_dEta [maxSysts] ;systZero.initHistogramsSysts(h_dEta, "h_dEta", "#Delta#eta(j0,j1)", 100, 0, 10);
  TH1F *h_dPhimin [maxSysts] ;systZero.initHistogramsSysts(h_dPhimin,"h_dPhimin", "min(#Delta#Phi_{1},#Delta#Phi_{2})", 100, 0, 3.5);
  TH1F *h_transverseratio [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio, "h_transverseratio", "MET/M_{T}", 100, 0, 1);
  TH1F *h_Mt [maxSysts] ;systZero.initHistogramsSysts(h_Mt,"h_Mt", "m_{T}", 750, 0, 7500);
  TH1F *h_Mmc [maxSysts] ;systZero.initHistogramsSysts(h_Mmc,"h_Mmc", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj [maxSysts] ;systZero.initHistogramsSysts(h_Mjj,"h_Mjj", "m_{JJ}", 750, 0, 7500);
  TH1F *h_METPt [maxSysts] ;systZero.initHistogramsSysts(h_METPt,"h_METPt", "MET_{p_{T}}", 100, 0, 2000);
  TH1F *h_dPhi [maxSysts] ;systZero.initHistogramsSysts(h_dPhi,"h_dPhi", "#Delta#Phi(j0,j1)", 100, 0, 5);

  TH1F *h_dEta_CR [maxSysts] ;systZero.initHistogramsSysts(h_dEta_CR,"h_dEta_CR", "#Delta#eta(j0,j1)", 100, 0, 10);
  TH1F *h_dPhimin_CR [maxSysts] ;systZero.initHistogramsSysts(h_dPhimin_CR,"h_dPhimin_CR", "min(#Delta#Phi_{1},#Delta#Phi_{2})", 100, 0, 3.5);
  TH1F *h_transverseratio_CR [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio_CR,"h_transverseratio_CR", "MET/M_{T}", 100, 0, 1);
  TH1F *h_Mt_CR [maxSysts] ;systZero.initHistogramsSysts(h_Mt_CR,"h_Mt_CR", "m_{T}", 750, 0, 7500);
  TH1F *h_Mmc_CR [maxSysts] ;systZero.initHistogramsSysts(h_Mmc_CR,"h_Mmc_CR", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj_CR [maxSysts] ;systZero.initHistogramsSysts(h_Mjj_CR,"h_Mjj_CR", "m_{JJ}", 750, 0, 7500);
  TH1F *h_METPt_CR [maxSysts] ;systZero.initHistogramsSysts(h_METPt_CR,"h_METPt_CR", "MET_{p_{T}}", 100, 0, 2000);
  TH1F *h_dPhi_CR [maxSysts] ;systZero.initHistogramsSysts(h_dPhi_CR,"h_dPhi_CR", "#Delta#Phi(j0,j1)", 100, 0, 5);

  TH1F *h_dEta_BDT [maxSysts] ;systZero.initHistogramsSysts(h_dEta_BDT,"h_dEta_BDT", "#Delta#eta(j0,j1)", 100, 0, 10);
  TH1F *h_dPhimin_BDT [maxSysts] ;systZero.initHistogramsSysts(h_dPhimin_BDT,"h_dPhimin_BDT", "min(#Delta#Phi_{1},#Delta#Phi_{2})", 100, 0, 3.5);
  TH1F *h_transverseratio_BDT [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio_BDT,"h_transverseratio_BDT", "MET/M_{T}", 100, 0, 1);
  TH1F *h_Mt_BDT [maxSysts] ;systZero.initHistogramsSysts(h_Mt_BDT,"h_Mt_BDT", "m_{T}", 750, 0, 7500);
  TH1F *h_Mmc_BDT [maxSysts] ;systZero.initHistogramsSysts(h_Mmc_BDT,"h_Mmc_BDT", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj_BDT [maxSysts] ;systZero.initHistogramsSysts(h_Mjj_BDT,"h_Mjj_BDT", "m_{JJ}", 750, 0, 7500);
  TH1F *h_METPt_BDT [maxSysts] ;systZero.initHistogramsSysts(h_METPt_BDT,"h_METPt_BDT", "MET_{p_{T}}", 100, 0, 2000);
  TH1F *h_dPhi_BDT [maxSysts] ;systZero.initHistogramsSysts(h_dPhi_BDT,"h_dPhi_BDT", "#Delta#Phi(j0,j1)", 100, 0, 5);

  TH1F *h_transverseratio_BDT0 [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio_BDT0,"h_transverseratio_BDT0", "MET/M_{T}", 100, 0, 1);
  TH1F *h_Mt_BDT0 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_BDT0,"h_Mt_BDT0", "m_{T}", 750, 0, 7500);
  TH1F *h_Mmc_BDT0 [maxSysts] ;systZero.initHistogramsSysts(h_Mmc_BDT0,"h_Mmc_BDT0", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj_BDT0 [maxSysts] ;systZero.initHistogramsSysts(h_Mjj_BDT0,"h_Mjj_BDT0", "m_{JJ}", 750, 0, 7500);
  TH1F *h_METPt_BDT0 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_BDT0,"h_METPt_BDT0", "MET_{p_{T}}", 100, 0, 2000);

  TH1F *h_transverseratio_BDT1 [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio_BDT1,"h_transverseratio_BDT1", "MET/M_{T}", 100, 0, 1);
  TH1F *h_Mt_BDT1 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_BDT1,"h_Mt_BDT1", "m_{T}", 750, 0, 7500);
  TH1F *h_Mmc_BDT1 [maxSysts] ;systZero.initHistogramsSysts(h_Mmc_BDT1,"h_Mmc_BDT1", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj_BDT1 [maxSysts] ;systZero.initHistogramsSysts(h_Mjj_BDT1,"h_Mjj_BDT1", "m_{JJ}", 750, 0, 7500);
  TH1F *h_METPt_BDT1 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_BDT1,"h_METPt_BDT1", "MET_{p_{T}}", 100, 0, 2000);

  TH1F *h_transverseratio_BDT2 [maxSysts] ;systZero.initHistogramsSysts(h_transverseratio_BDT2,"h_transverseratio_BDT2", "MET/M_{T}", 100, 0, 1);
  TH1F *h_Mt_BDT2 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_BDT2,"h_Mt_BDT2", "m_{T}", 750, 0, 7500);
  TH1F *h_Mmc_BDT2 [maxSysts] ;systZero.initHistogramsSysts(h_Mmc_BDT2,"h_Mmc_BDT2", "m_{MC}", 750, 0, 7500);
  TH1F *h_Mjj_BDT2 [maxSysts] ;systZero.initHistogramsSysts(h_Mjj_BDT2,"h_Mjj_BDT2", "m_{JJ}", 750, 0, 7500);
  TH1F *h_METPt_BDT2 [maxSysts] ;systZero.initHistogramsSysts(h_METPt_BDT2,"h_METPt_BDT2", "MET_{p_{T}}", 100, 0, 2000);

  TH1F *h_Mt_CRBDT0 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_CRBDT0,"h_Mt_CRBDT0", "m_{T}", 750, 0, 7500);
  TH1F *h_Mt_CRBDT1 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_CRBDT1, "h_Mt_CRBDT1", "m_{T}", 750, 0, 7500);
  TH1F *h_Mt_CRBDT2 [maxSysts] ;systZero.initHistogramsSysts(h_Mt_CRBDT2, "h_Mt_CRBDT2", "m_{T}", 750, 0, 7500);

  TH1F *h_bdt_mult_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_mult_0_presel, "h_bdt_mult_0_presel", "mult jet 0", 500, 0, 500);
  TH1F *h_bdt_mult_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_mult_1_presel, "h_bdt_mult_1_presel", "mult jet 1", 500, 0, 500);
  TH1F *h_bdt_axisminor_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_axisminor_0_presel, "h_bdt_axisminor_0_presel", "axisminor jet 0", 100, 0, 0.2);
  TH1F *h_bdt_axisminor_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_axisminor_1_presel, "h_bdt_axisminor_1_presel", "axisminor jet 1", 100, 0, 0.2);
  TH1F *h_bdt_girth_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_girth_0_presel, "h_bdt_girth_0_presel", "girth jet 0", 100, 0, 0.7);
  TH1F *h_bdt_girth_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_girth_1_presel, "h_bdt_girth_1_presel", "girth jet 1", 100, 0, 0.7);
  TH1F *h_bdt_tau21_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_tau21_0_presel, "h_bdt_tau21_0_presel", "tau21 jet 0", 100, 0, 1);
  TH1F *h_bdt_tau21_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_tau21_1_presel, "h_bdt_tau21_1_presel", "tau21 jet 1", 100, 0, 1.4);
  TH1F *h_bdt_tau32_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_tau32_0_presel, "h_bdt_tau32_0_presel", "tau32 jet 0", 100, 0, 1.4);
  TH1F *h_bdt_tau32_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_tau32_1_presel, "h_bdt_tau32_1_presel", "tau32 jet 1", 100, 0, 1);
  TH1F *h_bdt_deltaphi_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_deltaphi_0_presel, "h_bdt_deltaphi_0_presel", "deltaphi jet 0", 100, 0, 3.5);
  TH1F *h_bdt_deltaphi_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_deltaphi_1_presel, "h_bdt_deltaphi_1_presel", "deltaphi jet 1", 100, 0, 3.5);
  TH1F *h_bdt_msd_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_msd_0_presel, "h_bdt_msd_0_presel", "msd jet 0", 120, 0, 600);
  TH1F *h_bdt_msd_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_msd_1_presel, "h_bdt_msd_1_presel", "msd jet 1", 120, 0, 600);
  TH1F *h_bdt_mva_0_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_mva_0_presel, "h_bdt_mva_0_presel", "mva jet 0", 100, -1, 1);
  TH1F *h_bdt_mva_1_presel [maxSysts] ;systZero.initHistogramsSysts(h_bdt_mva_1_presel, "h_bdt_mva_1_presel", "mva jet 0", 100, -1, 1);

  float n_twojets(0.), n_prejetspt(0.), n_pretrigplateau(0.),  n_transratio(0.), n_MT(0.), n_METfilters(0.), n_dPhi(0.), n_transverse(0.);
  float  n_muonveto(0.), n_electronveto(0.), n_BDT(0.) ;
  std::cout<< "===> Number of Processed Events: "<<nEventsPrePres << " ===> Number of Skimmed Events: "<<nEvents <<std::endl;


  systZero.createFilesSysts(allMyFiles,outdir+"/res/"+sample +syststr);

  bool onlyNominal=false;
  systZero.setOnlyNominal(onlyNominal);
  systSVJ.setOnlyNominal(onlyNominal);

  //if(isData=="MC"){
  //  LumiWeights_ = edm::LumiReWeighting("data_Mo17/puMC.root", "data_Mo17/MyDataPileupHistogram.root","MC_pu","pileup");
  //  LumiWeightsUp_ = edm::LumiReWeighting("data_Mo17/puMC.root", "data_Mo17/MyDataPileupHistogramUP.root","MC_pu","pileup");
  //  LumiWeightsDown_ = edm::LumiReWeighting("data_Mo17/puMC.root", "data_Mo17/MyDataPileupHistogramDOWN.root","MC_pu","pileup");
  // }

  //nEvents = 1000;
  for(Int_t i=0; i<nEvents; i++ )
    {

      chain.GetEntry(i);
      w = 1;
      w_zero = 1;

      if(isData=="MC"){ 

	std::vector<double> PDFweights = *PDFweightsPtr;
	w_zero = PDFweights[0];

	if(addQ2){
	  w_q2up = PDFweights[4];
	  w_q2down = PDFweights[8];
	}

	if(addPDF){
	  for (int p = 1; p <= nPDF;++p){
	    stringstream pdfss;
	    //pdfss<<(p+8);
	    pdfss<<(p);
	    //string pstr =(pdfss.str());
	    //w_pdfs[p-1] = PDFweights[p+8];
	    w_pdfs[p-1] = PDFweights[p];
	    //chain.SetBranchAddress(("Event_LHEWeight"+pstr).c_str(), &w_pdfs[p-1]);
	  }
	}
	
	w_pu = 1.;//puWeight; //LumiWeights_.weight(NumInteractions);
	w = w_pu;

	std::vector<TLorentzVector> GenElectrons = *GenElectronsPtr;
	std::vector<TLorentzVector> GenMuons = *GenMuonsPtr;
	std::vector<TLorentzVector> GenTaus = *GenTausPtr;
	
	bool tt_stitching=false;
	
	if(madHT < 600 && GenElectrons.size()==0 && GenMuons.size()==0 && GenTaus.size()==0 && tt_stitching_TTJets==true) tt_stitching=true;
	else if(madHT < 600 && GenMET < 150 && tt_stitching_TTJets_DiLept==true) tt_stitching=true;
	else if(madHT < 600 && GenMET >= 150 && tt_stitching_TTJets_DiLept_genMET150==true) tt_stitching=true;
	else if(madHT < 600 && GenMET < 150 && tt_stitching_TTJets_SingleLeptFromT==true) tt_stitching=true;
	else if(madHT < 600 && GenMET >= 150 && tt_stitching_TTJets_SingleLeptFromT_genMET150==true) tt_stitching=true;
	else if(madHT < 600 && GenMET < 150 && tt_stitching_TTJets_SingleLeptFromTbar==true) tt_stitching=true;
	else if(madHT < 600 && GenMET >= 150 && tt_stitching_TTJets_SingleLeptFromTbar_genMET150==true) tt_stitching=true;
	else if(madHT >= 600 && tt_stitching_TTJets_HT600to800==true) tt_stitching=true;
	else if(madHT >= 600 && tt_stitching_TTJets_HT800to1200==true) tt_stitching=true;
	else if(madHT >= 600 && tt_stitching_TTJets_HT1200to2500==true) tt_stitching=true;
	else if(madHT >= 600 && tt_stitching_TTJets_HT2500toInf==true) tt_stitching=true;
	
	else if(tt_stitching_noTT==true) tt_stitching=true;

	if(tt_stitching==false) continue;
      }
      if(isData=="DATA"){
	
	systZero.setWeight(0,1.);
	systZero.setWeight("puDown",1.);
	systZero.setWeight("puUp",1.);

	systSVJ.copySysts(systZero);
	systSVJ.setWeight(0,1.);
	systSVJ.setWeight("puDown",1.);
	systSVJ.setWeight("puUp",1.);

	systWeightsSVJ[NOSYST]=1.;
        systWeightsSVJ[PUUP]=1.;
        systWeightsSVJ[PUDOWN]=1.;
      }

      if(isData=="MC"){
	//double puUpFact=(LumiWeightsUp_.weight(NumInteractions))/(LumiWeights_.weight(NumInteractions));
	//double puDownFact=(LumiWeightsDown_.weight(NumInteractions))/(LumiWeights_.weight(NumInteractions));

	double puUpFact = puSysUp;
	double puDownFact = puSysDown;
	
	if(NumInteractions>75){
	  //cout << " --> NumInteractions very high!!" << endl;
	  puUpFact =0;
	  puDownFact=0;
	}
	
	systZero.setWeight(0,1.);
	systZero.setWeight("puUp",1.);
	systZero.setWeight("puDown",1.);

	if(addPDF)systZero.setPDFWeights(w_pdfs, PDFsplittedWeight, nPDF,w_zero, true);
	if(addQ2)systZero.setQ2Weights(w_q2up,w_q2down,w_zero,true);

	systSVJ.copySysts(systZero);
	systSVJ.setWeight(0, 1.);
	systSVJ.setWeight("puUp", puUpFact);
	systSVJ.setWeight("puDown", puDownFact);

	if(addPDF)systSVJ.setPDFWeights(w_pdfs, PDFsplittedWeight, nPDF,w_zero,true);
	if(addQ2)systSVJ.setQ2Weights(w_q2up,w_q2down,w_zero,true);

	systWeightsSVJ[NOSYST]=1.;
	systWeightsSVJ[PUUP]= puUpFact;
	systWeightsSVJ[PUDOWN]= puDownFact;
      }
      //JET                
      std::vector<int> mult = *multPtr;
      std::vector<double> axisminor = *axisminorPtr;
      std::vector<double> girth = *girthPtr;
      std::vector<double> tau1 = *tau1Ptr;
      std::vector<double> tau2 = *tau2Ptr;
      std::vector<double> tau3 = *tau3Ptr;
      std::vector<double> msd = *msdPtr;
      std::vector<double> muMiniIso = *muMiniIsoPtr;
      std::vector<bool> jetsID = *jetsIDPtr;

      std::vector<double> NHF = *JetsAK8_NHFPtr;
      std::vector<double> CHF = *JetsAK8_CHFPtr;

      std::vector<TLorentzVector> AK8Jets = *jetsAK8CHSPtr;

      // std::vector<TLorentzVector> Muons;
      TLorentzVector Muon;
      std::vector<TLorentzVector> Electrons;
      TLorentzVector Electron;
      std::vector<TLorentzVector> muons = *MuonsPtr;
      std::vector<TLorentzVector> electrons = *ElectronsPtr;

      std::vector<int>& triggerPass = *triggerPassPtr;

      bool preselection_metfilters = 0;
      bool preselection_metfilters_1 = 0;
      bool preselection_metfilters_2 = 0;
      bool preselection_metfilters_3 = 0;
      bool preselection_metfilters_4 = 0;
      bool preselection_metfilters_5 = 0;
      bool preselection_metfilters_6 = 0;
      bool preselection_metfilters_7 = 0;
      preselection_metfilters = BadChargedCandidateFilter>0 && BadPFMuonFilter>0 && EcalDeadCellTriggerPrimitiveFilter>0 && HBHEIsoNoiseFilter>0 && HBHENoiseFilter>0 && globalTightHalo2016Filter>0 && NVtx > 0;
      preselection_metfilters_1 = BadChargedCandidateFilter>0;
      preselection_metfilters_2 = BadPFMuonFilter>0; 
      preselection_metfilters_3 = EcalDeadCellTriggerPrimitiveFilter>0;
      preselection_metfilters_4 = HBHEIsoNoiseFilter>0;
      preselection_metfilters_5 = HBHENoiseFilter>0; 
      preselection_metfilters_6 = globalTightHalo2016Filter>0;
      preselection_metfilters_7 = NVtx > 0;
      //preselection_metfilters = 1;

      if(preselection_metfilters){
	systSVJ.fillHistogramsSysts( h_AK8jetsmult_presel, AK8Jets.size(),w,systWeightsSVJ);
      }

      //HV quarks definition
      TLorentzVector vHVsum;
      if(isData=="MC"){
	std::vector<TLorentzVector>& genParts = *genPartsPtr;
	
	struct genParticle{
	  TLorentzVector vect;
	  int pdgId;
	  int status;
	};
	
	genParticle HVgenParticlesInv;
	std::vector<genParticle> HVgenParticlesInvVect;
	
	for(int j=0; j<sizeMax_gen; j++){
	  if(std::abs(genPart_pdgId[j])==4900211 || std::abs(genPart_pdgId[j])==4900213 ){
	    HVgenParticlesInv.vect = genParts[j];
	    HVgenParticlesInv.pdgId=genPart_pdgId[j];
	    HVgenParticlesInv.status=genPart_Status[j];
	    HVgenParticlesInvVect.push_back(HVgenParticlesInv);  
	    vHVsum += (HVgenParticlesInv.vect);
	  }
	}
      }
      //start of the analysis and of the variable definition                                                                                   
      if(AK8Jets.size()>1){
	//if(1>0){
	//Compute angular distances between the two leading jets
	double AK8Jets_dr=-1;
	AK8Jets_dr = (AK8Jets.at(0)).DeltaR(AK8Jets.at(1));
	double dEta=0, dPhi=0;
	dEta= (std::fabs((AK8Jets.at(0)).Eta() - (AK8Jets.at(1)).Eta()));
	dPhi= std::fabs(reco::deltaPhi(AK8Jets.at(0).Phi(),AK8Jets.at(1).Phi()));

	double dPhi_j0_met=0., dPhi_j1_met=0.;//, dPhi_min=0.;  
	dPhi_j0_met = std::fabs(reco::deltaPhi(AK8Jets.at(0).Phi(),metFull_Phi));
	dPhi_j1_met = std::fabs(reco::deltaPhi(AK8Jets.at(1).Phi(),metFull_Phi));
	//dPhi_min = std::min(dPhi_j0_met,dPhi_j1_met);

	//Compute masses
	double Mjj=0., Mmc=0., MT2=0.;
	double  Mjj2=0., ptjj = 0., ptjj2 = 0., ptMet = 0.;
	//Mjj = mass of the two large reclustered jets                                                                                    
	TLorentzVector vjj = AK8Jets.at(0) + AK8Jets.at(1);
	if(AK8Jets.size()>=3){
	  bool merge = false;
	  merge = (AK8Jets.at(1).DeltaR(AK8Jets.at(2))<1.4 ) || (AK8Jets.at(0).DeltaR(AK8Jets.at(2))<1.4 );
	  merge = false;
	  if(merge) {
	    std::cout<<"==========> We found a third jet"<<std::endl;
	    vjj +=  AK8Jets.at(2);
	  }
	}
	
	double metFull_Px=0., metFull_Py=0.; 
	metFull_Py = metFull_Pt*sin(metFull_Phi);
	metFull_Px = metFull_Pt*cos(metFull_Phi);

	Mjj = vjj.M();
	Mjj2 = Mjj*Mjj;
	ptjj = vjj.Pt();
	ptjj2 = ptjj * ptjj;
	ptMet = vjj.Px()*metFull_Px +  vjj.Py()*metFull_Py;
	
	//Mmc = the reconstructed Z' mass using all the dark matter particles in the MC                                                           
	if(isData=="DATA"){vHVsum = {0, 0, 0, 0};}
	TLorentzVector vmc = vHVsum + vjj;
	Mmc = vmc.M();

	MT2 = sqrt(Mjj2 + 2*(sqrt(Mjj2 + ptjj2)*metFull_Pt   -  ptMet) );   
	//std::cout<<"Old MT: "<<MT<<std::endl;
	//std::cout<<"New MT: "<<MT2<<std::endl;
	//cout << MT2 << " " << MT << endl;
	MT2 = MT;

	//Define preselection
	bool preselection_jetseta = 0, preselection_jetsID = 0, preselection_trigplateau=0, preselection_deltaeta=0, preselection_transratio = 0, preselection_leptonveto = 0 , preselection_jetspt = 0, preselection = 0;
	bool preselection_dijet(0), preselection_muonveto(0), preselection_muonLooseveto(1), preselection_electronveto(0);
	bool preselection_trigger(0);

	int sizeMax_muons=muons.size();

	for(int j=0; j< sizeMax_muons; j++){
	  if (muMiniIso[j]<0.4){preselection_muonLooseveto = 0;}
	}

	preselection_jetseta = (std::abs((AK8Jets.at(0)).Eta()) < 2.5 && std::abs((AK8Jets.at(1)).Eta()) < 2.5);
	preselection_jetsID = jetsID.at(0) == 1 && jetsID.at(1)==1;
	preselection_deltaeta = std::abs(AK8Jets.at(0).Eta() - AK8Jets.at(1).Eta()) < 1.5;
	preselection_trigplateau = preselection_deltaeta;
	preselection_transratio = metFull_Pt/MT2 > 0.15;
	preselection_leptonveto = nElectrons + nMuons < 1;
	preselection_muonveto = nMuons<1 && preselection_muonLooseveto;
	preselection_electronveto =nElectrons<1;
	preselection_jetspt = (AK8Jets.at(0)).Pt() > 200 && (AK8Jets.at(1)).Pt()>200;
	preselection_dijet =  preselection_jetseta &&  preselection_jetsID && preselection_jetspt;
	
	preselection_trigger = 1;

	if(isData2016 == true)	preselection_trigger = triggerPass[10] == 1 || triggerPass[13] == 1 || triggerPass[103] == 1 || triggerPass[105] == 1 || triggerPass[106] == 1;
	else if (isData2017 == true) preselection_trigger = triggerPass[11] == 1 || triggerPass[13] == 1 || triggerPass[67] == 1 || triggerPass[107] == 1;
	
	bool selection_transverseratio_window=0;
	selection_transverseratio_window = (metFull_Pt/MT2)> 0.15 && (metFull_Pt/MT2) < 0.25 ;
	
	bool preselection_CR = 0;
	preselection_CR = preselection_muonveto && preselection_jetseta && preselection_jetsID && preselection_deltaeta && preselection_leptonveto && preselection_metfilters && MT2 > 1500 && preselection_trigger;

	preselection = preselection_muonveto && preselection_jetseta && preselection_jetsID && preselection_deltaeta && preselection_transratio && preselection_leptonveto && preselection_trigger && MT2 > 1500 && preselection_metfilters;

	bool selection_dPhi = 0, selection_transverseratio = 0, selection_mt = 0,  selection = 0;
	selection_dPhi = DeltaPhiMin < 0.75;
	selection_mt = MT2 > 1500;
	selection_transverseratio = (metFull_Pt/MT2) > 0.25; 

	selection = (preselection  && selection_dPhi && selection_transverseratio && selection_mt);
	bool selection_CR = 0;
	selection_CR = preselection_CR && selection_dPhi && selection_transverseratio_window && selection_mt;

	double mva1_(0.), mva2_(0.);

	bdt_mult= mult.at(0); bdt_axisminor = axisminor.at(0); bdt_girth = girth.at(0); bdt_tau21 = tau2.at(0)/tau1.at(0);  bdt_tau32 = tau3.at(0)/tau1.at(0);
	bdt_msd = msd.at(0); bdt_deltaphi = deltaphi1;  bdt_pt = AK8Jets.at(0).Pt();  bdt_eta =  AK8Jets.at(0).Eta(); bdt_mt =  MT2;

	mva1_ = reader.EvaluateMVA("BDTG");
	bdt_mult= mult.at(1); bdt_axisminor = axisminor.at(1); bdt_girth = girth.at(1); bdt_tau21 = tau2.at(1)/tau1.at(1);  bdt_tau32 = tau3.at(1)/tau1.at(1);
	bdt_msd = msd.at(1); bdt_deltaphi = deltaphi2;  bdt_pt = AK8Jets.at(1).Pt();  bdt_eta = AK8Jets.at(1).Eta(); bdt_mt = MT2;

	mva2_ = reader.EvaluateMVA("BDTG");

	if(preselection){
	  if(preselection_metfilters_1){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_1,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_1,metFull_Pt,w,systWeightsSVJ);
	  }
	  if(preselection_metfilters_2){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_2,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_2,metFull_Pt,w,systWeightsSVJ);
	  }
	  if(preselection_metfilters_3){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_3,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_3,metFull_Pt,w,systWeightsSVJ);
	  }
	  if(preselection_metfilters_4){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_4,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_4,metFull_Pt,w,systWeightsSVJ);
	  }
	  if(preselection_metfilters_5){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_5,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_5,metFull_Pt,w,systWeightsSVJ);
	  }
	  if(preselection_metfilters_6){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_6,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_6,metFull_Pt,w,systWeightsSVJ);
	  }
	  if(preselection_metfilters_7){
	    systSVJ.fillHistogramsSysts(h_Mt_presel_7,MT2,w,systWeightsSVJ);
	    systSVJ.fillHistogramsSysts(h_METPt_presel_7,metFull_Pt,w,systWeightsSVJ);
	  }
	}
 
	if(preselection){
	  if(metFull_Pt>300){
	    h_AK8jet_etaphi_lead->Fill(AK8Jets.at(0).Eta(), AK8Jets.at(0).Phi());
	    h_AK8jet_etaphi_sublead->Fill(AK8Jets.at(1).Eta(), AK8Jets.at(1).Phi());
	    if(NHF.at(0) < 0.8 && CHF.at(0) > 0.1) h_AK8jet_etaphi_lead_newID->Fill(AK8Jets.at(0).Eta(), AK8Jets.at(0).Phi());
	    if(NHF.at(1) < 0.8 && CHF.at(1) > 0.1) h_AK8jet_etaphi_sublead_newID->Fill(AK8Jets.at(1).Eta(), AK8Jets.at(1).Phi());
	  }
	  
	  systSVJ.fillHistogramsSysts(h_nPV,NVtx,1.,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_nPV_w,NVtx,w,systWeightsSVJ);
	  
	  //Leptons
	  systSVJ.fillHistogramsSysts(h_Muonsmult_presel,nMuons,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Electronsmult_presel,nElectrons,w,systWeightsSVJ);

	  //MET
	  systSVJ.fillHistogramsSysts(h_METPt_presel, metFull_Pt,w,systWeightsSVJ);
	  
	  //Hadronic objects
	  systSVJ.fillHistogramsSysts(h_Ht_presel,Ht,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPt_presel,AK8Jets.at(0).Pt(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPt_presel,AK8Jets.at(1).Pt(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetEta_presel,AK8Jets.at(0).Eta(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetEta_presel,AK8Jets.at(1).Eta(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPhi_presel,AK8Jets.at(0).Phi(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPhi_presel,AK8Jets.at(1).Phi(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetE_lead_presel,AK8Jets.at(0).E(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetE_sublead_presel,AK8Jets.at(1).E(),w,systWeightsSVJ);

	  systSVJ.fillHistogramsSysts(h_AK8jetPt_lead_presel,AK8Jets.at(0).Pt(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPt_sublead_presel,AK8Jets.at(1).Pt(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetEta_lead_presel,AK8Jets.at(0).Eta(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetEta_sublead_presel,AK8Jets.at(1).Eta(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPhi_lead_presel,AK8Jets.at(0).Phi(),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetPhi_sublead_presel,AK8Jets.at(1).Phi(),w,systWeightsSVJ);

	  systSVJ.fillHistogramsSysts(h_AK8jetdR_presel,AK8Jets_dr,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetdP_presel,dPhi,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_AK8jetdE_presel,dEta,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhi1_presel,dPhi_j0_met,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhi2_presel,dPhi_j1_met,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc_presel,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj_presel,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt_presel,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhimin_presel,DeltaPhiMin,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_transverseratio_presel,metFull_Pt/MT2,w,systWeightsSVJ);

	  systSVJ.fillHistogramsSysts(h_bdt_mult_0_presel,mult.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_mult_1_presel,mult.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_axisminor_0_presel,axisminor.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_axisminor_1_presel,axisminor.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_girth_0_presel,girth.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_girth_1_presel,girth.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_tau21_0_presel,tau2.at(0)/tau1.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_tau21_1_presel,tau2.at(1)/tau1.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_tau32_0_presel,tau3.at(0)/tau2.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_tau32_1_presel,tau3.at(1)/tau2.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_tau32_0_presel,tau3.at(0)/tau2.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_tau32_1_presel,tau3.at(1)/tau2.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_msd_0_presel,msd.at(0),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_msd_1_presel,msd.at(1),w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_deltaphi_0_presel,deltaphi1,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_deltaphi_1_presel,deltaphi2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_mva_0_presel,mva1_,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_bdt_mva_1_presel,mva2_,w,systWeightsSVJ);

	}

        double bdtCut = -0.14;
	bool selection_BDT2(0), selection_BDT1(0), selection_BDT0(0);
	selection_BDT2= selection && (mva1_>bdtCut) && (mva2_>bdtCut);
 	selection_BDT1= selection && (((mva1_>bdtCut) && (mva2_<bdtCut)) || ((mva1_<bdtCut) && (mva2_>bdtCut)));
 	selection_BDT0= selection && ((mva1_<bdtCut) && (mva2_<bdtCut));
	bool selection_CRBDT2(0), selection_CRBDT1(0), selection_CRBDT0(0);
	selection_CRBDT2= selection_CR && (mva1_>bdtCut) && (mva2_>bdtCut);
 	selection_CRBDT1= selection_CR && (((mva1_>bdtCut) && (mva2_<bdtCut)) || ((mva1_<bdtCut) && (mva2_>bdtCut)));
 	selection_CRBDT0= selection_CR && ((mva1_<bdtCut) && (mva2_<bdtCut));

	if(selection){	  

	  systSVJ.fillHistogramsSysts(h_dEta,dEta,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhi,dPhi,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhimin,DeltaPhiMin,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_transverseratio,metFull_Pt/MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_METPt,metFull_Pt,w,systWeightsSVJ);

	}

	if(selection && (selection_BDT1 || selection_BDT2)){	  

	  systSVJ.fillHistogramsSysts(h_dEta_BDT,dEta,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhi_BDT,dPhi,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhimin_BDT,DeltaPhiMin,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_transverseratio_BDT,metFull_Pt/MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt_BDT,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj_BDT,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc_BDT,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_METPt_BDT,metFull_Pt,w,systWeightsSVJ);

	}

	if(selection_BDT0){

	  systSVJ.fillHistogramsSysts(h_transverseratio_BDT0,metFull_Pt/MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt_BDT0,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj_BDT0,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc_BDT0,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_METPt_BDT0,metFull_Pt,w,systWeightsSVJ);

	}

	if(selection_BDT1){

	  systSVJ.fillHistogramsSysts(h_transverseratio_BDT1,metFull_Pt/MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt_BDT1,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj_BDT1,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc_BDT1,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_METPt_BDT1,metFull_Pt,w,systWeightsSVJ);

	}

	if(selection_BDT2){

	  systSVJ.fillHistogramsSysts(h_transverseratio_BDT2,metFull_Pt/MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt_BDT2,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj_BDT2,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc_BDT2,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_METPt_BDT2,metFull_Pt,w,systWeightsSVJ);

	}

	if(selection_CRBDT0) systSVJ.fillHistogramsSysts(h_Mt_CRBDT0,MT2,w,systWeightsSVJ);
	if(selection_CRBDT1) systSVJ.fillHistogramsSysts(h_Mt_CRBDT1,MT2,w,systWeightsSVJ);
	if(selection_CRBDT2) systSVJ.fillHistogramsSysts(h_Mt_CRBDT2,MT2,w,systWeightsSVJ);

	if(selection_CR){	  

	  systSVJ.fillHistogramsSysts(h_dEta_CR,dEta,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhi_CR,dPhi,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_dPhimin_CR,DeltaPhiMin,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_transverseratio_CR,metFull_Pt/MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mt_CR,MT2,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mjj_CR,Mjj,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_Mmc_CR,Mmc,w,systWeightsSVJ);
	  systSVJ.fillHistogramsSysts(h_METPt_CR,metFull_Pt,w,systWeightsSVJ);

	}

	// Fill cutflow entries
	n_twojets+=1;
	if(preselection_dijet && preselection_trigger){
	  n_prejetspt+=1;
	  if(preselection_transratio ){
	    n_transratio +=1;
	      if(preselection_muonveto){
		n_muonveto+=1;
		if(preselection_electronveto){
		  n_electronveto+=1;
		  if(preselection_trigplateau){
		    n_pretrigplateau+=1;
		    if(selection_mt){
		      n_MT+=1;
		      if(selection_transverseratio){
			n_transverse+=1;
			if(selection_dPhi){
			  n_dPhi+=1;
			  if(preselection_metfilters){
			    n_METfilters +=1;
			    if(selection_BDT2){
			      n_BDT+=1;
			    }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }//end of nAK8CHSJets>1
	
    }//end of loop over events

  
  h_cutFlow->SetBinContent(1,nEventsPrePres);
  h_cutFlow->GetXaxis()->SetBinLabel(1,"no selection");
  h_cutFlow->SetBinContent(2,nEvents);
  h_cutFlow->GetXaxis()->SetBinLabel(2,"ntuples skimming");
  h_cutFlow->SetBinContent(3, n_twojets);
  h_cutFlow->GetXaxis()->SetBinLabel(3,"n(AK8 jets) > 1");
  h_cutFlow->SetBinContent(4, n_prejetspt);
  h_cutFlow->GetXaxis()->SetBinLabel(4,"p_{T, j1/j2} > 200 GeV & jetID");
  h_cutFlow->SetBinContent(5, n_transratio);
  h_cutFlow->GetXaxis()->SetBinLabel(5,"MET/M_T > 0.15"); 
  h_cutFlow->SetBinContent(6, n_pretrigplateau);
  h_cutFlow->GetXaxis()->SetBinLabel(6,"|#Delta#eta(j1,j2)| < 1.5 or p_{T, j1} > 600");
  h_cutFlow->SetBinContent(7, n_muonveto);
  h_cutFlow->GetXaxis()->SetBinLabel(7,"Muon veto ");
  h_cutFlow->SetBinContent(8, n_electronveto);
  h_cutFlow->GetXaxis()->SetBinLabel(8,"Electron veto ");
  h_cutFlow->SetBinContent(9, n_MT);
  h_cutFlow->GetXaxis()->SetBinLabel(9,"M_T > 1500");
  h_cutFlow->SetBinContent(10, n_transverse);
  h_cutFlow->GetXaxis()->SetBinLabel(10, "MET/M_{T} > 0.25"); 
  h_cutFlow->SetBinContent(11, n_dPhi);
  h_cutFlow->GetXaxis()->SetBinLabel(11,"#Delta#Phi < 0.75");
  h_cutFlow->SetBinContent(12, n_METfilters);
  h_cutFlow->GetXaxis()->SetBinLabel(12,"MET filters");
  h_cutFlow->SetBinContent(13, n_BDT);
  h_cutFlow->GetXaxis()->SetBinLabel(13,"2 SVJ (BDTG>-0.14)");
 
  std::cout<<"===================="<<std::endl;
  std:: cout<<"Cutflow"<<"Raw events  Abs Eff (%)     Rel Eff (%)"<<std::endl;
  std:: cout<<"Dijet:          "<<n_prejetspt <<   "    "<< float(n_prejetspt/nEventsPrePres) * 100 << "    "<< float(n_prejetspt/nEventsPrePres) * 100.<< std::endl;
  std:: cout<<"MET/MT>0.15:    "<<n_transratio <<   "    "<< float(n_transratio/nEventsPrePres) * 100<< "    "<< float(n_transratio/n_prejetspt) * 100.<< std::endl;
  std:: cout<<"Muon Veto:      "<< n_muonveto<<   "    "<< float(n_muonveto/nEventsPrePres) * 100<< "    "<< float(n_muonveto/n_transratio) * 100<<std::endl;
  std:: cout<<"Electron Veto:  "<< n_electronveto<<  "    "<< float(n_electronveto/nEventsPrePres) * 100<< "    "<< float(n_electronveto/n_muonveto) * 100<<std::endl;
  std:: cout<<"DeltaEta < 1.5: "<<n_pretrigplateau <<   "    "<< float(n_pretrigplateau/nEventsPrePres) * 100<< "    "<<  float(n_pretrigplateau/n_electronveto) * 100<< std::endl;
  std:: cout<<"MT>1500:        "<< n_MT<<  "    "<< float(n_MT/nEventsPrePres) * 100<< "    "<< float(n_MT/n_electronveto) * 100<<std::endl;
  std:: cout<<"MET/MT>0.25:    "<< n_transverse<<  "    "<< float(n_transverse/nEventsPrePres) * 100<< "    "<< float(n_transverse/n_MT) * 100<<std::endl;
  std:: cout<<"DeltaPhi<0.75:  "<<n_dPhi <<  "    "<< float(n_dPhi/nEventsPrePres) * 100<< "    "<< float(n_dPhi/n_transverse) * 100<<std::endl;
  std:: cout<<"MetFilters:     "<<n_METfilters <<  "    "<< float(n_METfilters/nEventsPrePres) * 100<< "    "<< float(n_METfilters/n_dPhi) * 100<<std::endl;
  std:: cout<<"2 SVJ, BDGT>-0.14: "<< n_BDT <<  "    "<< float(n_BDT/nEventsPrePres) * 100<< "    "<< float(n_BDT/n_METfilters) * 100<<std::endl;


  fout.cd();
  
  systSVJ.writeHistogramsSysts(h_nPV_w, allMyFiles);
  systSVJ.writeHistogramsSysts(h_nPV, allMyFiles);

  //h_cutFlow->Write();
  //systSVJ.writeHistogramsSysts(h_cutFlow, allMyFiles);
  systZero.writeSingleHistogramSysts(h_cutFlow, allMyFiles);

  systSVJ.writeHistogramsSysts(h_AK8jetsmult_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Muonsmult_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Electronsmult_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Ht_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetPt_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetPhi_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetEta_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetE_lead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetPt_lead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetPhi_lead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetEta_lead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetE_sublead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetPt_sublead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetPhi_sublead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetEta_sublead_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetdR_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetdE_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_AK8jetdP_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_Mt_presel_1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel_2, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_2, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel_3, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_3, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel_4, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_4, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel_5, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_5, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel_6, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_6, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_presel_7, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_presel_7, allMyFiles);
  
  
  systSVJ.writeHistogramsSysts(h_Mjj_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhimin_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhi1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhi2_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_transverseratio_presel, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_bdt_mult_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_mult_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_axisminor_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_axisminor_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_girth_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_girth_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_tau21_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_tau21_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_tau32_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_tau32_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_deltaphi_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_deltaphi_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_msd_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_msd_1_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_mva_0_presel, allMyFiles);
  systSVJ.writeHistogramsSysts(h_bdt_mva_1_presel, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_dEta, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhimin, allMyFiles);
  systSVJ.writeHistogramsSysts(h_transverseratio, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mjj, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhi, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_dEta_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhimin_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_transverseratio_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mjj_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_CR, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhi_CR, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_dEta_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhi_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_dPhimin_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_transverseratio_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mjj_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc_BDT, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_BDT, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_transverseratio_BDT0, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_BDT0, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mjj_BDT0, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc_BDT0, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_BDT0, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_transverseratio_BDT1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_BDT1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mjj_BDT1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc_BDT1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_BDT1, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_transverseratio_BDT2, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_BDT2, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mjj_BDT2, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mmc_BDT2, allMyFiles);
  systSVJ.writeHistogramsSysts(h_METPt_BDT2, allMyFiles);
  
  systSVJ.writeHistogramsSysts(h_Mt_CRBDT0, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_CRBDT1, allMyFiles);
  systSVJ.writeHistogramsSysts(h_Mt_CRBDT2, allMyFiles);
  
  fileout.close();
  
  TFile *myfile = new TFile(outfile_hotspot, "RECREATE");
  h_AK8jet_etaphi_lead->Write();
  h_AK8jet_etaphi_sublead->Write();
  h_AK8jet_etaphi_lead_newID->Write();
  h_AK8jet_etaphi_sublead_newID->Write();
  myfile->Close();

}//end of main


TLorentzVector Rematch(TLorentzVector gp, std::vector<TLorentzVector> jet, float dR){
  float eta=0, pt=0, phi=0, e=0;
  float hardestPt=-1.;

  int sizeMax=jet.size();

  //cout << sizeMax << endl;
  for(int i=0; i<sizeMax;i++){
    //cout << "dR: " << gp.Pt() << " " <<  jet[i].Pt() << " " << jet[i].DeltaR(gp) << endl;
    
    //cout << jet[i].DeltaR(gp) << " " << jet[i].Pt() <<  " " << jet[i].Eta() << endl;
    if  (jet[i].DeltaR(gp) >dR) continue;
    if ( hardestPt <0 or jet[i].Pt() > hardestPt)
      {
	hardestPt = jet[i].Pt();
	//deltaR = jet.DeltaR(gp);
	pt = jet[i].Pt();
	phi = jet[i].Phi();
	eta = jet[i].Eta();
	e = jet[i].E();
      }

  }
  
  TLorentzVector Matched_jet;
  Matched_jet.SetPtEtaPhiE(pt, eta, phi, e);
  //cout << Matched_jet.Pt() << endl;
  return Matched_jet;
}


float check_match(TLorentzVector gp1, TLorentzVector gp2, std::vector<TLorentzVector> jet, float dR){

  int sizeMax = jet.size();

  // look for leading and subleading jets
  int i_lead_Pt = 0;
  int i_subl_Pt = 0;
  float lead_Pt = -1.;
  float subl_Pt = -1.;
  for(int i=0; i<sizeMax; i++){
    float tmp_Pt=jet[i].Pt();
    if( tmp_Pt > subl_Pt ){
      if( tmp_Pt < lead_Pt ){
        subl_Pt = tmp_Pt;
        i_subl_Pt = i;
      } else {
        subl_Pt = lead_Pt;
        i_subl_Pt = i_lead_Pt;
        lead_Pt = tmp_Pt;
        i_lead_Pt = i;
      }
    }
  }

  // match partons to jets
  int i_part_1 = -1;
  int i_part_2 = -1;
  float hardest_Pt_part_1 = -1;
  float hardest_Pt_part_2 = -1;
  for(int i=0; i<sizeMax; i++){

    if(jet[i].DeltaR(gp1)<dR && (hardest_Pt_part_1<0 || jet[i].Pt()>hardest_Pt_part_1)){
      i_part_1 = i;
    } else if(jet[i].DeltaR(gp2)<dR && (hardest_Pt_part_2<0 || jet[i].Pt()>hardest_Pt_part_2)){
      i_part_2 = i;
    }
    
    //if(i_part_2==i_part_1 && i_part_1==-1){
    //  cout << jet[i].DeltaR(gp1) << " " << jet[i].DeltaR(gp2) << " " << jet[i].Pt() << " " << gp1.Pt() << " " << gp2.Pt() << endl;
    //}
  }

  // sanity check (assuming leading and subleading jets cannot be the same
  if(i_part_1==i_part_2){
    //cout << i_part_1 << " " << jet[i_part_1].Pt() << " " << gp1.Pt() << " " << jet[i_part_2].Pt() << " " << gp2.Pt() << endl;
    //std::cout << "Warning same parton match to jet" << std::endl;
  }

  // check if the matches are also the leading
  // and subleading jets
  float azzecched = 0;
  if(i_part_1==i_lead_Pt || i_part_1==i_subl_Pt) azzecched += 0.5;
  if(i_part_2==i_lead_Pt || i_part_2==i_subl_Pt) azzecched += 0.5;

  return azzecched;

}

TH1F * initproduct(TH1F * hA,TH1F* hB, int rebinA = 1, int rebinB=1,double integral = -1.){
  int nbinsA = hA->GetNbinsX();
  int nbinsB = hA->GetNbinsX();
  double min = hA->GetBinLowEdge(1)*hB->GetBinLowEdge(1);
  double max = hA->GetBinLowEdge(nbinsA+1)*hB->GetBinLowEdge(nbinsB+1);
  //Get the actual name from the original histograms 
  string name =(string)(hA->GetName()) +"_vs_"+ (string)(hB->GetName());
  
  //Initialize histogram 
  TH1F * result = new TH1F(name.c_str(),name.c_str(),nbinsA*nbinsB,min,max);
  return result;
}

TH1F * makeproduct(TH1F * hA,TH1F* hB, int rebinA = 1, int rebinB=1,double integral = -1.){

  //Make temporary histos to rebin
  //  TH1F *hA = (TH1F*)h_A->Clone("hA");
  // TH1F *hB = (TH1F*)h_B->Clone("hB");

  //  hA->Rebin(rebinA);
  // hB->Rebin(rebinB);
  
  //get nbins from new histos
  int nbinsA = hA->GetNbinsX();
  int nbinsB = hA->GetNbinsX();
  double min = hA->GetBinLowEdge(1)*hB->GetBinLowEdge(1);
  double max = hA->GetBinLowEdge(nbinsA+1)*hB->GetBinLowEdge(nbinsB+1);
  //Get the actual name from the original histograms 
  string name =(string)(hA->GetName()) +"_vs_"+ (string)(hB->GetName());
  
  //Initialize histogram 
  TH1F * result = new TH1F(name.c_str(),name.c_str(),nbinsA*nbinsB,min,max);
  //Fill histogram
  for(int i =1; i<= nbinsA;++i){
    for(int j =1; j<= nbinsB;++j){
      double value = hA->GetBinContent(i)*hB->GetBinContent(j);
      int k = ((i-1)*nbinsB)+j;
      result->SetBinContent(k,value);
    }
  }
  if( integral <= 0.)integral = hB->Integral()/result->Integral();
  else integral = integral / result->Integral();
  result->Scale(integral);
  return result;

}

//void initHistogramsSysts(TH1F* histo, TString name, TString, int, float, float , bool useOnlyNominal=false);

void systWeights::copySysts(systWeights sys){
  for(int i =0; i < sys.maxSysts;++i){
    this->weightedNames[i]=sys.weightedNames[i];
    this->weightedSysts[i]=sys.weightedSysts[i];

  }
  this->setMax(sys.maxSysts);
  this->setMaxNonPDF(sys.maxSystsNonPDF);
  this->nPDF=sys.nPDF;
  this->nCategories=sys.nCategories;  
  this->addQ2=sys.addQ2;
  this->addPDF=sys.addPDF;
  this->addTopPt=sys.addTopPt;
  this->addVHF=sys.addVHF;
  this->addTTSplit=sys.addTTSplit;
}


void systWeights::prepareDefault(bool addDefault, bool addQ2, bool addPDF, bool addTopPt,bool addVHF, bool addTTSplit, int numPDF){
  this->addPDF=addPDF;
  this->addQ2=addQ2;
  this->addTopPt=addTopPt;
  this->addVHF=addVHF;
  this->addTTSplit=addTTSplit;
  this->nPDF=numPDF;
  this->nCategories=1;
  categoriesNames[0]="";
  this->wCats[0]=1.0;
  if(addDefault){
    //    int MAX = this->maxSysts;
    this->weightedNames[0]="";
    this->weightedNames[1]="puUp";
    this->weightedNames[2]="puDown";
    //this->weightedNames[11]="trigUp";
    //this->weightedNames[12]="trigDown";
    this->setMax(3);
    this->setMaxNonPDF(3);
    this->weightedNames[this->maxSysts]="";
  }
  if(addQ2){
    this->weightedNames[this->maxSysts]= "q2Up";
    this->weightedNames[this->maxSysts+1]= "q2Down";
    this->setMax(this->maxSysts+2);
    this->setMaxNonPDF(this->maxSystsNonPDF+2);
    this->weightedNames[this->maxSysts]= "";
  }

  if(addTopPt){
    this->weightedNames[this->maxSysts]="topPtWeightUp";
    this->weightedNames[this->maxSysts+1]="topPtWeightDown";
    this->setMax(this->maxSysts+2);
    this->setMaxNonPDF(this->maxSystsNonPDF+2);
    this->weightedNames[this->maxSysts]= "";
  }
  

  if(addVHF){
    this->weightedNames[this->maxSysts]="VHFWeightUp";
    this->weightedNames[this->maxSysts+1]="VHFWeightDown";
    this->setMax(this->maxSysts+2);
    this->setMaxNonPDF(this->maxSystsNonPDF+2);
    this->weightedNames[this->maxSysts]= "";
  }

  if(addTTSplit){
    //    this->weightedNames[this->maxSysts]="2lep";
    //    this->weightedNames[this->maxSysts+1]="1lep";
    //    this->weightedNames[this->maxSysts+2]="0lep";
    //    this->setMax(this->maxSysts+3);
    //    this->setMaxNonPDF(this->maxSystsNonPDF+3);
    //    this->weightedNames[this->maxSysts]= "";
    this->nCategories=4;
    categoriesNames[1]="TT0lep";
    categoriesNames[2]="TT1lep";
    categoriesNames[3]="TT2lep";
    this->wCats[1]=1.0;
    this->wCats[2]=1.0;
    this->wCats[3]=1.0;

  }


  /*  if(addkFact){
    this->weightedNames[this->maxSysts]="VHFWeightUp";
    this->weightedNames[this->maxSysts+1]="VHFWeightDown";
    this->setMax(this->maxSysts+2);
    this->setMaxNonPDF(this->maxSystsNonPDF+2);
    this->weightedNames[this->maxSysts]= "";
    }*/

  if(addPDF){
    this->weightedNames[this->maxSysts]= "pdf_totalUp";
    this->weightedNames[this->maxSysts+1]= "pdf_totalDown";
    //this->weightedNames[this->maxSysts+2]= "pdf_asUp";
    //this->weightedNames[this->maxSysts+3]= "pdf_asDown";
    //this->weightedNames[this->maxSysts+4]= "pdf_zmUp";
    //this->weightedNames[this->maxSysts+5]= "pdf_zmDown";
    this->setMax(this->maxSysts+2);
    this->setMaxNonPDF(this->maxSystsNonPDF+2);
    int nPDF=this->nPDF;
    for(int i =0; i < nPDF;++i){
      stringstream ss;
      ss<< i+1;
      this->weightedNames[i+this->maxSysts]= "pdf"+ss.str();
    }
    this->setMax(maxSysts+nPDF);
    this->weightedNames[this->maxSysts]= "";
  }
  
}
void systWeights::addSyst(string name){
  this->weightedNames[this->maxSysts]= name;
  this->setMax(maxSysts+1);
  if(name.find("pdf")!=std::string::npos)this->setMaxNonPDF(maxSysts+1);
  this->weightedNames[this->maxSysts]= "";
}

void systWeights::addSystNonPDF(string name){
  this->weightedNames[this->maxSystsNonPDF]= name;
  this->setMaxNonPDF(maxSystsNonPDF+1);
  int nPDF=this->nPDF;
  for(int i =0; i < nPDF;++i){
    stringstream ss;
    ss<< i+1;
    this->weightedNames[i+this->maxSystsNonPDF]= "pdf"+ss.str();
  }
  this->setMax(maxSystsNonPDF+nPDF);
  this->weightedNames[this->maxSysts]= "";
}

void systWeights::addkFact(string name){
  string up=name+"Up";
  string down=name+"Down";
  cout << " adding syst "<< up<<endl;
  this->addSystNonPDF(up);
  this->addSystNonPDF(down);
}

void systWeights::setkFact(string name, float kfact_nom, float kfact_up,float kfact_down, bool mult){
  //  void setkFact(string name,float kfact_nom, float kfact_up,float kfact_down, float w_zero=1.0, mult=true);
  float zerofact=1.0;
  if(mult)zerofact=this->weightedSysts[0];
  string up = name+"Up";
  string down = name+"Down";
  float valueup=kfact_up/kfact_nom;
  float valuedown=kfact_down/kfact_nom;
  //  cout << "setting syst "<< up<<endl;
  //  cout << "values nom "<<kfact_nom<< " up "<< kfact_up << " down "<< kfact_down << " valup "<< valueup<< " valdown "<< valuedown <<" zerofact "<< zerofact<<endl;
  this->setSystValue(up, valueup*zerofact);
  this->setSystValue(down, valuedown*zerofact);
}

void systWeights::setPDFWeights(float * wpdfs, double * xsections, int numPDFs, float wzero, bool mult){
  float zerofact=1.0;
  if(mult)zerofact=this->weightedSysts[0];
  for (int i = 1; i <= numPDFs; ++i){
    this->setPDFValue(i,zerofact*wpdfs[i]/(wzero*xsections[i]));
  }
  this->setSystValue("pdf_asUp", this->getPDFValue(this->nPDF-2)/wzero);
  this->setSystValue("pdf_asDown", zerofact);
  this->setSystValue("pdf_zmUp", this->getPDFValue(this->nPDF-1)/wzero);
  this->setSystValue("pdf_zmDown", zerofact);
  this->setSystValue("pdf_totalUp", zerofact);
  this->setSystValue("pdf_totalDown", zerofact);
}

//void systWeights::setTWeight(float tweight, float totalweight){
void systWeights::setTWeight(float tweight, float wtotsample,bool mult){
  float zerofact=1.0;
  //  cout << " weighted syst 0 is "<< weightedSysts[0]<<endl;
  if(mult)zerofact=this->weightedSysts[0];
  this->setSystValue("topPtWeightUp", zerofact*tweight/wtotsample);
  this->setSystValue("topPtWeightDown", zerofact/tweight*wtotsample);
}

void systWeights::setVHFWeight(int vhf,bool mult,double shiftval){
  float zerofact=1.0;
  double w_shift=0.0;
  //  cout << "vhf is "<<vhf<<endl;
  if (vhf>1)w_shift=shiftval;
  //  cout << " weighted syst 0 is "<< weightedSysts[0]<<endl;
  if(mult)zerofact=this->weightedSysts[0];
  this->setSystValue("VHFWeightUp", zerofact*(1+w_shift));
  this->setSystValue("VHFWeightDown", zerofact*(1-w_shift));
}


void systWeights::setQ2Weights(float q2up, float q2down, float wzero, bool mult){
  float zerofact=1.0;
  if(mult){
    zerofact=this->weightedSysts[0];
    //    cout <<  "zerofact "<< zerofact << endl;
  }
  //cout <<  "zerofact "<< zerofact << " q2up weight "<< q2up/wzero << " tot to fill "<< zerofact*q2up/wzero<<endl;
  //cout <<  "zerofact "<< zerofact << " q2down weight "<< q2down/wzero << " tot to fill "<< zerofact*q2down/wzero<<endl;
  this->setSystValue("q2Up", zerofact*q2up/wzero);
  this->setSystValue("q2Down", zerofact*q2down/wzero);
}

double systWeights::getPDFValue(int numPDF){
  if(!addPDF){ cout << "error! No PDF used, this will do nothing."<<endl;return 0.;}
  int MIN = this->maxSystsNonPDF;
  return (double)this->weightedSysts[numPDF+MIN];

}
void systWeights::setPDFValue(int numPDF, double w){
  if(!addPDF){ cout << "error! No PDF used, this will do nothing."<<endl;return;}
  int MIN = this->maxSystsNonPDF;
  this->weightedSysts[numPDF+MIN]=w;

}

void systWeights::calcPDFHisto(TH1F** histo, TH1F* singleHisto, double scalefactor, int c){//EXPERIMENTAL
  if(!addPDF){ cout << "error! No PDF used, this will do nothing."<<endl;return;}
  int MAX = this->maxSysts;
  //  for (int c = 0; c < this->nCategories; c++){
    int MIN = this->maxSystsNonPDF+(MAX+1)*c;
    for(int b = 0; b< singleHisto->GetNbinsX();++b){
      float val = singleHisto->GetBinContent(b);
      //      cout << "bin # "<<b << " val "<<val<<endl;
      float mean = 0, devst=0;
      //      cout << "name is "<< singleHisto->GetName()<<endl;
      for(int i = 0; i<this->nPDF;++i ){
	//cout<< " now looking at pdf # "<<i<< " coordinate is "<< MIN+i<<endl;
	//	cout << "is histo there? "<< histo[i+MIN]<<endl;
	//	cout << " histo should be "<< (histo[i+MIN])->GetName()<<endl;
	mean = mean+ histo[i+MIN]->GetBinContent(b);
      }
      mean = mean/this->nPDF;
      //mean = val;//mean/this->nPDF;
      for(int i = 0; i<this->nPDF;++i ){
	devst+=(mean-histo[i+MIN]->GetBinContent(b))*(mean-histo[i+MIN]->GetBinContent(b));
      }
      devst= sqrt(devst/this->nPDF);
      singleHisto->SetBinContent(b,val+devst*scalefactor);
      //      singleHisto->SetBinContent(b,mean+devst*scalefactor);
    }
    //}
}

void systWeights::initHistogramsSysts(TH1F** histo,TString name, TString title, int nbins, float min, float max){
  for (int c = 0; c < this->nCategories; c++){
    int MAX = this->maxSysts;
    bool useOnlyNominal = this->onlyNominal;
    TString cname= (this->categoriesNames[c]).c_str();
    for(int sy=0;sy<(int)MAX;++sy){
      TString ns= (this->weightedNames[sy]).c_str();
      if(sy==0){
	if(c==0) histo[sy+((MAX+1)*(c))]=new TH1F(name,title,nbins,min,max);
	else histo[sy+((MAX+1)*(c))]=new TH1F(name+"_"+cname,title,nbins,min,max);
      }
      if(sy!=0 && !useOnlyNominal) {
	if(c==0)histo[sy+(MAX+1)*c]=new TH1F(name+"_"+ns,title,nbins,min,max);
	else histo[sy+(MAX+1)*c]=new TH1F(name+"_"+ns+"_"+cname,title,nbins,min,max);
      }
      //cout << " initialized histogram "<< histo[sy+(MAX+1)*c]->GetName() <<" sy " << sy << " c  "<< c <<" location " << sy+(MAX+1)*c << endl;

    }
  }
}

void systWeights::setOnlyNominal(bool useOnlyNominal){
  this->onlyNominal=useOnlyNominal;
}

void systWeights::setWCats(double * wcats){
  for(int i =0;i<this->nCategories;++i){
    //    cout << "setting wcat #"<< i << " to be "<<wcats[i]<<endl;
    this->wCats[i]=wcats[i];
  }
 
}

void systWeights::fillHistogramsSysts(TH1F** histo, float v, float w,  float *systWeights, int nFirstSysts, double * wcats, bool verbose){
  if(wcats== NULL){
    wcats = this->wCats;
  }
  //cout << *wcats << endl;
  for (int c = 0; c < this->nCategories; c++){
    int MAX = this->maxSysts;
    bool useOnlyNominal = this->onlyNominal;
    //cout << " filling histo " << histo[0+(MAX+1)*(c)]->GetName()<< endl;
    //cout << "MAX " << MAX <<endl;
    //cout << " filling histo " << histo[0+(MAX+1)*(c)]->GetName()<< " MAX "<<MAX*(1+c)<<" nFirstSysts"<< nFirstSysts<< endl;
    //    cout << "weight 0 "<< systWeights[0]<< " weighted syst 0 "<< this->weightedSysts[0]<<endl;
    for(int sy=0;sy<(int)MAX;++sy){
      //cout << wcats[c] << endl;
      //cout << "sy" << sy << endl;
      //cout << "filling histo " << histo[(int)sy]->GetName()<<endl;
      if(sy!=0 && useOnlyNominal)continue;
      float ws=1.0;
      if(sy<nFirstSysts){
	//cout << "wcats" << "\t" << wcats[c] << endl;
	wcats[c] =1.0;
	//ws=systWeights[sy];
	ws=systWeights[sy]*wcats[c];
	//cout << "wc" << wcats[c] << endl;
	//cout << sy<<"\t"<<systWeights[sy] << endl; 
	//cout << "ws" << ws << endl;
	//cout << sy<<"\t"<<systWeights[sy] << endl;
      }
      else {
	//cout << "wcats" << "\t" << wcats[c] << endl;
	wcats[c] =1.0;
	ws = (this->weightedSysts[(int)sy]*wcats[c]);
	//ws = (this->weightedSysts[(int)sy]);
	//cout << "ws" << ws << endl;
	//cout << this->weightedSysts[(int)sy] << endl;
      }
      //cout << ws << endl;
      //cout << "filling histo " << histo[sy+1]->GetName()<<" value "<< v << " wevt "<< w << " syst number "<< sy<< " name "<< weightedNames[sy]<<" ws value " <<ws<< " wcats" << wcats[c] << endl;    
      //cout <</* "filling histo "<< histo[sy+((MAX+1)*(c))]->GetName()<<*/" value "<< v << " wevt "<< w << " syst number "<< sy<< " name "<< weightedNames[sy]<<" ws value " <<ws<<endl;
      //cout << "c\t" << c << endl;
      //cout << MAX << endl;
      //cout << sy << endl;
      //cout <<sy+((MAX+1)*(c)) << endl;
      //cout << "filling histo " << histo[sy]->GetName()<<endl;
      //histo[1]->Fill(v);
      histo[sy+(MAX+1)*(c)]->Fill(v, w * ws);
      //cout << histo[sy+(MAX+1)*(c)]->Integral() << " " << v << " " << w << " " << ws << endl;
    }
  }
}

void systWeights::fillHistogramsSysts(TH1F** histo, float v, float w, double * wcats, bool verbose ){
  if(wcats==NULL){
    wcats=this->wCats;
  }
  for (int c = 0; c < this->nCategories; c++){
    int MAX = this->maxSysts;
    bool useOnlyNominal = this->onlyNominal;
    for(int sy=0;sy<(int)MAX;++sy){
      if(sy!=0 && useOnlyNominal)continue;
      float ws = (this->weightedSysts[(int)sy])*wcats[c];
      // cout << " filling histogram "<< histo[(int)sy]->GetName() << " with value "<< v <<" and weight "<< w <<" ws "<< ws<<endl;
      histo[sy+(MAX+1)*(c)]->Fill(v, w*ws);
    }
  }
}


void systWeights::createFilesSysts(  TFile ** allFiles, TString basename, TString opt){
  for (int c = 0; c < this->nCategories; c++){
    int MAX = this->maxSystsNonPDF;
    int MAXTOT = this->maxSystsNonPDF;
    bool useOnlyNominal = this->onlyNominal;
    TString cname= (this->categoriesNames[c]).c_str();
    if (c!=0) cname= "_"+cname;
    for(int sy=0;sy<(int)MAX;++sy){
      TString ns= (this->weightedNames[(int)sy]);
      cout << " creating file for syst "<< ns<<endl;
      if (c!=0)     cout << " category is "<< c<<endl;
      //cout << "onlynominal is "<<useOnlyNominal<<endl;

      //    "/afs/cern.ch/user/o/oiorio/public/xAnnapaola/Nov10/res/"+sample + "_" +channel+".root";
      if(sy==0){
	//cout<<" filename is "<< basename+ns+cname+".root"<<endl;
	allFiles[sy+(MAX+1)*c]= TFile::Open((basename+ns+cname+".root"), opt);
      }
      else{
	if(!useOnlyNominal){
	  //if((ns!="1lep") && (ns!="2lep")&& (ns!="0lep")){
	  //	  cout<<" filename is "<< basename+ns+cname+".root"<<endl;
	  allFiles[sy+(MAX+1)*c]= TFile::Open((basename+"_"+ns+cname+".root"), opt);
	}
      }
      //TFile *outTree = TFile::Open(("trees/tree_"+outFileName).c_str(), "RECREATE");
      //cout << " created file at c "<< c << " s "<< sy << " location "<< sy+(MAXTOT+1)*c<< " fname "<<allFiles[sy+(MAXTOT+1)*c]->GetName()<<endl;   
    }
    if(this->addPDF){
      if(!useOnlyNominal)allFiles[MAX+((MAX+1)*c)]= TFile::Open((basename+"_pdf"+cname+".root"), opt);
      //cout << " created file at c "<< c << " s "<< MAX+(MAX+1)*c << " location "<< MAX+(MAX+1)*c<<endl;
      cout<< " fname "<<allFiles[MAX+(MAXTOT+1)*c]->GetName()<<endl;   
    }
  }
  //return allFiles;
}

void systWeights::writeHistogramsSysts(TH1F** histo, TFile **filesout){  
  int MAX= this->maxSystsNonPDF;
  int MAXTOT= this->maxSysts;
  bool useOnlyNominal = this->onlyNominal;
  for (int c = 0; c < this->nCategories; c++){
    TString cname= (this->categoriesNames[c]).c_str();
    if (c!=0) cname= "_"+cname;
    for(int sy=0;sy<(int)MAX;++sy){
      //      cout << "c is now "<< c << " sy "<< sy << " location "<< sy+(MAXTOT+1)*c <<" is histo there? " << histo[sy+(MAXTOT+1)*c] << " file location "<<sy+(MAX+1)*c << " is file there "<< filesout[sy+(MAX+1)*c]<< endl;
      //      cout << " writing histo "<< histo[sy+(MAXTOT+1)*c]->GetName()<< " in file "<< filesout[sy+(MAX+1)*c]->GetName()<<endl;;
      //TString ns= weightedSystsNames((weightedSysts)sy);
      if(!(!useOnlyNominal || sy==0)) continue;
      
      filesout[(int)sy+(MAX+1)*(c)]->cd();
      if(this->addPDF){
	if(this->weightedNames[sy]=="pdf_totalUp")calcPDFHisto(histo, histo[sy+(MAXTOT+1)*(c)],1.0,c);
	if(this->weightedNames[sy]=="pdf_totalDown")calcPDFHisto(histo, histo[sy+(MAXTOT+1)*(c)],-1.0,c);
	;      //this->
      }
      
      histo[sy+(MAXTOT+1)*c]->Write(histo[0]->GetName());
      //    histo[sy]=new TH1F(name+ns,name+ns,nbins,min,max);
      //    filesout[(int)sy]->Close();
    }
    if(this->addPDF){
      if(!useOnlyNominal){
	filesout[MAX+(MAX+1)*(c)]->cd();
	//	cout << " file max is "<< filesout[MAX+(MAX+1)*c]->GetName()<<endl;
	//	int npdf=this->maxSysts-this->maxSystsNonPdf;
	int MAXPDF=this->maxSysts;
	for(int sy=MAX;sy<MAXPDF;++sy){
	  //	  cout << " writing sy "<<sy+(MAXTOT+1)*c<<endl;
	  //	  cout << " histo is there? "<< histo[sy+(MAXTOT+1)*c]<<endl;
	  histo[sy+(MAXTOT+1)*(c)]->Write();
	  //	  cout << " written sy "<< histo[sy+(MAXTOT+1)*c]->GetName()<<endl;
	}
      }
    }
  }
}

void systWeights::writeSingleHistogramSysts(TH1F* histo, TFile **filesout){  
  int MAX= this->maxSystsNonPDF;
  bool useOnlyNominal = this->onlyNominal;
  for (int c = 0; c < this->nCategories; c++){
    TString cname= (this->categoriesNames[c]).c_str();
    cout << c << " " << cname << endl;
    if (c!=0) cname= "_"+cname;
    for(int sy=0;sy<(int)MAX;++sy){
      if(!(!useOnlyNominal || sy==0)) continue;
      cout << " writing histo "<< histo->GetName()<< " in file "<< filesout[(int)sy]->GetName()<<endl;;
      filesout[(int)sy+(MAX+1)*c]->cd();
      histo->Write();
      //histo[sy]=new TH1F(name+ns,name+ns,nbins,min,max);
    }
    if(this->addPDF){
      if(!useOnlyNominal){
	filesout[MAX+(MAX+1)*c]->cd();
	int MAXPDF=this->maxSysts;
	for(int sy=MAX;sy<MAXPDF;++sy){
	  //      cout << " writing sy "<< histo[sy]->GetName()<<endl;
	  histo->Write();
	  //      cout << " written sy "<< histo[sy]->GetName()<<endl;
	}
      }
    }
  }
}


void systWeights::setMax(int max){
  this->maxSysts =  max;
}
void systWeights::setMaxNonPDF(int max){
  this->maxSystsNonPDF =  max;
}
void systWeights::setSystValue(string name, double value, bool mult){
  float zerofact=1.0;
  if(mult)zerofact=this->weightedSysts[0];
  int MAX = this->maxSysts;
  for(int sy=0;sy<(int)MAX;++sy){
    if(this->weightedNames[(int)sy] ==name){
      this->weightedSysts[(int)sy] =value*zerofact;
    }
  }
}

void systWeights::setSystValue(int place, double value, bool mult){
  float zerofact=1.0;
  if(mult)zerofact=this->weightedSysts[0];
  this->weightedSysts[place] =value*zerofact;
}

void systWeights::setWeight(string name, double value, bool mult){
  this->setSystValue(name, value, mult);
}

void systWeights::setWeight(int place, double value, bool mult){
  this->setSystValue(place, value, mult);
}

TString  weightedSystsNames (weightedSysts sy){
  switch(sy){
  case NOSYST : return "";
  case PUUP : return "puUp";
  case PUDOWN : return "puDown";
  case MAXSYSTS : return "";
  }
  return "noSyst";
}

void  initHistogramsSysts (TH1F* histo[(int)MAXSYSTS],TString name, TString title, int nbins, float min, float max, bool useOnlyNominal=false){
  for(int sy=0;sy<(int)MAXSYSTS;++sy){
    TString ns= weightedSystsNames((weightedSysts)sy);
    histo[sy]=new TH1F(name+ns,title,nbins,min,max);
  }
}

void fillHistogramsSysts(TH1F* histo[(int)MAXSYSTS], float v, float w, float systWeight[(int)MAXSYSTS] , bool useOnlyNominal=false){
  for(int sy=0;sy<(int)MAXSYSTS;++sy){
    float ws = systWeight[(int)sy];
    histo[sy]->Fill(v, w*ws);
  }
}

void createFilesSysts(  TFile * allFiles[(int)MAXSYSTS], TString basename, bool useOnlyNominal=false,TString opt = "RECREATE"){
  for(int sy=0;sy<(int)MAXSYSTS;++sy){
    TString ns= weightedSystsNames((weightedSysts)sy);
    //    "/afs/cern.ch/user/o/oiorio/public/xAnnapaola/Nov10/res/"+sample + "_" +channel+".root";
    if(sy==0){
	allFiles[sy]= TFile::Open((basename+ns+".root"), opt);}
    else{
      if(!useOnlyNominal) allFiles[sy]= TFile::Open((basename+"_"+ns+".root"), opt);}
    //TFile *outTree = TFile::Open(("trees/tree_"+outFileName).c_str(), "RECREATE");
    
  }
  //return allFiles;
}

void writeHistogramsSysts(TH1F* histo[(int)MAXSYSTS], TFile *filesout[(int)MAXSYSTS], bool useOnlyNominal=false){  
  for(int sy=0;sy<(int)MAXSYSTS;++sy){
    //cout << " writing histo "<< histo[(int)sy]->GetName()<< " in file "<< filesout[(int)sy]->GetName()<<endl;;
    //TString ns= weightedSystsNames((weightedSysts)sy);
    filesout[(int)sy]->cd();
    histo[sy]->Write(histo[0]->GetName());
    //    histo[sy]=new TH1F(name+ns,name+ns,nbins,min,max);
  }
}

void writeSingleHistogramSysts(TH1F* histo, TFile *filesout[(int)MAXSYSTS], bool useOnlyNominal=false){  
  for(int sy=0;sy<(int)MAXSYSTS;++sy){
    cout << " writing histo "<< histo->GetName()<< " in file "<< filesout[(int)sy]->GetName()<<endl;;
    //TString ns= weightedSystsNames((weightedSysts)sy);
    filesout[(int)sy]->cd();
    histo->Write();
    //    histo[sy]=new TH1F(name+ns,name+ns,nbins,min,max);
  }
}

TH1F * makeproduct(TH2F * h){

  //Make temporary histos to rebin
  //  TH1F *hA = (TH1F*)h_A->Clone("hA");
  // TH1F *hB = (TH1F*)h_B->Clone("hB");

  //  hA->Rebin(rebinA);
  // hB->Rebin(rebinB);
  
  //get nbins from new histos
  int nbinsA = h->GetNbinsX();
  int nbinsB = h->GetNbinsY();
  double min = h->GetXaxis()->GetBinLowEdge(1)*h->GetYaxis()->GetBinLowEdge(1);
  double max = h->GetXaxis()->GetBinLowEdge(nbinsA+1)*h->GetYaxis()->GetBinLowEdge(nbinsB+1);
  //Get the actual name from the original histograms 
  string name = (string)(h->GetName()) + "_1D";
  
  //Initialize histogram 
  TH1F * result = new TH1F(name.c_str(),name.c_str(),nbinsA*nbinsB,min,max);
  //Fill histogram
  for(int i =1; i<= nbinsA;++i){
    for(int j =1; j<= nbinsB;++j){
      double value = h->GetBinContent(i,j);
      int k = ((i-1)*nbinsB)+j;
      result->SetBinContent(k,value);
    }
  }
  //  if( integral <= 0.)integral = hA->Integral()/result->Integral();
  //else integral = integral / result->Integral();
  //result->Scale(integral);
  return result;

}
