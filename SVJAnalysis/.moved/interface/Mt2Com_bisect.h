#ifndef Mt2Com_bisect_h
#define Mt2Com_bisect_h

#include <vector>

//#include "NTFormat/interface/NTEvent.h"
#include <TVector2.h>
#include <TLorentzVector.h>

#include "MT2Utility.h"
#include "mt2bl_bisect.h"
#include "mt2w_bisect.h"
#include <string>

using namespace std;

class Mt2Com_bisect : public mt2bl_bisect::mt2bl, 
                                            public mt2w_bisect::mt2w, 
                                            public mt2_bisect::mt2
{
    public:
    
    Mt2Com_bisect();
    ~Mt2Com_bisect();
    
/*     double calculateMT2w(const std::vector<TLorentzVector>& jets,  */
/*                          const std::vector<TLorentzVector>& bjets,  */
/*                          //const std::vector<IPHCTree::NTMuon>& muon,  */
/*                          const std::vector<IPHCTree::NTElectron>& n, */
/*                          TVector2 vecMet, */
/*                          string mt2type); */

    double calculateMT2w(const std::vector<TLorentzVector>& jets, 
                         const std::vector<TLorentzVector>& bjets, 
                         const TLorentzVector leptonInput, 
                         TVector2 vecMet,
                         string mt2type);


    double mt2wWrapper(TLorentzVector& lep, const TLorentzVector& jet_o,
                       const TLorentzVector& jet_b, float& met, float& metphi,
                       string& mt2type);

    private:
    
    std::vector<TLorentzVector> jets; 
    std::vector<TLorentzVector> bjets; 
    //    std::vector<IPHCTree::NTMuon> muon; 
    //    std::vector<IPHCTree::NTElectron> electron;
    TVector2 vecMet;
    string mt2type;
    TLorentzVector jet_o;
    TLorentzVector jet_b;
    float metval;
    float metphi;
    TLorentzVector lep;
    int n_btag;
    int nMax;
    float metx;
    float mety;
    double pl[4];     // Visible lepton
    double pb1[4];    // bottom on the same side as the visible lepton
    double pb2[4];    // other bottom, paired with the invisible W
    double pmiss_lep[3];
    double pmiss[3];  // <unused>, pmx, pmy   missing pT


};
#endif
