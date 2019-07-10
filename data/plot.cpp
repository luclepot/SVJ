#include <string> 

using std::string; 

TCanvas *leptonCount(string filename) {
    THStack *hs = new THStack("hs","Lepton Count");
    
    TFile *f = new TFile(filename.c_str()); 
    TH1F *pre = (TH1F*)f->Get("h_pre_lep");
    TH1F *post = (TH1F*)f->Get("h_post_lep");

    pre->SetMarkerStyle(21);
    pre->SetLineColor(kRed);

    post->SetMarkerStyle(21);
    post->SetLineColor(kBlue);
    
    hs->Add(pre);
    hs->Add(post);

    TCanvas *cst = new TCanvas("cst","stacked hists",10,10,700,700);

    cst->cd();
    hs->Draw("nostack");  

    auto legend = new TLegend(0.6, 0.7, .95, .92);
    legend->SetHeader("The Legend Title","C"); // option "C" allows to center the header
    legend->AddEntry(pre,"Pre-cut","f");
    legend->AddEntry(post,"Post-cut","f");
    legend->Draw();

    return cst;
}

TCanvas *ptCompare(string filename) {
    THStack *hs = new THStack("hs","Jet PT;Pt;Count");
    
    TFile *f = new TFile(filename.c_str()); 
    TH1F *pre1 = (TH1F*)f->Get("h_pre_1pt");
    TH1F *pre2 = (TH1F*)f->Get("h_pre_2pt");

    TH1F *post1 = (TH1F*)f->Get("h_post_1pt");
    TH1F *post2 = (TH1F*)f->Get("h_post_2pt");



    pre1->SetLineColor(kRed);
    pre1->SetLineStyle(1);
    pre2->SetLineColor(kRed); 
    pre2->SetLineStyle(2);
    
    post1->SetLineColor(kBlue);
    post1->SetLineStyle(1);
    post2->SetLineColor(kBlue); 
    post2->SetLineStyle(2);

    hs->Add(post1);
    hs->Add(post2);
    hs->Add(pre1);
    hs->Add(pre2);


    TCanvas *cst = new TCanvas("cst","stacked hists",10,10,700,700);

    float xmax, ymax;
    xmax = 1000.;
    ymax = 1500.;

    cst->cd();

    // cst->SetLogy(); 
    hs->Draw("nostack"); 
    hs->GetXaxis()->SetLimits(0., xmax);
    hs->SetMinimum(0);
    hs->SetMaximum(ymax);
    hs->Draw("nostack"); 

    auto legend = new TLegend(0.6, 0.7, .95, .92);
    legend->SetHeader("Legend","C"); // option "C" allows to center the header
    legend->AddEntry(pre1,"leading jet pre-cut","l");
    legend->AddEntry(pre2,"subleading jet pre-cut","l");
    legend->AddEntry(post1,"leading jet post-cut","l");
    legend->AddEntry(post2,"subleading jet pos-cut","l");
    legend->Draw();

    // cst->SetRangeUser(0, 1500); 

    return cst;
}
