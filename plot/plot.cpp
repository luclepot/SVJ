#include <string> 
#include <iostream>

using std::cout;
using std::endl; 
using std::string; 

TCanvas *leptonCount(string filename) {
    THStack *hs = new THStack("hs","Lepton Count");
    
    TFile *f = new TFile(filename.c_str()); 
    TH1F *pre = (TH1F*)f->Get("h_pre_lep");
    TH1F *post = (TH1F*)f->Get("h_post_lep");

    pre->SetMarkerStyle(21);
    pre->SetLineStyle(1); 
    pre->SetLineColor(kRed);

    post->SetMarkerStyle(21);
    post->SetLineColor(kBlue);
    pre->SetLineStyle(2); 

    
    hs->Add(post);
    hs->Add(pre);

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

TCanvas *ptCompare(string filename, float xmax=-1., float ymax=-1.) {
    THStack *hs = new THStack("hs","Jet PT;Pt;Count");
    
    TFile *f = new TFile(filename.c_str()); 
    TH1F *pre1 = (TH1F*)f->Get("h_pre_1pt");
    TH1F *pre2 = (TH1F*)f->Get("h_pre_2pt");

    TH1F *post1 = (TH1F*)f->Get("h_post_1pt");
    TH1F *post2 = (TH1F*)f->Get("h_post_2pt");

    pre1->SetLineColor(kRed);
    pre1->SetLineStyle(1);
    pre2->SetLineColor(kOrange); 
    pre2->SetLineStyle(1);
    
    post1->SetLineColor(kBlue);
    post1->SetLineStyle(2);
    post2->SetLineColor(kBlack); 
    post2->SetLineStyle(2);


    hs->Add(pre1);
    hs->Add(pre2);
    hs->Add(post1);
    hs->Add(post2);

    TCanvas *cst = new TCanvas("cst","stacked hists",10,10,700,700);


    cst->cd();

    // cst->SetLogy(); 
    hs->Draw("nostack"); 
    if (xmax > 0)
        hs->GetXaxis()->SetLimits(0., xmax);
    hs->SetMinimum(0);
    
    if (ymax > 0)
        hs->SetMaximum(ymax);

    if (xmax > 0)
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

int plot(string filepath, string outputpath, float xmax=-1., float ymax=-1.) {
    cout << "plotting!" << endl;
    cout << filepath << endl; 

    TImage *img1 = TImage::Create();
    TImage *img2 = TImage::Create();
    
    img1->FromPad(leptonCount(filepath)); 
    string s1 = outputpath + "lepton_counts.png";
    cout << "saving image to path '" << s1 << "'" << endl;
    img1->WriteImage(s1.c_str());


    img2->FromPad(ptCompare(filepath, xmax, ymax)); 
    string s2 = outputpath + "pt_compare.png"; 
    cout << "saving image to path '" << s2 << "'" << endl;
    img2->WriteImage(s2.c_str());
    
    // cout << endl; 
    // string s;
    // cout << "press anything to continue...";
    // cin >> s;
    // cout << endl;
    // cout << "press anything to continue...";
    // cin >> s;
    // cout << endl;
    
    delete img1;
    delete img2;

    
    return 0; 
}
