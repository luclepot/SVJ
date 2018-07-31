#include "TFile.h"
#include "TH2.h"
#include <string>

class kFactor{

 public:

  kFactor(std::string type ){
    type_ = type;
  }

  float getkFact(float pt){ 
    float kFact(-1);

    if(type_=="WQCD"){
      if( pt < 150) kFact = 1.43896;
      if( pt > 150 &&  pt< 170) kFact = 1.43896;
      if( pt > 170 &&  pt< 200) kFact = 1.45307;
      if( pt > 200 &&  pt< 230) kFact = 1.41551;
      if( pt > 230 &&  pt< 260) kFact = 1.42199;
      if( pt > 260 &&  pt< 290) kFact = 1.3477;
      if( pt > 290 &&  pt< 320) kFact = 1.35302;
      if( pt > 320 &&  pt< 350) kFact = 1.34289;
      if( pt > 350 &&  pt< 390) kFact = 1.32474;
      if( pt > 390 &&  pt< 430) kFact = 1.23267;
      if( pt > 430 &&  pt< 470) kFact = 1.22641;
      if( pt > 470 &&  pt< 510) kFact = 1.23149;
      if( pt > 510 &&  pt< 550) kFact = 1.21593;
      if( pt > 550 &&  pt< 590) kFact = 1.16506;
      if( pt > 590 &&  pt< 640) kFact = 1.01718;
      if( pt > 640 &&  pt< 690) kFact = 1.01575;
      if( pt > 690 &&  pt< 740) kFact = 1.05425;
      if( pt > 740 &&  pt< 790) kFact = 1.05992;
      if( pt > 790 &&  pt< 840) kFact = 1.01503;
      if( pt > 840 &&  pt< 900) kFact = 1.01761;
      if( pt > 900 &&  pt< 960) kFact = 0.947194;
      if( pt > 960 &&  pt< 1020) kFact = 0.932754;
      if( pt > 1020 &&  pt< 1090) kFact = 1.00849;
      if( pt > 1090 &&  pt< 1160) kFact = 0.94805;
      if( pt > 1160 &&  pt< 1250) kFact = 0.86956;
      if( pt > 1250) kFact = 0.86956;
    }
    


    else if(type_=="WEWK"){
      if( pt < 150 ) kFact = 0.94269;
      if( pt > 150 &&  pt< 170) kFact = 0.94269;
      if( pt > 170 &&  pt< 200) kFact = 0.902615;
      if( pt > 200 &&  pt< 230) kFact = 0.898827;
      if( pt > 230 &&  pt< 260) kFact = 0.959081;
      if( pt > 260 &&  pt< 290) kFact = 0.891248;
      if( pt > 290 &&  pt< 320) kFact = 0.860188;
      if( pt > 320 &&  pt< 350) kFact = 0.884811;
      if( pt > 350 &&  pt< 390) kFact = 0.868131;
      if( pt > 390 &&  pt< 430) kFact = 0.848655;
      if( pt > 430 &&  pt< 470) kFact = 0.806186;
      if( pt > 470 &&  pt< 510) kFact = 0.848507;
      if( pt > 510 &&  pt< 550) kFact = 0.83763;
      if( pt > 550 &&  pt< 590) kFact = 0.792152;
      if( pt > 590 &&  pt< 640) kFact = 0.730731;
      if( pt > 640 &&  pt< 690) kFact = 0.778061;
      if( pt > 690 &&  pt< 740) kFact = 0.771811;
      if( pt > 740 &&  pt< 790) kFact = 0.795004;
      if( pt > 790 &&  pt< 840) kFact = 0.757859;
      if( pt > 840 &&  pt< 900) kFact = 0.709571;
      if( pt > 900 &&  pt< 960) kFact = 0.702751;
      if( pt > 960 &&  pt< 1020) kFact = 0.657821;
      if( pt > 1020 &&  pt< 1090) kFact = 0.762559;
      if( pt > 1090 &&  pt< 1160) kFact = 0.845925;
      if( pt > 1160 &&  pt< 1250) kFact = 0.674034;
      if( pt > 1250) kFact = 0.674034;
    }

    else if(type_=="ZQCD"){
      if( pt < 150 ) kFact = 1.47528;
      if( pt > 150 &&  pt< 170) kFact = 1.47528;
      if( pt > 170 &&  pt< 200) kFact = 1.5428;
      if( pt > 200 &&  pt< 230) kFact = 1.49376;
      if( pt > 230 &&  pt< 260) kFact = 1.39119;
      if( pt > 260 &&  pt< 290) kFact = 1.40538;
      if( pt > 290 &&  pt< 320) kFact = 1.44661;
      if( pt > 320 &&  pt< 350) kFact = 1.38176;
      if( pt > 350 &&  pt< 390) kFact = 1.37381;
      if( pt > 390 &&  pt< 430) kFact = 1.29145;
      if( pt > 430 &&  pt< 470) kFact = 1.33452;
      if( pt > 470 &&  pt< 510) kFact = 1.25765;
      if( pt > 510 &&  pt< 550) kFact = 1.24265;
      if( pt > 550 &&  pt< 590) kFact = 1.24331;
      if( pt > 590 &&  pt< 640) kFact = 1.16187;
      if( pt > 640 &&  pt< 690) kFact = 1.07349;
      if( pt > 690 &&  pt< 740) kFact = 1.10748;
      if( pt > 740 &&  pt< 790) kFact = 1.06617;
      if( pt > 790 &&  pt< 840) kFact = 1.05616;
      if( pt > 840 &&  pt< 900) kFact = 1.1149;
      if( pt > 900 &&  pt< 960) kFact = 1.03164;
      if( pt > 960 &&  pt< 1020) kFact = 1.06872;
      if( pt > 1020 &&  pt< 1090) kFact = 0.981645;
      if( pt > 1090 &&  pt< 1160) kFact = 0.81729;
      if( pt > 1160 &&  pt< 1250) kFact = 0.924246;
      if( pt > 1250 ) kFact = 0.924246;
    }

    else if(type_=="ZEWK"){
      if( pt < 150) kFact = 0.970592;
      if( pt > 150 &&  pt< 170) kFact = 0.970592;
      if( pt > 170 &&  pt< 200) kFact = 0.964424;
      if( pt > 200 &&  pt< 230) kFact = 0.956695;
      if( pt > 230 &&  pt< 260) kFact = 0.948747;
      if( pt > 260 &&  pt< 290) kFact = 0.941761;
      if( pt > 290 &&  pt< 320) kFact = 0.934246;
      if( pt > 320 &&  pt< 350) kFact = 0.927089;
      if( pt > 350 &&  pt< 390) kFact = 0.919181;
      if( pt > 390 &&  pt< 430) kFact = 0.909926;
      if( pt > 430 &&  pt< 470) kFact = 0.900911;
      if( pt > 470 &&  pt< 510) kFact = 0.892561;
      if( pt > 510 &&  pt< 550) kFact = 0.884353;
      if( pt > 550 &&  pt< 590) kFact = 0.8761;
      if( pt > 590 &&  pt< 640) kFact = 0.867687;
      if( pt > 640 &&  pt< 690) kFact = 0.858047;
      if( pt > 690 &&  pt< 740) kFact = 0.849014;
      if( pt > 740 &&  pt< 790) kFact = 0.840317;
      if( pt > 790 &&  pt< 840) kFact = 0.832017;
      if( pt > 840 &&  pt< 900) kFact = 0.823545;
      if( pt > 900 &&  pt< 960) kFact = 0.814596;
      if( pt > 960 &&  pt< 1020) kFact = 0.806229;
      if( pt > 1020 &&  pt< 1090) kFact = 0.798038;
      if( pt > 1090 &&  pt< 1160) kFact = 0.789694;
      if( pt > 1160 &&  pt< 1250) kFact = 0.781163;
      if( pt > 1250) kFact = 0.781163;
    }


    else {
      std::cout<<"kFact type" + type_ + "not recognised. Please, choose a defined kFact type: WQCD, WEWK"<<std::endl;
 
    }

    return kFact;
  }
 
  float getSyst(float pt)
  { 
    float syst(-1);


    if(type_=="WQCDrenUp"){

      if( pt < 150 ) syst = 1.33939;
      if( pt > 150 &&  pt< 170) syst = 1.33939;
      if( pt > 170 &&  pt< 200) syst = 1.34649;
      if( pt > 200 &&  pt< 230) syst = 1.3115;
      if( pt > 230 &&  pt< 260) syst = 1.30782;
      if( pt > 260 &&  pt< 290) syst = 1.24619;
      if( pt > 290 &&  pt< 320) syst = 1.24531;
      if( pt > 320 &&  pt< 350) syst = 1.23472;
      if( pt > 350 &&  pt< 390) syst = 1.20709;
      if( pt > 390 &&  pt< 430) syst = 1.13368;
      if( pt > 430 &&  pt< 470) syst = 1.12227;
      if( pt > 470 &&  pt< 510) syst = 1.12612;
      if( pt > 510 &&  pt< 550) syst = 1.1118;
      if( pt > 550 &&  pt< 590) syst = 1.06549;
      if( pt > 590 &&  pt< 640) syst = 0.931838;
      if( pt > 640 &&  pt< 690) syst = 0.929282;
      if( pt > 690 &&  pt< 740) syst = 0.959553;
      if( pt > 740 &&  pt< 790) syst = 0.955823;
      if( pt > 790 &&  pt< 840) syst = 0.920614;
      if( pt > 840 &&  pt< 900) syst = 0.917243;
      if( pt > 900 &&  pt< 960) syst = 0.855649;
      if( pt > 960 &&  pt< 1020) syst = 0.84587;
      if( pt > 1020 &&  pt< 1090) syst = 0.906862;
      if( pt > 1090 &&  pt< 1160) syst = 0.858763;
      if( pt > 1160 &&  pt< 1250) syst = 0.794909;
      if( pt > 1250) syst = 0.794909;
    }

    else if(type_=="WQCDrenDown"){

    if( pt < 150) syst = 1.52924;
    if( pt > 150 &&  pt< 170) syst = 1.52924;
    if( pt > 170 &&  pt< 200) syst = 1.55189;
    if( pt > 200 &&  pt< 230) syst = 1.51041;
    if( pt > 230 &&  pt< 260) syst = 1.53402;
    if( pt > 260 &&  pt< 290) syst = 1.44328;
    if( pt > 290 &&  pt< 320) syst = 1.45859;
    if( pt > 320 &&  pt< 350) syst = 1.4486;
    if( pt > 350 &&  pt< 390) syst = 1.44818;
    if( pt > 390 &&  pt< 430) syst = 1.33058;
    if( pt > 430 &&  pt< 470) syst = 1.33239;
    if( pt > 470 &&  pt< 510) syst = 1.33976;
    if( pt > 510 &&  pt< 550) syst = 1.32336;
    if( pt > 550 &&  pt< 590) syst = 1.26854;
    if( pt > 590 &&  pt< 640) syst = 1.10489;
    if( pt > 640 &&  pt< 690) syst = 1.1055;
    if( pt > 690 &&  pt< 740) syst = 1.15559;
    if( pt > 740 &&  pt< 790) syst = 1.17682;
    if( pt > 790 &&  pt< 840) syst = 1.118;
    if( pt > 840 &&  pt< 900) syst = 1.13097;
    if( pt > 900 &&  pt< 960) syst = 1.04988;
    if( pt > 960 &&  pt< 1020) syst = 1.02796;
    if( pt > 1020 &&  pt< 1090) syst = 1.12438;
    if( pt > 1090 &&  pt< 1160) syst = 1.04704;
    if( pt > 1160 &&  pt< 1250) syst = 0.94791;
    if( pt> 1250) syst = 0.94791;
    }
    
    else if(type_=="WQCDfacUp"){
      if( pt < 150 ) syst = 1.44488;
      if( pt > 150 &&  pt< 170) syst = 1.44488;
      if( pt > 170 &&  pt< 200) syst = 1.45847;
      if( pt > 200 &&  pt< 230) syst = 1.41004;
      if( pt > 230 &&  pt< 260) syst = 1.39864;
      if( pt > 260 &&  pt< 290) syst = 1.3391;
      if( pt > 290 &&  pt< 320) syst = 1.33663;
      if( pt > 320 &&  pt< 350) syst = 1.32073;
      if( pt > 350 &&  pt< 390) syst = 1.3076;
      if( pt > 390 &&  pt< 430) syst = 1.20904;
      if( pt > 430 &&  pt< 470) syst = 1.20066;
      if( pt > 470 &&  pt< 510) syst = 1.20462;
      if( pt > 510 &&  pt< 550) syst = 1.18286;
      if( pt > 550 &&  pt< 590) syst = 1.12586;
      if( pt > 590 &&  pt< 640) syst = 0.990615;
      if( pt > 640 &&  pt< 690) syst = 0.984473;
      if( pt > 690 &&  pt< 740) syst = 1.0171;
      if( pt > 740 &&  pt< 790) syst = 1.01706;
      if( pt > 790 &&  pt< 840) syst = 0.986107;
      if( pt > 840 &&  pt< 900) syst = 0.972452;
      if( pt > 900 &&  pt< 960) syst = 0.910183;
      if( pt > 960 &&  pt< 1020) syst = 0.885284;
      if( pt > 1020 &&  pt< 1090) syst = 0.950662;
      if( pt > 1090 &&  pt< 1160) syst = 0.89605;
      if( pt > 1160 &&  pt< 1250) syst = 0.823212;
      if( pt> 1250) syst = 0.823212;
    }
    
    else if(type_=="WQCDfacDown"){
      if( pt < 150 ) syst = 1.43694;
      if( pt > 150 &&  pt< 170) syst = 1.43694;
      if( pt > 170 &&  pt< 200) syst = 1.45181;
      if( pt > 200 &&  pt< 230) syst = 1.42649;
      if( pt > 230 &&  pt< 260) syst = 1.4513;
      if( pt > 260 &&  pt< 290) syst = 1.3624;
      if( pt > 290 &&  pt< 320) syst = 1.37472;
      if( pt > 320 &&  pt< 350) syst = 1.3718;
      if( pt > 350 &&  pt< 390) syst = 1.34637;
      if( pt > 390 &&  pt< 430) syst = 1.26276;
      if( pt > 430 &&  pt< 470) syst = 1.25762;
      if( pt > 470 &&  pt< 510) syst = 1.26389;
      if( pt > 510 &&  pt< 550) syst = 1.25504;
      if( pt > 550 &&  pt< 590) syst = 1.2116;
      if( pt > 590 &&  pt< 640) syst = 1.04846;
      if( pt > 640 &&  pt< 690) syst = 1.052;
      if( pt > 690 &&  pt< 740) syst = 1.09788;
      if( pt > 740 &&  pt< 790) syst = 1.10967;
      if( pt > 790 &&  pt< 840) syst = 1.046;
      if( pt > 840 &&  pt< 900) syst = 1.06961;
      if( pt > 900 &&  pt< 960) syst = 0.989001;
      if( pt > 960 &&  pt< 1020) syst = 0.987722;
      if( pt > 1020 &&  pt< 1090) syst = 1.07552;
      if( pt > 1090 &&  pt< 1160) syst = 1.00802;
      if( pt > 1160 &&  pt< 1250) syst = 0.922172;
      if( pt> 1250) syst = 0.922172;
    }

    else if(type_=="ZQCDrenUp"){

      if( pt < 150 ) syst = 1.38864;
      if( pt > 150 &&  pt< 170) syst = 1.38864;
      if( pt > 170 &&  pt< 200) syst = 1.43444;
      if( pt > 200 &&  pt< 230) syst = 1.39055;
      if( pt > 230 &&  pt< 260) syst = 1.2934;
      if( pt > 260 &&  pt< 290) syst = 1.30618;
      if( pt > 290 &&  pt< 320) syst = 1.33453;
      if( pt > 320 &&  pt< 350) syst = 1.27239;
      if( pt > 350 &&  pt< 390) syst = 1.25981;
      if( pt > 390 &&  pt< 430) syst = 1.18738;
      if( pt > 430 &&  pt< 470) syst = 1.21686;
      if( pt > 470 &&  pt< 510) syst = 1.14351;
      if( pt > 510 &&  pt< 550) syst = 1.13305;
      if( pt > 550 &&  pt< 590) syst = 1.12346;
      if( pt > 590 &&  pt< 640) syst = 1.05153;
      if( pt > 640 &&  pt< 690) syst = 0.98056;
      if( pt > 690 &&  pt< 740) syst = 1.00651;
      if( pt > 740 &&  pt< 790) syst = 0.96642;
      if( pt > 790 &&  pt< 840) syst = 0.956161;
      if( pt > 840 &&  pt< 900) syst = 0.998079;
      if( pt > 900 &&  pt< 960) syst = 0.927421;
      if( pt > 960 &&  pt< 1020) syst = 0.954094;
      if( pt > 1020 &&  pt< 1090) syst = 0.883954;
      if( pt > 1090 &&  pt< 1160) syst = 0.729348;
      if( pt > 1160 &&  pt< 1250) syst = 0.833531;
      if( pt > 1250) syst = 0.833531;
    }

    else if(type_=="ZQCDrenDown"){

      if( pt < 150) syst = 1.55029;
      if( pt > 150 &&  pt< 170) syst = 1.55029;
      if( pt > 170 &&  pt< 200) syst = 1.64816;
      if( pt > 200 &&  pt< 230) syst = 1.59095;
      if( pt > 230 &&  pt< 260) syst = 1.48394;
      if( pt > 260 &&  pt< 290) syst = 1.49876;
      if( pt > 290 &&  pt< 320) syst = 1.55923;
      if( pt > 320 &&  pt< 350) syst = 1.49283;
      if( pt > 350 &&  pt< 390) syst = 1.49279;
      if( pt > 390 &&  pt< 430) syst = 1.39829;
      if( pt > 430 &&  pt< 470) syst = 1.46136;
      if( pt > 470 &&  pt< 510) syst = 1.38252;
      if( pt > 510 &&  pt< 550) syst = 1.36081;
      if( pt > 550 &&  pt< 590) syst = 1.37848;
      if( pt > 590 &&  pt< 640) syst = 1.28575;
      if( pt > 640 &&  pt< 690) syst = 1.17335;
      if( pt > 690 &&  pt< 740) syst = 1.21841;
      if( pt > 740 &&  pt< 790) syst = 1.17707;
      if( pt > 790 &&  pt< 840) syst = 1.16846;
      if( pt > 840 &&  pt< 900) syst = 1.25225;
      if( pt > 900 &&  pt< 960) syst = 1.15246;
      if( pt > 960 &&  pt< 1020) syst = 1.20461;
      if( pt > 1020 &&  pt< 1090) syst = 1.09389;
      if( pt > 1090 &&  pt< 1160) syst = 0.921666;
      if( pt > 1160 &&  pt< 1250) syst = 1.02897;
      if( pt > 1250) syst = 1.02897;

    }

    else if(type_=="ZQCDfacUp"){
      if( pt < 150 ) syst = 1.48202;
      if( pt > 150 &&  pt< 170) syst = 1.48202;
      if( pt > 170 &&  pt< 200) syst = 1.54551;
      if( pt > 200 &&  pt< 230) syst = 1.49889;
      if( pt > 230 &&  pt< 260) syst = 1.37781;
      if( pt > 260 &&  pt< 290) syst = 1.39565;
      if( pt > 290 &&  pt< 320) syst = 1.42918;
      if( pt > 320 &&  pt< 350) syst = 1.35817;
      if( pt > 350 &&  pt< 390) syst = 1.34966;
      if( pt > 390 &&  pt< 430) syst = 1.26377;
      if( pt > 430 &&  pt< 470) syst = 1.30245;
      if( pt > 470 &&  pt< 510) syst = 1.22426;
      if( pt > 510 &&  pt< 550) syst = 1.20589;
      if( pt > 550 &&  pt< 590) syst = 1.20726;
      if( pt > 590 &&  pt< 640) syst = 1.12468;
      if( pt > 640 &&  pt< 690) syst = 1.04086;
      if( pt > 690 &&  pt< 740) syst = 1.06898;
      if( pt > 740 &&  pt< 790) syst = 1.02577;
      if( pt > 790 &&  pt< 840) syst = 1.01591;
      if( pt > 840 &&  pt< 900) syst = 1.06486;
      if( pt > 900 &&  pt< 960) syst = 0.986133;
      if( pt > 960 &&  pt< 1020) syst = 1.01224;
      if( pt > 1020 &&  pt< 1090) syst = 0.937427;
      if( pt > 1090 &&  pt< 1160) syst = 0.77774;
      if( pt > 1160 &&  pt< 1250) syst = 0.87435;
      if( pt > 1250) syst = 0.87435;

    }

    else if(type_=="ZQCDfacDown"){
      if( pt < 150 ) syst = 1.47257;
      if( pt > 150 &&  pt< 170) syst = 1.47257;
      if( pt > 170 &&  pt< 200) syst = 1.54471;
      if( pt > 200 &&  pt< 230) syst = 1.49357;
      if( pt > 230 &&  pt< 260) syst = 1.41154;
      if( pt > 260 &&  pt< 290) syst = 1.42113;
      if( pt > 290 &&  pt< 320) syst = 1.46933;
      if( pt > 320 &&  pt< 350) syst = 1.41145;
      if( pt > 350 &&  pt< 390) syst = 1.40379;
      if( pt > 390 &&  pt< 430) syst = 1.32597;
      if( pt > 430 &&  pt< 470) syst = 1.37278;
      if( pt > 470 &&  pt< 510) syst = 1.2973;
      if( pt > 510 &&  pt< 550) syst = 1.28653;
      if( pt > 550 &&  pt< 590) syst = 1.28624;
      if( pt > 590 &&  pt< 640) syst = 1.20542;
      if( pt > 640 &&  pt< 690) syst = 1.11142;
      if( pt > 690 &&  pt< 740) syst = 1.1518;
      if( pt > 740 &&  pt< 790) syst = 1.11242;
      if( pt > 790 &&  pt< 840) syst = 1.1022;
      if( pt > 840 &&  pt< 900) syst = 1.17247;
      if( pt > 900 &&  pt< 960) syst = 1.08313;
      if( pt > 960 &&  pt< 1020) syst = 1.13457;
      if( pt > 1020 &&  pt< 1090) syst = 1.03169;
      if( pt > 1090 &&  pt< 1160) syst = 0.862166;
      if( pt > 1160 &&  pt< 1250) syst = 0.981254;
      if( pt > 1250) syst = 0.981254;

    }

    else {
      std::cout<<"kFact type" + type_ + "not recognised. Please, choose a defined kFact type: WQCDrenUp/Down, WQCDfacUp/Down, ZQCDrenUp/Down, ZQCDfacUp/Down"<<std::endl;
      
    }

    return syst;
  }
 
 private:

  std::string type_;
 
};

