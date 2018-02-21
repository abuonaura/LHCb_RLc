import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from root_numpy import root2array, rec2array
from root_pandas import read_root

from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooStats
from ROOT import RooFit as RF
from ROOT import RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
import ROOT as r

from utils import *

f = r.TFile('../MyTuples/Lb_Data_Feb18_NewProd_sWeights.root', 'READ')
t = f.Get('DecayTreeWCuts')

f_bdt = r.TFile('../MyTuples/BDTMC_new.root', 'READ')
t_bdt = f_bdt.Get('BDToutput')

nentries = t.GetEntries()
nentries_bdt = t_bdt.GetEntries()

h_mLc = CreateHisto('h_mLc',100,2230,2330,1,0,'m_{#Lambda_{c}} (MeV/c^{2})','')
h_mLc_s = CreateHisto('h_mLc_s',100,2230,2330,2,0,'m_{#Lambda_{c}} (MeV/c^{2})','')
h_mLc_b = CreateHisto('h_mLc_b',100,2230,2330,600,0,'m_{#Lambda_{c}} (MeV/c^{2})','')

#Check if the two trees have the same number of entries
if nentries == nentries_bdt:
    print('All ok!')

    for i in range(nentries):
        t.GetEntry(i)
        if(applytrigger(t,i)==1):
            h_mLc.Fill(t.Lc_M)
            #print(i)
            bdt_pass = CheckBDTpass(t_bdt,i,-1.7)
            if(bdt_pass==1):
                mLc_s = t.Lc_M
                h_mLc_s.Fill(mLc_s)
            else:  #BKG evts
                mLc_b = t.Lc_M
                h_mLc_b.Fill(mLc_b)
else:
    print('ERROR! The number of entries does not match!')

c = r.TCanvas('c','',500,500)
h_mLc.Draw()
h_mLc_s.Draw('sames')
h_mLc_b.Draw('sames')
legend = r.TLegend(0.2,0.6,0.45,0.8)
legend.AddEntry(h_mLc, 'no cut','l')
legend.AddEntry(h_mLc_s, 'signal','l')
legend.AddEntry(h_mLc_b, 'background','l')
legend.Draw()


