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

f_data = r.TFile('../MyTuples/Lb_Data_Feb18_NewProd_sWeights.root', 'READ')
t_data = f_data.Get('DecayTreeWCuts')

h_pipi_d = CreateHisto('pi_ProbNNpi data',50,0,1,1,0,'pi_ProbNNpi','')
h_kk_d = CreateHisto('k_ProbNNk data',50,0,1,1,0,'k_ProbNNk','')
h_pp_d = CreateHisto('p_ProbNNp data',50,0,1,1,0,'p_ProbNNp','')
h_mu_d = CreateHisto('mu_ProbNNmu data',50,0,1,1,0,'mu_ProbNNmu','')

h_pipi_d_tr = CreateHisto('pi_ProbNNpi data WT trigger',50,0,1,2,0,'pi_ProbNNpi','')
h_kk_d_tr = CreateHisto('k_ProbNNk data WT trigger',50,0,1,2,0,'k_ProbNNk','')

h_pp_d_tr = CreateHisto('p_ProbNNp data WT trigger',50,0,1,2,0,'p_ProbNNp','')
h_pp_d_tr_bkg = CreateHisto('p_ProbNNp data WT triggern in bkg region',50,0,1,600,0,'p_ProbNNp','')

h_mu_d_tr = CreateHisto('mu_ProbNNmu data WT trigger',50,0,1,2,0,'mu_ProbNNmu','')
h_mu_d_tr_bkg = CreateHisto('mu_ProbNNmu data WT trigger in bkg region',50,0,1,600,0,'mu_ProbNNmu','')


print (t_data.GetEntries())
for i in range(t_data.GetEntries()):
    t_data.GetEntry(i)
    #print(t_data.Lb_L0Global_TIS, '  ', t_data.Lc_L0HadronDecision_TOS, '  ', t_data.Lc_Hlt1TrackMVADecision_TOS,
    #'  ', t_data.Lc_Hlt1TwoTrackMVADecision_TOS, '  ', t_data.Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS )
    h_pipi_d.Fill(t_data.pi_MC15TuneV1_ProbNNpi, t_data.sw_sig)
    h_kk_d.Fill(t_data.K_MC15TuneV1_ProbNNk, t_data.sw_sig)
    h_pp_d.Fill(t_data.p_MC15TuneV1_ProbNNp, t_data.sw_sig)
    h_mu_d.Fill(t_data.mu_MC15TuneV1_ProbNNmu, t_data.sw_sig)
    if (t_data.Lb_L0Global_TIS==1 or t_data.Lc_L0HadronDecision_TOS==1):
        if (t_data.Lc_Hlt1TrackMVADecision_TOS==1 or t_data.Lc_Hlt1TwoTrackMVADecision_TOS==1):
            if t_data.Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS==1:
                #print(i)
                h_pipi_d_tr.Fill(t_data.pi_MC15TuneV1_ProbNNpi, t_data.sw_sig)
                h_kk_d_tr.Fill(t_data.K_MC15TuneV1_ProbNNk, t_data.sw_sig)
                h_pp_d_tr.Fill(t_data.p_MC15TuneV1_ProbNNp, t_data.sw_sig)
                #mu_MC15TuneFLAT4dV1_ProbNNmu it is Yandex Tune which should work almost the same for the whole mu_p range
                #mu_MC15TuneV1_ProbNNmu does not work well in the low p region
                h_mu_d_tr.Fill(t_data.mu_MC15TuneFLAT4dV1_ProbNNmu, t_data.sw_sig)

                if t_data.Lc_M <2260 or t_data.Lc_M > 2310:
                    h_mu_d_tr_bkg.Fill(t_data.mu_MC15TuneFLAT4dV1_ProbNNmu)
                    h_pp_d_tr_bkg.Fill(t_data.p_MC15TuneV1_ProbNNp, t_data.sw_sig)

c = r.TCanvas('c','',500,500)
h_pp_d.Draw()
h_pp_d_tr.Draw('same')

c1 = r.TCanvas('c1','',500,500)
h_pipi_d.Draw()
h_pipi_d_tr.Draw('same')

c2 = r.TCanvas('c2','',500,500)
h_kk_d.Draw()
h_kk_d_tr.Draw('same')

c3 = r.TCanvas('c3','',500,500)
h_mu_d.Draw()
h_mu_d_tr.Draw('same')

scale_sig = 1./h_mu_d_tr.Integral()
scale_bkg = 1./h_mu_d_tr_bkg.GetEntries()
h_mu_d_tr.Scale(scale_sig)
h_mu_d_tr_bkg.Scale(scale_bkg)
c4 = r.TCanvas('c4','',500,500)
h_mu_d_tr.Draw()
h_mu_d_tr_bkg.Draw('same')
legend=r.TLegend(0.6,0.6,0.8,0.8)
legend.AddEntry(h_mu_d_tr, 'signal', "l")
legend.AddEntry(h_mu_d_tr_bkg, 'bkg', "l")
legend.Draw()

scale_sig = 1./h_pp_d_tr.Integral()
scale_bkg = 1./h_pp_d_tr_bkg.GetEntries()
h_pp_d_tr.Scale(scale_sig)
h_pp_d_tr_bkg.Scale(scale_bkg)
c5 = r.TCanvas('c5','',500,500)
h_pp_d_tr.Draw()
h_pp_d_tr_bkg.Draw('same')
legend1=r.TLegend(0.6,0.6,0.8,0.8)
legend1.AddEntry(h_pp_d_tr, 'signal', "l")
legend1.AddEntry(h_pp_d_tr_bkg, 'bkg', "l")
legend1.Draw()
