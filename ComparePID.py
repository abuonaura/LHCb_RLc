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

#t_data = OpenTree("../MyTuples/Lb_Data_sWeights.root","DecayTreeWCuts")
#t_MC_taunu = OpenTree("../OutTuples/Lb_Lctaunu.wMVA1.def.wMVA2.def.addVar.pid.root","DecayTree")
f_data = r.TFile('../MyTuples/Lb_Data_sWeights.root', 'READ')
t_data = f_data.Get('DecayTreeWCuts')

f_MC = r.TFile('../OutTuples/Lb_Lctaunu.wMVA1.def.wMVA2.def.addVar.pid.root')
t_MC_taunu = f_MC.Get('DecayTree')

h_pipi_d = CreateHisto('pi_ProbNNpi data',50,0,1,1,0,'pi_ProbNNpi','')
h_piPID_d = CreateHisto('pi_pidk d',50,-100,0,1,0,'pi_pidk','')
h_kk_d = CreateHisto('k_ProbNNk data',50,0,1,1,0,'k_ProbNNk','')
h_kPID_d = CreateHisto('K_pidk data',50,0,100,1,0,'k_pidk','')
h_pp_d = CreateHisto('p_ProbNNp data',50,0,1,1,0,'p_ProbNNp','')
h_pPID_d = CreateHisto('p_pidp data',50,0,100,1,0,'p_pidp','')

print t_data.GetEntries()

#For OLD data file
'''
for i in range(t_data.GetEntries()):
    t_data.GetEntry(i)
    if t_data.K_PIDK>4 and t_data.p_PIDp>0 and t_data.pi_PIDK<2:
        if t_data.pi_ProbNNpi>0.3:
            #h_pipi_d.Fill(t_data.pi_ProbNNpi)
            h_pipi_d.Fill(t_data.pi_ProbNNpi,t_data.sw_sig)
        if t_data.K_ProbNNk>0.2:
            #h_kk_d.Fill(t_data.K_ProbNNk)
            h_kk_d.Fill(t_data.K_ProbNNk,t_data.sw_sig)
        if t_data.p_ProbNNp>0.3:
            #h_pp_d.Fill(t_data.p_ProbNNp)
            h_pp_d.Fill(t_data.p_ProbNNp,t_data.sw_sig)
        h_piPID_d.Fill(t_data.pi_PIDK,t_data.sw_sig)
        h_kPID_d.Fill(t_data.K_PIDK,t_data.sw_sig)
        h_pPID_d.Fill(t_data.p_PIDp,t_data.sw_sig)
'''
for i in range(t_data.GetEntries()):
    t_data.GetEntry(i)
    if t_data.pi_MC15TuneV1_ProbNNpi>0.3:
        h_pipi_d.Fill(t_data.pi_MC15TuneV1_ProbNNpi,t_data.sw_sig)
    #    h_pipi_d.Fill(t_data.pi_MC15TuneV1_ProbNNpi)
    if t_data.K_MC15TuneV1_ProbNNk>0.2:
        h_kk_d.Fill(t_data.K_MC15TuneV1_ProbNNk,t_data.sw_sig)
    #    h_kk_d.Fill(t_data.K_MC15TuneV1_ProbNNk)
    if t_data.p_MC15TuneV1_ProbNNp>0.3:
        h_pp_d.Fill(t_data.p_MC15TuneV1_ProbNNp,t_data.sw_sig)
    #    h_pp_d.Fill(t_data.p_MC15TuneV1_ProbNNp)

h_pipi_mc = CreateHisto('pi_ProbNNpi MC',50,0,1,2,0,'pi_ProbNNpi','')
h_piPID_mc = CreateHisto('pi_pidk MC',50,-100,0,2,0,'pi_pidk','')
h_kk_mc = CreateHisto('k_ProbNNk MC',50,0,1,2,0,'k_ProbNNk','')
h_kPID_mc = CreateHisto('K_pidk MC',50,0,100,2,0,'k_pidk','')
h_pp_mc = CreateHisto('p_ProbNNp MC',50,0,1,2,0,'p_ProbNNp','')
h_pPID_mc = CreateHisto('p_pidp MC',50,0,100,2,0,'p_pidp','')

for i in range(t_MC_taunu.GetEntries()):
    t_MC_taunu.GetEntry(i)
    if t_MC_taunu.K_pidk_corr>4 and t_MC_taunu.p_pidp_corr>0 and t_MC_taunu.pi_pidk_corr<2:
        if t_MC_taunu.pi_ProbNNpi_corr>0.3:
            h_pipi_mc.Fill(t_MC_taunu.pi_ProbNNpi_corr)
        if t_MC_taunu.p_ProbNNp_corr>0.3:
            h_pp_mc.Fill(t_MC_taunu.p_ProbNNp_corr)
        if t_MC_taunu.K_ProbNNK_corr>0.2:
            h_kk_mc.Fill(t_MC_taunu.K_ProbNNK_corr)
        h_piPID_mc.Fill(t_MC_taunu.pi_pidk_corr)
        h_kPID_mc.Fill(t_MC_taunu.K_pidk_corr)
        h_pPID_mc.Fill(t_MC_taunu.p_pidp_corr)

#r.gStyle.SetOptStat(0)
scale_d = 1./h_pipi_d.Integral()
scale_mc = 1./h_pipi_mc.GetEntries()
h_pipi_d.Scale(scale_d)
h_pipi_mc.Scale(scale_mc)
c = r.TCanvas('c','',500,500)
c.SetLogy()
h_pipi_mc.Draw()
h_pipi_d.Draw('sames')
legend=r.TLegend(0.6,0.6,0.8,0.8)
legend.AddEntry(h_pipi_mc, 'MC', "l")
legend.AddEntry(h_pipi_d, 'data', "l")
legend.Draw()

scale_d = 1./h_kk_d.Integral()
scale_mc = 1./h_kk_mc.GetEntries()
h_kk_d.Scale(scale_d)
h_kk_mc.Scale(scale_mc)
c1 = r.TCanvas('c1','',500,500)
c1.SetLogy()
h_kk_mc.Draw()
h_kk_d.Draw('sames')
legend1=r.TLegend(0.6,0.6,0.8,0.8)
legend1.AddEntry(h_kk_mc, 'MC', "l")
legend1.AddEntry(h_kk_d, 'data', "l")
legend1.Draw()

scale_d = 1./h_pp_d.Integral()
scale_mc = 1./h_pp_mc.GetEntries()
h_pp_d.Scale(scale_d)
h_pp_mc.Scale(scale_mc)
c2 = r.TCanvas('c2','',500,500)
c2.SetLogy()
h_pp_mc.Draw()
h_pp_d.Draw('sames')
legend2=r.TLegend(0.6,0.6,0.8,0.8)
legend2.AddEntry(h_pp_mc, 'MC', "l")
legend2.AddEntry(h_pp_d, 'data', "l")
legend2.Draw()

'''
c3 = r.TCanvas('c3','',1500,500)
c3.Divide(3,1)
c3.cd(1)
scale_d = 1./h_piPID_d.Integral()
scale_mc = 1./h_piPID_mc.GetEntries()
h_piPID_d.Scale(scale_d)
h_piPID_mc.Scale(scale_mc)
h_piPID_d.Draw()
h_piPID_mc.Draw('same')
c3.cd(2)
scale_d = 1./h_kPID_d.Integral()
scale_mc = 1./h_kPID_mc.GetEntries()
h_kPID_d.Scale(scale_d)
h_kPID_mc.Scale(scale_mc)
h_kPID_d.Draw()
h_kPID_mc.Draw('same')
c3.cd(3)
scale_d = 1./h_pPID_d.Integral()
scale_mc = 1./h_pPID_mc.GetEntries()
h_pPID_d.Scale(scale_d)
h_pPID_mc.Scale(scale_mc)
h_pPID_d.Draw()
h_pPID_mc.Draw('same')
'''
