import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from root_numpy import root2array, rec2array
from root_pandas import read_root

from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import gStyle
from ROOT import RooStats
from ROOT import RooFit as RF
from ROOT import RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
import ROOT as r

from utils import *

f = r.TFile('../MyTuples/Lb_Data_Final_Sweighted.root', 'READ')
t = f.Get('DecayTree')

h_mTOT = CreateHisto('h_mTOT', 80, 2200, 2400, 1, 0, 'MeV/c^{2}', '')
h_mTOT_ppi = CreateHisto('h_mTOT_ppi', 80, 1500, 2100, 2, 0, 'MeV/c^{2}', '')
h_mTOT_pk = CreateHisto('h_mTOT_pk', 80, 1500, 2100, 600, 0, 'MeV/c^{2}', '')

h_mTOT_ppi_cut = CreateHisto('h_mTOT_ppi_cut', 80, 1500, 2100, 2, 3004, 'MeV/c^{2}', '')
h_mTOT_pk_cut = CreateHisto('h_mTOT_pk_cut', 80, 1500, 2100, 600, 3004, 'MeV/c^{2}', '')

h_pProbNNpi = r.TH2F('h_pProbNNpi','',30,1500,2200,30,0,1)
h_pProbNNk = r.TH2F('h_pProbNNk','',30,1200,2200,30,0,1)


h_pProbNNk_mLc = CreateHisto('h_pProbNNk_mLc', 30, -1,1, 1, 0, 'p_ProbNNp - p_ProbNNK', '')
h_pProbNNk_mDp = CreateHisto('h_pProbNNk_mDp', 30, -1,1, 2, 0, 'p_ProbNNp - p_ProbNNK', '')
h_pProbNNk_mDs = CreateHisto('h_pProbNNk_mDs', 30, -1,1, 4, 0, 'p_ProbNNp - p_ProbNNK', '')
h_pProbNNk_mLc_loss = CreateHisto('h_pProbNNk_mLc_loss', 30, -1,1, 1, 0, 'p_ProbNNp - p_ProbNNK', '')

h_pProbNNpi_mLc = CreateHisto('h_pProbNNpi_mLc', 30, -1,1, 1, 0, 'p_ProbNNp - p_ProbNNpi', '')
h_pProbNNpi_mDp = CreateHisto('h_pProbNNpi_mDp', 30, -1,1, 2, 0, 'p_ProbNNp - p_ProbNNpi', '')
h_pProbNNpi_mDs = CreateHisto('h_pProbNNpi_mDs', 30, -1,1, 4, 0, 'p_ProbNNp - p_ProbNNpi', '')
h_pProbNNpi_mLc_loss = CreateHisto('h_pProbNNpi_mLc_loss', 30, -1,1, 1, 0, 'p_ProbNNp - p_ProbNNK', '')




nentries = t.GetEntries()

for i in range(nentries):
    t.GetEntry(i)

    P_p = np.matrix([t.p_PX, t.p_PY, t.p_PZ]) #Momentum proton
    P_pi = np.matrix([t.pi_PX, t.pi_PY, t.pi_PZ]) #Momentum pion
    P_k = np.matrix([t.K_PX, t.K_PY, t.K_PZ]) #Momentum kaon
    m_pi = t.pi_M
    m_k = t.K_M
    m_p = t.p_M
    
    #Different mass hypothesis for the proton
    #  1.   The proton is a misID pion +/-
    m_ppi = 139.57018 #+/- 0.00035 MeV (PDG)
    #  2.   The proton is a misID kaon +/-
    m_pk = 493.677 #+/- 0.016 MeV (PDG)
    
    #Compute Energy of pion, kaon and proton in the 3 hypothesis (right p, p as pi and p as k)
    E_pi = r.TMath.Sqrt(P_pi*P_pi.transpose() + m_pi*m_pi)
    E_k = r.TMath.Sqrt(P_k*P_k.transpose() + m_k*m_k)
    E_p = r.TMath.Sqrt(P_p*P_p.transpose() + m_p*m_p)
    E_ppi = r.TMath.Sqrt(P_p*P_p.transpose() + m_ppi*m_ppi)
    E_pk = r.TMath.Sqrt(P_p*P_p.transpose() + m_pk*m_pk)
    
    #Compute total mother momentum and energy in the 3 hypothesis (right p, p as pi and p as k)
    pTOT = P_p+P_pi+P_k
    ETOT = E_pi + E_k + E_p
    ETOT_ppi = E_pi + E_k + E_ppi
    ETOT_pk = E_pi + E_k + E_pk
    
    #Mother particle reco mass in the 3 hypothesis (right p, p as pi and p as k)
    mTOT = r.TMath.Sqrt(ETOT*ETOT - pTOT*pTOT.transpose())
    mTOT_ppi = r.TMath.Sqrt(ETOT_ppi*ETOT_ppi - pTOT*pTOT.transpose())
    mTOT_pk = r.TMath.Sqrt(ETOT_pk*ETOT_pk - pTOT*pTOT.transpose())
    
    #Fill histos
    h_mTOT.Fill(mTOT,t.sw_sig)
    
    #select bkg regions
    if t.Lc_M<2260 or t.Lc_M>2310:
        
        h_mTOT_ppi.Fill(mTOT_ppi)
        h_mTOT_pk.Fill(mTOT_pk)
        
        h_pProbNNpi.Fill(mTOT_ppi,t.p_MC15TuneV1_ProbNNp-t.p_MC15TuneV1_ProbNNpi)
        h_pProbNNk.Fill(mTOT_pk, t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk)
        
        #ProbNNp - ProbNNpi in Dplus Peak
        if (mTOT_ppi>1850 and mTOT_ppi<1890):
            h_pProbNNpi_mDp.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNpi)
            #ProbNNp - ProbNNpi in Ds Peak
        if(mTOT_ppi>1950 and mTOT_ppi<1990):
            h_pProbNNpi_mDs.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNpi)
                
        #ProbNNp - ProbNNk in Dplus Peak
        if (mTOT_pk>1850 and mTOT_pk<1890):
            h_pProbNNk_mDp.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk)
            #ProbNNp - ProbNNpi in Ds Peak
        if(mTOT_pk>1950 and mTOT_pk<1990):
            h_pProbNNk_mDs.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk)

    #ProbNNp - ProbNNpi in Lc Peak
    h_pProbNNpi_mLc.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNpi,t.sw_sig)
    #ProbNNp - ProbNNpi in Lc Peak
    h_pProbNNk_mLc.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk,t.sw_sig)

        
    #try to cut out evts with ProbNN differences below 0.2:
    #select bkg regions
    if t.Lc_M<2260 or t.Lc_M>2310:
        #cut on differences
        #if t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNpi>0.3:
            
        if t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk>0:
            h_mTOT_pk_cut.Fill(mTOT_pk)
            h_mTOT_ppi_cut.Fill(mTOT_ppi)
            
    #Check fraction of evts loss with cuts:
    if t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNpi<0.3:
        h_pProbNNpi_mLc_loss.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNpi,t.sw_sig)
    if t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk<0:
        h_pProbNNk_mLc_loss.Fill(t.p_MC15TuneV1_ProbNNp -t.p_MC15TuneV1_ProbNNk,t.sw_sig)

#print('Fraction of signal evts not passing the selection on ProbNNp-ProbNNpi: ', h_pProbNNpi_mLc_loss.Integral()/h_pProbNNpi_mLc.Integral())
print('Fraction of signal evts not passing the selection on ProbNNp-ProbNNk: ', h_pProbNNk_mLc_loss.Integral()/h_pProbNNk_mLc.Integral())

gStyle.SetOptStat(0)

c = r.TCanvas('c','',500,500)
h_mTOT_pk.Draw()
h_mTOT_ppi.Draw('same')
legend0=r.TLegend(0.2,0.6,0.45,0.8)
legend0.AddEntry(h_mTOT_ppi, 'p as #pi', "l")
legend0.AddEntry(h_mTOT_pk, 'p as K ', "l")
legend0.Draw()

c1 = r.TCanvas('c1','',500,500)
h_mTOT.Draw()

c2 = r.TCanvas('c2','',1000,500)
c2.Divide(2,1)
c2.cd(1)
h_pProbNNpi.Draw('colz')
c2.cd(2)
h_pProbNNk.Draw('colz')

'''   -----------------------------------> TO MODIFY:
c4 = r.TCanvas('c4','',1000,500)
c4.Divide(2,1)
c4.cd(1)
h.Draw()
c4.cd(2)
h1.Draw()

c5 = r.TCanvas('c5','',1000,500)
c5.Divide(2,1)
c5.cd(1)
h2.Draw()
c5.cd(2)
h3.Draw()
<-----------------------------------
'''

scale_ppi_mLc = 1./h_pProbNNpi_mLc.Integral()
scale_ppi_mDp = 1./h_pProbNNpi_mDp.Integral()
scale_ppi_mDs = 1./h_pProbNNpi_mDs.Integral()

scale_pk_mLc = 1./h_pProbNNk_mLc.Integral()
scale_pk_mDp = 1./h_pProbNNk_mDp.Integral()
scale_pk_mDs = 1./h_pProbNNk_mDs.Integral()

h_pProbNNpi_mLc.Scale(scale_ppi_mLc)
h_pProbNNpi_mDp.Scale(scale_ppi_mDp)
h_pProbNNpi_mDs.Scale(scale_ppi_mDs)

h_pProbNNk_mLc.Scale(scale_pk_mLc)
h_pProbNNk_mDp.Scale(scale_pk_mDp)
h_pProbNNk_mDs.Scale(scale_pk_mDs)

c3 = r.TCanvas('c3','',1000,500)
c3.Divide(2,1)
c3.cd(1)
c3.SetLogy()
h_pProbNNpi_mLc.Draw()
h_pProbNNpi_mDs.Draw('same')
h_pProbNNpi_mDp.Draw('same')
legend=r.TLegend(0.6,0.6,0.8,0.8)
legend.AddEntry(h_pProbNNpi_mDp, 'D^{+} peak', "l")
legend.AddEntry(h_pProbNNpi_mDs, 'D_{s} peak ', "l")
legend.AddEntry(h_pProbNNpi_mLc,'#Lambda_{c} peak','l')
legend.Draw()
c3.cd(2)
c3.SetLogy(1)
h_pProbNNk_mLc.Draw()
h_pProbNNk_mDs.Draw('same')
h_pProbNNk_mDp.Draw('same')
legend1=r.TLegend(0.6,0.6,0.8,0.8)
legend1.AddEntry(h_pProbNNk_mDp, 'D^{+} peak', "l")
legend1.AddEntry(h_pProbNNk_mDs, 'D_{s} peak ', "l")
legend1.AddEntry(h_pProbNNk_mLc,'#Lambda_{c} peak','l')
legend1.Draw()

c4 = r.TCanvas('c4','',500,500)
h_mTOT_pk.Draw()
h_mTOT_pk_cut.Draw('same')
legend2=r.TLegend(0.2,0.6,0.6,0.8)
legend2.AddEntry(h_mTOT_pk, 'All events', "f")
legend2.AddEntry(h_mTOT_pk_cut, 'Events passing cut', 'f')
legend2.Draw()

c5 = r.TCanvas('c5','',500,500)
h_mTOT_ppi.Draw('same')
h_mTOT_ppi_cut.Draw('same')
legend3=r.TLegend(0.2,0.6,0.6,0.8)
legend3.AddEntry(h_mTOT_ppi, 'All events', 'f')
legend3.AddEntry(h_mTOT_ppi_cut, 'Events passing cut', 'f')
legend3.Draw()
