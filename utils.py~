import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os.path

import ROOT as r

from root_numpy import root2array, rec2array
from root_pandas import read_root

def CloneAndReduceTree(fname, tname, newfile_name):
  f = r.TFile(fname,"READ")
  t = f.Get(tname)
  print('Started Cloning')

  newfile = r.TFile(newfile_name,"recreate");
  newtree= t.CloneTree(0);
  newtree.SetName("DecayTree");

  for i in range(500000):
      t.GetEntry(i)
      newtree.Fill()
  newtree.Write()
  newfile.Close()
  f.Close()
  print('Finished cloning')

def CreateReducedDataTree(filename, treename,newfilename, newtreename, range=[]):
    if os.path.isfile(newfilename):
        print('File already created')
    else:
        df= read_root(filename,treename)
        df = df.where((df['Lc_M']>range[0]) & (df['Lc_M']<range[1])).dropna()
        df.to_root(newfilename, key=newtreename)

def GetDataset(fname, tname, *var):
    f = r.TFile(fname)
    t = f.Get(tname)
    ds = r.RooDataSet("ds","ds",t,r.RooArgSet(*var))
    return ds

def addSignalBkgYields(start_sig,start_bkg,range_sig=[],range_bkg=[]):
    yields= r.RooArgList()
    sig = r.RooRealVar("nsig","N signal evts",start_sig,range_sig[0],range_sig[1])
    bkg = r.RooRealVar("nbkg","N bkg evts",start_bkg,range_bkg[0],range_bkg[1]);
    yields.add(sig)
    yields.add(bkg)
    return yields

def CreateMassFitModel(var, nsig,nbkg):
    w = r.RooWorkspace()

    #Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
    mean = r.RooRealVar("mean","mean of gaussians",2290,2270,2310) ;
    sigma1 = r.RooRealVar("sigma1","width of gaussians",3,0,10) ;
    sigma2 = r.RooRealVar("sigma2","width of gaussians",15,0,30) ;
    sig1 = r.RooGaussian("sig1","Signal component 1",var,mean,sigma1)
    sig2 = r.RooGaussian("sig2","Signal component 2",var,mean,sigma2)

    #Build exponential PDF
    l = r.RooRealVar("l", "slope of expo",-0.1, -1., 0.);
    bkg = r.RooExponential("bkg", " bkg with exponential PDF",var,l)

    #Sum the signal components into a composite signal p.d.f.
    sig1frac = r.RooRealVar("sig1frac","fraction of component 1 in signal",0.8,0.,1.)
    sig = r.RooAddPdf("sig","Signal",sig1,sig2,sig1frac)

    yields = r.RooArgList(nsig,nbkg)
    functions = r.RooArgList(sig,bkg)
    #Sum the composite signal and background
    model= r.RooAddPdf("model","g1+g2+a",functions,yields)

    getattr(w,'import')(model)
    return w

def GetWeightedDS(ds):
    ds_w = r.RooDataSet()
    ds_w = ds
    ds_w.SetName('dataWithSWeights')
    return ds_w

def CreateHisto(name, Nbin, Min, Max, color, Fill, Xaxis, Yaxis):
    h = r.TH1F(name, " ", Nbin,Min, Max)
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetFillStyle(Fill)
    h.SetFillColor(color)
    h.GetXaxis().SetTitle(Xaxis)
    h.GetYaxis().SetTitle(Yaxis)
    return h

def addSweights(fname, tname, newfile_name,newtree_name,data):
  f = r.TFile(fname,"READ")
  t = f.Get(tname)
  print('adding Sweights')
  Lc_M = np.zeros(1, dtype=float)
  t.SetBranchAddress("Lc_M", Lc_M)

  newfile = r.TFile(newfile_name,"recreate")
  newtree= t.CloneTree(0)
  newtree.SetName(newtree_name)

  Lc_M_data = np.zeros(1, dtype=float)
  sw_bkg = np.zeros(1, dtype=float)
  sw_sig = np.zeros(1, dtype=float)

  newtree.Branch("sw_bkg",sw_bkg,"sw_bkg/D")
  newtree.Branch("sw_sig",sw_sig,"sw_sig/D")

  for i in range(data.numEntries()):
      t.GetEntry(i)
      sw_bkg[0]= data.get(i).getRealValue("nbkg_sw")
      sw_sig[0] = data.get(i).getRealValue("nsig_sw")
      newtree.Fill()
  newtree.Write()
  newfile.Close()
  f.Close()


def plotVariable(fname, tname, variable, Nbin, min, max,color, Fill, Xaxis='', Yaxis=''):
  f = r.TFile(fname,"READ")
  if os.path.isfile(fname):
      print('File found')
  t = f.Get(tname)
  var = np.zeros(1, dtype=float)
  h = CreateHisto(variable, Nbin, min, max, color, Fill, variable, Yaxis)
  h.Sumw2()
  for i in range(t.GetEntries()):
    t.GetEntry(i)
    h.Fill(getattr(t,variable))
  h.SetDirectory(0)
  return h


def splotVariable(fname, tname, variable, Nbin, min, max,color, Fill, Xaxis='', Yaxis=''):
  f = r.TFile(fname,"READ")
  if os.path.isfile(fname):
      print('File found')
  t = f.Get(tname)

  var = np.zeros(1, dtype=float)
  sw = np.zeros(1, dtype=float)

  histname = variable + '_sweighted'
  h = CreateHisto(histname, Nbin, min, max,  color, Fill, variable, Yaxis)
  h.Sumw2()

  for i in range(t.GetEntries()):
    t.GetEntry(i)
    h.Fill(getattr(t,variable),t.sw_sig)
  h.SetDirectory(0)

  return h

def CheckBDTpass(t_bdt, n_evt, BDTcut):
    t_bdt.GetEntry(n_evt)
    bdt = t_bdt.y
    #Apply a BDT cut
    if bdt>BDTcut: #SIGNAL evts
        bdt_pass = 1
    else:
        bdt_pass = 0
    return bdt_pass

def applytrigger(t,n_evt):
    t.GetEntry(n_evt)
    trigger = 0
    if (t.Lb_L0Global_TIS==1 or t.Lc_L0HadronDecision_TOS==1):
        if (t.Lc_Hlt1TrackMVADecision_TOS==1 or t.Lc_Hlt1TwoTrackMVADecision_TOS==1):
            if t.Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS==1:
                trigger = 1
    return trigger

def applymuProbNNcut(t, n_evt,cut=0.6):
  t.GetEntry(n_evt)
  muProbNN = 0
  if (t.mu_MC15TuneFLAT4dV1_ProbNNmu>cut):
    muProbNN = 1
  return muProbNN

def applyLcMcut(mass,M_min=2230,M_max=2330):
  massok=0
  if mass>M_min and mass<M_max:
    massok=1
  return massok

def CreateCuttedDataTuple(fname, tname, new_fname,t_bdt, apply_cuts=1, muPIDcut=0.6, LcMrange=[2230,2330],BDTcut=-1.7):
  if os.path.isfile(new_fname):
            print('File already created')
  else:
      f = r.TFile(fname,"READ")
      t = f.Get(tname)
      if f:
        print('Started Cloning')
      
      newfile = r.TFile(new_fname,"recreate")
      newtree= t.CloneTree(0)
      newtree.SetName("DecayTree")

      Mmin=LcMrange[0]
      Mmax=LcMrange[1]


      for i in range(t.GetEntries()):
        t.GetEntry(i)
        if apply_cuts==1:
          if applytrigger(t,i) and applymuProbNNcut(t,i,muPIDcut):
            if applyLcMcut(t.Lc_M,Mmin,Mmax) and CheckBDTpass(t_bdt,i,BDTcut):
              newtree.Fill()
            
      newtree.Write()
      newfile.Close()
      f.Close()
    
      print('Finished Cloning')
                                        



