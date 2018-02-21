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

import os,sys,getopt,time

from utils import *

fname = '../OutTuples/Lb_Data_Feb18_NewProd.root'
tname = 'tupleout/DecayTree'
new_fname = '../MyTuples/Lb_Data_Feb18_NewProd_AfterCut.root'
new_tname = 'DecayTree'
sw_fname = '../MyTuples/Lb_Data_Final_Sweighted.root'
sw_tname = 'DecayTree'
fname_bdt = '../MyTuples/BDT_mupid.root'
tname_bdt = 'BDToutput'
LcM_range = [2230,2330]

f_bdt = r.TFile(fname_bdt,"READ")
t_bdt = f_bdt.Get(tname_bdt)

opts, args = getopt.getopt(sys.argv[1:], "f:t:",['of=','bdt=','bdtcut=','muPIDcut=','toReduce','sw='])

for o, a in opts:
    if o in ("-f",):
        if a.lower() == "none": fname = None
        else: fname = a
    if o in ("-t",):
        if a.lower() == "none": tname = None
        else: tname = a
    if o in ("-of",):
        if a.lower() == "none": new_fname = None
        else: new_fname = a
    if o in ("-bdt",):
        if a.lower() == "none": fname_bdt = None
        else: fname_bdt = a
    if o in ("-bdtcut",):
        bdtcut = float(a)
    if o in ("-muPIDcut",):
        muPIDcut = float(a)
    if o in ('--toReduce'):
        print('**************************************************')
        print('Check to have inserted correct muPIDcut + bdtcut ')
        print('        Use options -muPIDcut and -bdtcut ')
        print('**************************************************')
        if fname == '../OutTuples/Lb_Data_Feb18_NewProd.root':
            CreateCuttedDataTuple(fname, tname, new_fname,t_bdt)
        else:
            CreateCuttedDataTuple(fname, tname, new_fname,t_bdt,1,muPIDcut,LcM_range,bdtcut)
    if o in ("-sw",):
        if a.lower() == "none": sw_fname = None
        else: sw_fname = a


Lc_M = r.RooRealVar('Lc_M','#Lambda_{c} mass',LcM_range[0],LcM_range[1])
ds = GetDataset(new_fname,new_tname,Lc_M)

#Create a workspace named 'w' with the model to fit the Lc_Mass peak
nsig = r.RooRealVar("nsig","N signal evts",100000,0,500000)
nbkg = r.RooRealVar("nbkg","N bkg evts",400000,0,500000)

w = CreateMassFitModel(Lc_M,nsig,nbkg)
model = w.pdf("model")
model.fitTo(ds,RF.Extended(True))

nsig = w.var('nsig')
nbkg = w.var('nbkg')

sData =r.RooStats.SPlot("sData","An SPlot",ds, model, r.RooArgList(nsig, nbkg))

ds_w = GetWeightedDS(ds)

addSweights(new_fname, new_tname, sw_fname,sw_tname, ds_w)

def CheckSweights():
    h= r.TH1F()
    h = splotVariable(sw_fname, sw_tname,"Lc_M", 50,LcM_range[0],LcM_range[1],1,0,'#Lambda_{c}','')
    h.SetDirectory(0)
    canvas = r.TCanvas()
    h.Draw()
