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



inputFile    = '../MyTuples/Lb_Data_Feb18_NewProd_AfterCut.root'
treeName = 'DecayTree'

defaultInputFile = True
defaultTreeName = True

opts, args = getopt.getopt(sys.argv[1:], "f:t:")

for o, a in opts:
    if o in ("-f",):
        if a.lower() == "none": inputFile = None
        else: inputFile = a
        defaultInputFile = False
    if o in ("-t",):
        if a.lower() == "none": treeName = None
        else: treeName = a
        defaultTreeName = False

print (inputFile, treeName)

m_range = [2230,2330]


Lc_M = r.RooRealVar('Lc_M','#Lambda_{c} mass',m_range[0],m_range[1])
ds = GetDataset(inputFile,treeName,Lc_M)

frame = Lc_M.frame(RF.Title('#Lambda_{c} mass peak distribution'))
ds.plotOn(frame)


#Create a workspace named 'w' with the model to fit the Lc_Mass peak
nsig = r.RooRealVar("nsig","N signal evts",100000,0,500000)
nbkg = r.RooRealVar("nbkg","N bkg evts",400000,0,500000);

w = CreateMassFitModel(Lc_M,nsig,nbkg);
model = w.pdf("model")
model.fitTo(ds,RF.Extended(True))
model.plotOn(frame)

c = r.TCanvas()
frame.Draw()


