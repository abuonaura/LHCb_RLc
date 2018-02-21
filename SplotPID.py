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

print ('*************************************')
print ('ATTENTION: Verify files used!')
print ('*************************************')

fname = '../MyTuples/Lb_Data_Final_Sweighted.root'
tname = 'DecayTree'

print('Data file read: ', fname)





def SplotPID(fname, tname):

    h = plotVariable(fname, tname,"pi_MC15TuneV1_ProbNNpi", 50,0,1,1,0,'pi_ProbNNpi')
    h1 = splotVariable(fname, tname,"pi_MC15TuneV1_ProbNNpi", 50,0,1,2,0)
    h.SetDirectory(0)
    h1.SetDirectory(0)
    
    h2 = plotVariable(fname, tname,"pi_MC15TuneV1_ProbNNk", 50,0,1,1,0,'pi_ProbNNk')
    h3 = splotVariable(fname,tname,"pi_MC15TuneV1_ProbNNk", 50,0,1,2,0)
    h2.SetDirectory(0)
    h3.SetDirectory(0)
    
    h4 = plotVariable(fname, tname,"pi_MC15TuneV1_ProbNNp", 50,0,1,1,0,'pi_ProbNNp')
    h5 = splotVariable(fname,tname,"pi_MC15TuneV1_ProbNNp", 50,0,1,2,0)
    h4.SetDirectory(0)
    h5.SetDirectory(0)
    
    return h,h1,h2,h3,h4,h5

h,h1,h2,h3,h4,h5 = SplotPID(fname, tname)

c = r.TCanvas()
c.SetLogy()
h.Draw()
h1.Draw('same')

c1 = r.TCanvas()
c1.SetLogy()
h2.Draw()
h3.Draw('same')

c2 = r.TCanvas()
c2.SetLogy()
h4.Draw()
h5.Draw('same')
