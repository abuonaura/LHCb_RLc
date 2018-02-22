from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'osx')

import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from root_numpy import root2array, rec2array
from root_pandas import read_root

import os,sys,getopt,time
import os.path

from utils import *

opts, args = getopt.getopt(sys.argv[1:],"",['sf=','st=','df=','dt=','FakeMu','bdtf='])

dir = '~/Work/LHCb/Analysis/RLc/OutTuples/'
outdir = '~/Work/LHCb/Analysis/RLc/MyTuples/'

signal_fname = "Lb_taunu_new2.pid.root"
signal_tname = "DecayTree"

cutsig_fname = outdir + signal_fname

data_fname = "Lb_Data_Feb18_NewProd.root"
data_tname = "tupleout/DecayTree"

cutdata_fname = outdir + signal_fname
cutdata_tname = "DecayTree"

bdt_fname = outdir + "BDT_data.root"


fakemu = False

for o, a in opts:
    print o
    if o in ("--sf",):
        if a.lower() == "none": signal_fname = None
        else: signal_fname = a
    if o in ("--st",):
        if a.lower() == "none": signal_tname = None
        else: signal_tname = a
    if o in ("--df",):
        if a.lower() == "none": data_fname = None
        else: data_fname = a
    if o in ("--dt",):
        if a.lower() == "none": data_tname = None
        else: data_tname = a
    if o in ("--bdtf",):
        if a.lower() == "none": bdt_fname = None
        else: bdt_fname = a
    if o in ("--FakeMu"):
        fakemu = True

print('Applying cuts')
ApplyCuts2Tuple(dir+signal_fname, signal_tname,cutsig_fname,1,0.6,[2230,2330],0,1,fakemu)
ApplyCuts2Tuple(dir+data_fname, data_tname,cutdata_fname,fakemu)

df_signal = read_root(cutsig_fname,signal_tname,
                      columns=['Lc_M','Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT',
                               'pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2','Lc_IPCHI2_OWNPV',
                               'Lc_BKGCAT','Lb_BKGCAT','p_ProbNNp_corr','pi_ProbNNpi_corr',
                               'K_ProbNNK_corr','pi_pidk_corr','K_pidk_corr','p_pidp_corr',
                               'mu_ProbNNmu_corr','Lb_L0Global_TIS','Lc_L0HadronDecision_TOS',
                               'Lc_Hlt1TrackMVADecision_TOS','Lc_Hlt1TwoTrackMVADecision_TOS',
                               'Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS'])

df_bkg = read_root(cutdata_fname, cutdata_tname,
                  columns=['Lc_M','Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT',
                           'pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2','Lc_IPCHI2_OWNPV',
                           'p_MC15TuneV1_ProbNNp','pi_MC15TuneV1_ProbNNpi','K_MC15TuneV1_ProbNNk',
                           'mu_MC15TuneFLAT4dV1_ProbNNmu','Lb_L0Global_TIS','Lc_L0HadronDecision_TOS',
                           'Lc_Hlt1TrackMVADecision_TOS','Lc_Hlt1TwoTrackMVADecision_TOS',
                           'Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS'])


mLc_s = df_signal['Lc_M']
mLc_b = df_bkg['Lc_M']


#========================================================================================
#Remove events in the Lc peak for background, consider only events in Lc peak for signal and which have Lc_bkgcat<30
df_signal =df_signal.loc[((df_signal['Lc_M']>2260)&(df_signal['Lc_M']<2310))&(df_signal['Lc_BKGCAT']<30)].dropna()
df_bkg = df_bkg.loc[(df_bkg['Lc_M']<2260)|(df_bkg['Lc_M']>2310)].dropna()

'''
#Apply cut on mu_ProbNNmu
if fakemu==False:
    df_signal = df_signal.loc[df_signal['mu_ProbNNmu_corr']>0.6]
    df_bkg = df_bkg.loc[df_bkg['mu_MC15TuneFLAT4dV1_ProbNNmu']>0.6]

#applytrigger requirements on signal + bkg events
df_signal = df_signal.loc[(df_signal['Lb_L0Global_TIS']==1)|(df_signal['Lc_L0HadronDecision_TOS']==1)].dropna()
df_signal = df_signal.loc[(df_signal['Lc_Hlt1TrackMVADecision_TOS']==1)|(df_signal['Lc_Hlt1TwoTrackMVADecision_TOS']==1)].dropna()
df_signal = df_signal.loc[(df_signal['Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS']==1)].dropna()

df_bkg = df_bkg.loc[(df_bkg['Lb_L0Global_TIS']==1)|(df_bkg['Lc_L0HadronDecision_TOS']==1)].dropna()
df_bkg = df_bkg.loc[(df_bkg['Lc_Hlt1TrackMVADecision_TOS']==1)|(df_bkg['Lc_Hlt1TwoTrackMVADecision_TOS']==1)].dropna()
df_bkg = df_bkg.loc[(df_bkg['Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS']==1)].dropna()


#Cut on PID as in stripping line
df_signal =df_signal.loc[((df_signal['pi_pidk_corr']<2)&(df_signal['K_pidk_corr']>4.))&(df_signal['p_pidp_corr']>0.)].dropna()
'''

mLc_s = df_signal['Lc_M']
mLc_b = df_bkg['Lc_M']
#df_bkg=df_bkg.sample(n=len(df_signal))

df_signal =df_signal.rename(columns={"p_ProbNNp_corr": "p_ProbNNp", "pi_ProbNNpi_corr": "pi_ProbNNpi","K_ProbNNK_corr" : 'K_ProbNNk'})
df_bkg =df_bkg.rename(columns={"p_MC15TuneV1_ProbNNp": "p_ProbNNp", "pi_MC15TuneV1_ProbNNpi": "pi_ProbNNpi","K_MC15TuneV1_ProbNNk" : 'K_ProbNNk'})




#========================================================================================
#                                 BDT PARAMETER SEARCH
#========================================================================================

#Remove events in the Lc peak for background, consider only events in Lc peak for signal and which have Lc_bkgcat<30

#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

BDTvars =['Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT', 'pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2','Lc_IPCHI2_OWNPV',
          'p_ProbNNp','pi_ProbNNpi','K_ProbNNk']

df_signal1=df_signal[BDTvars]
df_bkg1=df_bkg[BDTvars]

#Put together evts from signal and bkg tuple
X = np.concatenate((df_signal1, df_bkg1))
y = np.concatenate((np.ones(df_signal1.shape[0]),
                    np.zeros(df_bkg1.shape[0])))

df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                  columns=BDTvars+['y'])
bkg=df[df.y<0.5]
signal=df[df.y>=0.5]

#split data into X and y
X = df[BDTvars]
Y = df['y']

#Split data into test and training set
seed = 0
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print (len(signal), len(bkg), X_train.shape[0])
print (X_test.shape[0])
print('I got here!')

def modelfit(alg, Xtrain, ytrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(Xtrain, ytrain)

    #Predict training set:
    dtrain_predictions = alg.predict(Xtrain)
    dtrain_predprob = alg.predict_proba(Xtrain)[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, Xtrain, ytrain, cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(ytrain.values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(ytrain, dtrain_predprob)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

predictors=BDTvars

'''
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, X_train, y_train, predictors)


param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, min_samples_split=500,
                                                               min_samples_leaf=50,max_depth=3,max_features='sqrt',
                                                               subsample=0.8,random_state=10),
                        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y_train)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt',
                                                               subsample=0.8, random_state=10),
                        param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train, y_train)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

param_test3 = {'min_samples_leaf':range(30,60,10), 'max_features':range(6,14,2)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=15,
                                                               max_features='sqrt', subsample=0.8, random_state=10, min_samples_split=200),
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train, y_train)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
'''

gbm_tuned_4 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=80,max_depth=10, min_samples_split=200, min_samples_leaf=30,
                                         subsample=0.85, random_state=10, max_features=5,warm_start=True)
modelfit(gbm_tuned_4, X_train, y_train, predictors, performCV=False)

y_predicted = gbm_tuned_4.predict(X_test)

#Print area under ROC curve and plot ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, gbm_tuned_4.decision_function(X_test))

roc = roc_auc_score(y_test, gbm_tuned_4.decision_function(X_test))

print classification_report(y_test, y_predicted,
                    target_names=["background", "signal"])
print "Area under ROC curve: %.4f"%(roc)

#Plot ROC curve

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.show(block=False)

#Plot results
def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=20):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    #low_high = (-0.1,0.1)
    low_high = (low,high)

    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             #histtype='stepfilled', density=True,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             #histtype='stepfilled', density=True,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
    #                          bins=bins, range=low_high, density=True)
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              #bins=bins, range=low_high, density=True)
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')

fig6 = plt.figure()
compare_train_test(gbm_tuned_4, X_train, y_train, X_test, y_test)
plt.show(block = False)

'''
#Write the classifier to a TTree
from root_numpy import array2root

df_data = read_root(data_fname,data_tname,
                    columns=['Lc_M','Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT',
                             'pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2','Lc_IPCHI2_OWNPV',
                             'p_MC15TuneV1_ProbNNp','pi_MC15TuneV1_ProbNNpi','K_MC15TuneV1_ProbNNk',
                             'Lb_L0Global_TIS','Lc_L0HadronDecision_TOS','Lc_Hlt1TrackMVADecision_TOS',
                             'Lc_Hlt1TwoTrackMVADecision_TOS','Lb_Hlt2XcMuXForTauB2XcMuDecision_TOS'])

df_data =df_data.rename(columns={"p_MC15TuneV1_ProbNNp": "p_ProbNNp", "pi_MC15TuneV1_ProbNNpi": "pi_ProbNNpi","K_MC15TuneV1_ProbNNk" : 'K_ProbNNk'})


y_predicted = gbm_tuned_4.decision_function(df_data[BDTvars])
y_predicted.dtype = [('y', np.float64)]
nevt = df_data.index.values
nevt.dtype = [('nevt', np.int64)]
if os.path.isfile(bdt_fname):
    print('File already created! Delete file or change name!')
else:
    array2root(y_predicted, bdt_fname, "BDToutput")




data =df_data
data=data.assign(BDTMC=y_predicted)

mLc_s = data['Lc_M'].where(data['BDTMC']>-1.8).dropna()
mLc_b = data['Lc_M'].where(data['BDTMC']<-1.8).dropna()
mLc_tot = data['Lc_M']


fig2 = plt.figure()
plt_s = plt.subplot(1,3,1)
plt_s.set_xlabel('$\Lambda_{c}$ mass (MeV)')
plt.hist(mLc_s, bins=50)
plt_s.set_title('BDT > -1.8')
plt_b = plt.subplot(1,3,2,sharex = plt_s)
plt_b.set_xlabel('$\Lambda_{c}$ mass (MeV)')
plt.hist(mLc_b, bins=50)
plt_b.set_title('BDT < -1.8')
plt_tot = plt.subplot(1,3,3,sharex = plt_s)
plt_tot.set_xlabel('$\Lambda_{c}$ mass (MeV)')
plt_tot.hist(mLc_tot, bins=50, density=False, label='No cut')
plt_tot.hist(mLc_s, bins=50, density=False,color='yellow', label='BDT>-1.8')
plt_tot.hist(mLc_b, bins=50, density=False,alpha = 0.5,color='green', label='BDT<-1.8')
plt_tot.legend(prop={'size': 10})
plt.show(block = False)
'''
