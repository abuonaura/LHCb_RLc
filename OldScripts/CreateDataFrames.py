#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'osx')

import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from root_numpy import root2array, rec2array
from root_pandas import read_root

df_signal = read_root("~/Work/LHCb/Analysis/RLc/OutTuples/Lb_Lctaunu.root","tupleout/DecayTree",
                        columns=['Lc_M','Lb_FD_OWNPV', 'Lc_FD_OWNPV','Lb_FDCHI2_OWNPV', 'Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT','pi_MINIP','p_MINIP','K_MINIP',
                        'pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2','Lc_IP_OWNPV',
                        'Lc_IPCHI2_OWNPV','Lc_BKGCAT','Lb_BKGCAT'])
#print df_signal.head()

#df_bkg = read_root("~/Work/LHCb/Analysis/RLc/OutTuples/Lb2Lcmu_2016data.root","Lb2Lcmu/DecayTree",
df_bkg = read_root("~/Work/LHCb/Analysis/RLc/OutTuples/Lb_Data.root","tupleout/DecayTree",
                    columns=['Lc_M','Lb_FD_OWNPV', 'Lc_FD_OWNPV','Lb_FDCHI2_OWNPV', 'Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT','Lc_IP_OWNPV','Lc_IPCHI2_OWNPV',
                    'pi_MINIP','p_MINIP','K_MINIP',
                    'pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2'])


mLc_s = df_signal['Lc_M']
mLc_b = df_bkg['Lc_M']

fig = plt.figure()
ax1 = plt.subplot(1,2,1)
plt.hist(mLc_s, bins=100)
ax2 = plt.subplot(1,2,2, sharex = ax1)
plt.hist(mLc_b,bins=100)
plt.show(block = False)

fig1 = plt.figure()
plt_bkgcat_Lc = plt.subplot(1,2,1)
plt.hist(df_signal['Lc_BKGCAT'])
plt_bkgcat_Lc.set_xlabel('$\Lambda_c$ bkgcat')
plt_bkgcat_Lb = plt.subplot(1,2,2)
plt.hist(df_signal['Lb_BKGCAT'])
plt_bkgcat_Lb.set_xlabel('$\Lambda_b$ bkgcat')
plt.show(block = False)

#========================================================================================
#Remove events in the Lc peak for background, consider only events in Lc peak for signal and which have Lc_bkgcat<30

df_signal =df_signal.loc[((df_signal['Lc_M']>2260)&(df_signal['Lc_M']<2310))&(df_signal['Lc_BKGCAT']<30)].dropna()
df_bkg = df_bkg.loc[(df_bkg['Lc_M']<2260)|(df_bkg['Lc_M']>2310)].dropna()
mLc_s = df_signal['Lc_M']
mLc_b = df_bkg['Lc_M']
df_bkg=df_bkg.sample(n=len(df_signal))

#plot Lc mass distributions
fig2 = plt.figure()
plt_s = plt.subplot(1,2,1)
plt_s.set_xlabel('$\Lambda_{c}$ mass (MeV)')
plt.hist(mLc_s, bins=50, density=True)
plt_b = plt.subplot(1,2,2,sharex = plt_s)
plt_b.set_xlabel('$\Lambda_{c}$ mass (MeV)')
plt.hist(mLc_b, bins=50, density=True)
plt.show(block = False)

variables =['Lc_M','Lb_FD_OWNPV', 'Lc_FD_OWNPV','Lb_FDCHI2_OWNPV', 'Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT','Lc_IP_OWNPV','Lc_IPCHI2_OWNPV',
            'pi_MINIP','p_MINIP','K_MINIP','pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2']

fig4 = plt.figure(figsize=(10,10))
for i, name in enumerate(variables):
    plots = plt.subplot(4, 6, i+1)
    plots.set_title(name)
    plots.set_yscale('log')
    plots = plt.hist(df_signal[name],bins=50,density=True)
    plots = plt.hist(df_bkg[name],bins=50,density=True, alpha = 0.5)

#========================================================================================
#                                           BDT
#========================================================================================

#Remove events in the Lc peak for background, consider only events in Lc peak for signal and which have Lc_bkgcat<30

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV

#Variables to be considered for the dataframe for the BDT
BDTvars =['Lb_FD_OWNPV', 'Lc_FD_OWNPV','Lb_FDCHI2_OWNPV', 'Lc_FDCHI2_OWNPV','Lc_ENDVERTEX_CHI2','pi_PT','p_PT','K_PT','Lc_IP_OWNPV','Lc_IPCHI2_OWNPV'
            ,'pi_MINIP','p_MINIP','K_MINIP','pi_MINIPCHI2','p_MINIPCHI2','K_MINIPCHI2']


df_signal=df_signal[BDTvars]
df_bkg=df_bkg[BDTvars]

#Put together evts from signal and bkg tuple
X = np.concatenate((df_signal, df_bkg))
y = np.concatenate((np.ones(df_signal.shape[0]),
                    np.zeros(df_bkg.shape[0])))

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


'''
X_train1 = X_train[BDTvars1]
# plotting a scatter matrix
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
fig5 = plt.figure(figsize=(15,7))
scatter = pd.scatter_matrix(X_train1, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, cmap=cmap)
plt.show(block = False)


X_train2 = X_train[BDTvars2]
cmap = cm.get_cmap('gnuplot')
fig6 = plt.figure(figsize=(15,7))
scatter = pd.scatter_matrix(X_train1, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, cmap=cmap)
plt.show(block = False)

'''
'''
nestimators = [10,50,100,150,200]
l_rate = [1,0.5,0.1,0.05,0.01]
roc_auc = []
niter =0

for n in nestimators:
    for l in l_rate:
        print(n, l)
        #define DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_features =3, max_depth=3, min_samples_leaf=5)
        bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME.R',
                         n_estimators=n,
                         learning_rate=l)
        bdt.fit(X_train, y_train)

        y_predicted = bdt.predict(X_test)

        #Print area under ROC curve and plot ROC curve
        fpr_lr, tpr_lr, _ = roc_curve(y_test, bdt.decision_function(X_test))

        roc_auc.append(roc_auc_score(y_test, bdt.decision_function(X_test)))

        #print classification_report(y_test, y_predicted,
        #                    target_names=["background", "signal"])
        print "Area under ROC curve: %.4f"%(roc_auc[niter])
        niter = niter+1
'''




dt = DecisionTreeClassifier(max_features =3, max_depth=3, min_samples_leaf=5)
bdt = AdaBoostClassifier(dt,
                 algorithm='SAMME.R',
                 n_estimators=150,
                 learning_rate=0.5)
bdt.fit(X_train, y_train)

y_predicted = bdt.predict(X_test)

#Print area under ROC curve and plot ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, bdt.decision_function(X_test))

roc = roc_auc_score(y_test, bdt.decision_function(X_test))

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


for i,var in enumerate(BDTvars):
    print('Feature: ', var, '  importance: {:.3f}'.format(bdt.feature_importances_[i]))

#Plot results
def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=20):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (-0.1,0.1)
    #low_high = (low,high)

    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')

fig6 = plt.figure()
compare_train_test(bdt, X_train, y_train, X_test, y_test)
plt.show(block = False)
