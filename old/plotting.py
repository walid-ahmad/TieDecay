"""\
Scripts for plotting results

"""

import matplotlib.pyplot as plt
%matplotlib

import numpy as np
import storage

import pandas as pd
import datetime as dt
import matplotlib.dates as mdates

# load dates from fast_process script
dates = sampling_range.strftime('%m/%d/%Y')
dateLabels = dates[range(0,1000,100)]
x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]

TD1 = storage.load_obj('/Users/walid/Dropbox/Tie_Decay_Centrality/Data/TD_PRs_alpha_1')
TD24 = storage.load_obj('/Users/walid/Dropbox/Tie_Decay_Centrality/Data/TD_PRs_alpha_24')
TD168 = storage.load_obj('/Users/walid/Dropbox/Tie_Decay_Centrality/Data/TD_PRs_alpha_168')
TD_Agg = storage.load_obj('/Users/walid/Dropbox/Tie_Decay_Centrality/Data/TD_PRs_nodecay')

usersPath = '/Users/walid/Dropbox/Tie_Decay_Centrality/Data/NHSusersDict'
users = storage.load_obj(usersPath)

inds1 = [np.argmax(TD1[:, t]) for t in range(1000)]
topusers1 = np.unique(inds1)
inds24 = [np.argmax(TD24[:, t]) for t in range(1000)]
topusers24 = np.unique(inds24)
inds168 = [np.argmax(TD168[:,t]) for t in range(1000)]
topusers168 = np.unique(inds168)
indsAgg = [np.argmax(TD_Agg[:,t]) for t in range(1000)]
topusersAgg = np.unique(TD_Agg)

topusers = np.hstack([topusers1, topusers24, topusers168])
topusers = np.unique(topusers)

basecm = plt.cm.get_cmap('hsv')
clist = basecm(np.linspace(0,1,len(topusers)))
legDict = {tu: carr for tu, carr in zip(topusers, clist)}

## get top-user changes ##
def get_tu_changes(tuList):
    tuc = []
    for i in range(0,len(tuList)-1):
        ind1 = tuList[i]
        ind2 = tuList[i+1]
        if ind2 != ind1:
            tuc.append(i+1)
    return tuc

tuc1 = get_tu_changes(inds1)
tuc24 = get_tu_changes(inds24)
tuc168 = get_tu_changes(inds168)
tucAgg = get_tu_changes(indsAgg)

### Plot top user changes as gray spans
plt.figure(figsize=(20.16, 10.765))
ax1 = plt.subplot(311)
for j in range(0,len(tuc1)-1, 2):
    ax1.axvspan(tuc1[j], tuc1[j+1]-1, alpha=0.2, color='gray')
if len(tuc1)%2:
    ax1.axvspan(tuc1[-1], 1000, alpha=0.2, color='gray')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(312)
for j in range(0,len(tuc24)-1, 2):
    ax2.axvspan(tuc24[j], tuc24[j+1]-1, alpha=0.2, color='gray')
if len(tuc24)%2:
    ax2.axvspan(tuc24[-1], 1000, alpha=0.2, color='gray')
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(313)
for j in range(0,len(tuc168)-1, 2):
    ax3.axvspan(tuc168[j], tuc168[j+1]-1, alpha=0.2, color='gray')
if len(tuc168)%2:
    ax3.axvspan(tuc168[-1], 1000, alpha=0.2, color='gray')

### Plot top user PageRank scores

for i in range(1000):
    tu1 = inds1[i]
    tu24 = inds24[i]
    tu168 = inds168[i]
    ax1.scatter(i, TD1[tu1, i], marker='x', s=10, c=legDict[tu1])
    ax2.scatter(i, TD24[tu24, i], marker='x', s=10, c=legDict[tu24])
    ax3.scatter(i, TD168[tu168, i], marker='x', s=10, c=legDict[tu168])

ax3.set_xticks(range(0,1000,100))
ax3.set_xticklabels(dateLabels)
plt.subplots_adjust(hspace=.0)


### Plot legend
fig = plt.figure()
ax = plt.gca()

counter = 0
for k,v in legDict.items():
    plt.plot(counter, 1, label=usersDict[k], c=legDict[k])

figLegend = plt.figure()
plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',fontsize='x-small', ncol=2)


##### Ranking switch plot #####
"""
Top 10 users from fully aggregated network (across all time steps):
array([5276, 3244, 1425,   88, 2909, 4417, 4571, 7935, 2161, 4771])

"""

#~~~~ Aggregated ~~~~#
# get ranks
ranks = np.zeros(shape=TD_Agg.shape, dtype='uint8')
for t in range(1000):
    TD_t = TD_Agg[:,t]
    temp = TD_t.argsort()[::-1]
    ranks[temp, t] = np.arange(len(TD_t))

# increment by 1 to make rank 0 into rank 1, etc.
ranks += 1

selectedUsers = TD_Agg[:,-1].argsort()[-5:]

for u in selectedUsers:
    plt.plot(range(1000), ranks[u, :], c=legDict[u], label=usersDict[u])
plt.ylim([-0.5, 30])
plt.legend()
ax = plt.gca()
ax.set_xticks(range(0,1000,100))
ax.set_xticklabels(dateLabels)

#~~~~ various alphas ~~~~#
ranks1 = np.zeros(shape=TD1.shape, dtype='uint8')
ranks24 = np.zeros(shape=TD24.shape, dtype='uint8')
ranks168 = np.zeros(shape=TD168.shape, dtype='uint8')

for t in range(1000):
    TD_t1 = TD1[:,t]
    TD_t24 = TD24[:,t]
    TD_t168 = TD168[:,t]
    temp1 = TD_t1.argsort()[::-1]
    temp24 = TD_t24.argsort()[::-1]
    temp168 = TD_t168.argsort()[::-1]
    ranks1[temp1, t] = np.arange(len(TD_t1))
    ranks24[temp24, t] = np.arange(len(TD_t24))
    ranks168[temp168, t] = np.arange(len(TD_t168))

ranks1+=1
ranks24+=1
ranks168+=1

ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

for u in selectedUsers:
    ax1.plot(range(1000), ranks1[u,:], c=legDict[u],label=usersDict[u])
    ax2.plot(range(1000), ranks24[u,:], c=legDict[u],label=usersDict[u])
    ax3.plot(range(1000), ranks168[u,:], c=legDict[u],label=usersDict[u])

ax3.set_xticks(range(0,1000,100))
ax3.set_xticklabels(dateLabels)
plt.subplots_adjust(hspace=.0)
ax3.legend(loc=1)

#~~~~~ iterations/twitter activity ~~~~~#
PR_iterst = np.load('PR_iterst.npy')
PR_itersu = np.load('PR_itersu.npy')

# load up dataAdjDict from somewhere
t1 = pd.to_datetime('2012-03-18 08:00:00')
t2 = t1 + pd.Timedelta('4H')

drange = pd.date_range(t1, t2, freq='s')[0:-1]
activityVector = []
totalActivity = 0
dADkeys = dataAdjDict.keys()
for d in tqdm(drange):
    try:
        v = dataAdjDict[d]
        totalActivity+=1
    except:
        pass
    activityVector.append(totalActivity)


PR_iterst_cum = np.cumsum(PR_iterst)
PR_itersu_cum = np.cumsum(PR_itersu)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(range(len(drange)), PR_iterst_cum, c='green', label='Previous TDPR vector')
ax1.plot(range(len(drange)), PR_itersu_cum, c='blue', label='Uniform vector')
ax2.plot(range(len(drange)), activityVector,c='orange', label='Aggregate interactions')
