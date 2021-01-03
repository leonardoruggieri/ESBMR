#%% Script per provare se la classe funziona 
### con tutti i tipi di prior e covariate sin qui implementate.
import pandas as pd
import numpy as np 
import random
from numpy import matlib
from scipy.special import betaln
from scipy.special import gammaln
from scipy.stats import mode
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from hpf_vi import hpf_vi
import seaborn as sns
from esbmr import esbmr
from TrainTestSplit import train_val_split
from TrainTestSplit import train_val_split_zero

#%% 1. NO COVARIATES ------------------------------
#%% No covariates, DP-DP prior
Y = load_data("synth3")
mod = esbmr()
mod.fit(Y,100)
hpf = hpf_vi()
hpf.fit(Y,100)
#%% No covariates, DM-DM prior
mod1 = esbmr(prior_u = "DM", prior_i = "DM", beta = 0.5, components = 4)
mod1.fit(Y, 10)
#%% No covariates, PY-PY prior
mod2 = esbmr(prior_u = "PY", prior_i = "PY", sigma = 0.4)
mod2.fit(Y, 10)
#%% No covariates, GN-GN prior
mod2 = esbmr(prior_u = "GN", prior_i = "GN", gamma = 0.4)
mod2.fit(Y, 10)

#%% 2. INCLUSION OF COVARIATES ------------------------------
mod3 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
xu = np.random.randint(low = 0, high = Y.shape[0], size = Y.shape[0]) # Note: this must be proper data
xi = np.random.randint(low = 0, high = Y.shape[1], size = Y.shape[1])
mod3.fit(Y, 100, xu = xu, xi = xi, xu_type = "categ", xi_type = "categ", verbose = False)
#%%
Y = load_data("synth4")
mod5 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
xu1 = np.random.randint(low = 0, high = Y.shape[0], size = Y.shape[0]) # Note: this must be proper data
xu2 = np.random.randint(low = 0, high = Y.shape[0], size = Y.shape[0]) # Note: this must be proper data
xu3 = np.random.randint(low = 0, high = Y.shape[0], size = Y.shape[0]) # Note: this must be proper data
xu4 = np.random.randint(low = 0, high = Y.shape[0], size = Y.shape[0]) # Note: this must be proper data
xu = np.array(object = (xu1, xu2, xu3, xu4))
#xi = np.random.randint(low = 0, high = Y.shape[1], size = Y.shape[1])
mod5.fit(Y, 100, xu = xu, xi = None, xu_type = ["categ", "categ", "categ", "categ"], xi_type = None, verbose = False)
mod5.zu_labels
# Dubbio: come mai con due o più covariate per gli users sembra che ci sia più impatto delle covariate?
# Non dovrebbero sempre "pesare" lo stesso? (edit: no perché modifica il vlaore da moltipliacre/sommare) alla log-full conditional...)
# PROBELMA: con queste covariate random, i label degli user sono tutti diversi dopo 1000 iterazioni!
#%%
Y = load_data("synth4")
mod9 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
xu1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) # Note: this must be proper data
xu2 = xu1
xu3 = xu1
xu4 = xu1
xu = np.array(object = (xu1, xu2, xu3, xu4))
#xi = np.random.randint(low = 0, high = Y.shape[1], size = Y.shape[1])
mod9.fit(Y, 100, xu = xu, xi = None, xu_type = ["cont", "cont", "cont", "cont"], xi_type = None, verbose = False)
mod9.zu_labels
# BUONA NOTIZIA: covariate non informative non influenzano gli assignment.
#%%
Y = load_data("synth4")
mod10 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
xu = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7])
mod10.fit(Y, 100, xu = xu, xi = None, xu_type = "categ", xi_type = None, verbose = False)
# NOTA: le covariate influiscono meno rispetto alla adjacency matrix Y.
#%%
Y = load_data("synth4")
mod10 = esbmr(prior_u = "DM", prior_i = "DP", beta = 0.5, components = 7, sigma = 0.4)
xu1 = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7])
xu2 = xu1
xu3 = xu1
xu4 = xu1
xu = np.array(object = (xu1, xu2, xu3, xu4))
mod10.fit(Y, 100, xu = xu, xi = None, xu_type = ["categ", "categ", "categ", "categ"], xi_type = None, verbose = False)
# NOTA: le covariate influiscono meno rispetto alla adjacency matrix Y
# In presenza di molte covariate tutte uguali, che rinforzano l'informazione, i label sono più influenzati.
# Ad ogni modo, le stime di theta sono molto vicine per gli elementi che sarebbero stati clusterizzati insieme in assenza di covariate.
#%%
Y = load_data("synth4")
mod6 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
xu1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,10,0,0,0,0,0,0,0,0,0,0,0,0])
xu2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,10,0,0,0,0,0,0,0,0,0,0,0,0])
xu3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,1,1,1,1,1,1,1,1,1,1,1,1])
xu4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,1,1,1,1,1,1,1,1,1,1,1,1])
xu = np.array(object = (xu1, xu2, xu3, xu4))
mod6.fit(Y, 100, xu = xu, xi = None, xu_type = ["categ", "categ", "categ", "categ"], xi_type = None, verbose = False)
# RISULTATO users: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 
#                          1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.])
# Commento: I valori anomali della 15esima osservazione non permettono di clusterizzare i successivi users.
# Edit 25 Agosto: comportamento anomalo ovviabile in teoria prevedendo classi adiacenti (0,1,2,3,...) per categorical variables.
#%%
Y = load_data("synth4")
mod11 = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.5, components = 4, sigma = 0.4)
xu1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
xu2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
# xu3 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
xu4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
xu = np.array(object = (xu1, xu2, xu4))
mod11.fit(Y, 100, xu = xu, xi = None, xu_type = ["categ", "categ", "categ"], xi_type = None, verbose = False)
mod11.zu_labels
#%% WITH "PROPER" DATA
Y = load_data("synth4")
mod4 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
xu1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
xu2 = np.array([5,4,5,2,2,1,5,4,3,2,1,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
xu = np.array(object = (xu1, xu2))
#xi = np.random.randint(low = 0, high = Y.shape[1], size = Y.shape[1])

mod4.fit(Y, 100, xu = xu, xi = None, xu_type = ["categ", "categ"], xi_type = None, verbose = False)

#%% 3. INCLUSION OF MANY COUNT VARIABLES ------------------------------
#%%
Y = load_data("synth4")
mod12 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
mod13 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)

x1 = [1,2,1,1,1,1,1,1,1,2,10,10,10,10,10,10]
x2 = [1,2,1,1,1,1,1,1,1,2,10,10,10,10,10,10] # interpreted as count
xi_data = np.empty(shape= (2,Y.shape[1]))
xi_data[0] = x1
xi_data[1] = x2
#xi = xu3
mod12.fit(Y, 100, xi = xi_data, xu = None, xi_type = ["categ","count"], xu_type = None, verbose = False)
mod13.fit(Y, 100, xi = None, xu = None, xi_type = None, xu_type = None, verbose = False)
print(f"With cov: {mod12.zi_labels}. Without: {mod13.zi_labels}")
# OK, con una covariata count-type gira. Andranno fatte tutte le valutazioni fatte per categoriche.
# OK, gira anche con due count-type e con covariate di natura mista. Bene.
#%%
Y = load_data("synth4")
x1 = [1,2,1,1,1,1,1,1,1,2,10,10,10,10,10,10]
x2 = [1,2,1,1,1,1,1,1,1,2,10,10,10,10,10,10] # interpreted as count
xi_data = np.empty(shape= (2,Y.shape[1]))
xi_data[0] = x1
xi_data[1] = x2
mod14 = esbmr(prior_u = "DP", prior_i = "DP", beta = 0.5, components = 4, sigma = 0.4)
mod14.fit(Y, 100, xi = xi_data, xu = None, xi_type = ["categ","count"], xu_type = None, verbose = False)
mod14.beta_xi
#%%
Y = load_data("synth5")
mod14 = esbmr(prior_u = "DM", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.4, gamma = 0)
x1 = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0]
x2 = [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x3 = [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x4 = [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xi_data = np.empty(shape= (4,Y.shape[0]))
xi_data[0] = x1
xi_data[1] = x2
#xi_data[2] = x3
#xi_data[3] = x4
mod14.fit(Y, 100, xi = None, xu = np.array(x1), xu_type = "count", xi_type = None, verbose = False)
mod14.theta_est
#%% CONTINUOUS COVARIATES: -----------------------------------
Y = load_data("synth4")
mod14 = esbmr(prior_u = "DM", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.4, gamma = 0)
mod15 = esbmr(prior_u = "DM", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.4, gamma = 0)
mod16 = esbmr(prior_u = "DM", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.4, gamma = 0)
mod17 = esbmr(prior_u = "DM", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.4, gamma = 0)
x1 = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,1,1,1,1,1,1,0,0,0,0,0,0,0]
x5 = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,1,1,1,1,1,1,0,0,0,0,0,0,0]
x6 = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,1,1,1,1,1,1,0,0,0,0,0,0,0]
x7 = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,1,1,1,1,1,1,0,0,0,0,0,0,0]
x = np.array((x1,x5,x6,x7))
x2 = np.ones(shape = Y.shape[0])*10
x3 = [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x4 = [10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xi_data = np.empty(shape= (4,Y.shape[0]))
xi_data[0] = x1
xi_data[1] = x2
#xi_data[2] = x3
#xi_data[3] = x4
mod14.fit(Y, 100, xi = None, xu = np.array(x1), xu_type = "categ", xi_type = None, verbose = False)
mod15.fit(Y, 100, xi = None, xu = np.array(x1), xu_type = "count", xi_type = None, verbose = False)
mod16.fit(Y, 100, xi = None, xu = np.array(x1), xu_type = "cont", xi_type = None, verbose = False)
mod17.fit(Y, 100, xi = None, xu = None, xu_type = None, xi_type = None, verbose = False)
mod14.theta_est
# Comment: the higher likelihood is achieved (by chance? Hope not) with the correct specification of covariate type (count in this case).
#%% MANY CONTINUOUS COVARIATES:
Y = load_data("synth5")
x1 = [100,100,100,100,100,100,100,100,100,100,100,100,100,100]
x2 = [100,100,100,100,100,100,100,100,100,100,100,100,100,100]
x = np.empty(shape= (2,Y.shape[0]))
x[0] = x1
x[1] = x2
mod1 = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.01, gamma = 0)
mod1.fit(Y, 100, xu = None, xi = x, xi_type = ["cont","cont"], xu_type = None, verbose = False)
mod1.theta_est
#%% LIKELIHOOD COMPARISON: WITH AND WITHOUTH COVARIATES
Y = load_data("synth4")
x1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
x2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
x = np.empty(shape= (2,Y.shape[0]))
x[0] = x1
x[1] = x2

modno = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.01, gamma = 0)
modno.fit(Y, 1000, xi = None, xu = None, xu_type = None, xi_type = None, verbose = False)

modcateg = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.01, gamma = 0)
modcateg.fit(Y, 1000, xi = None, xu = x, xu_type = ["categ","categ"], xi_type = None, verbose = False)

modcount = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.01, gamma = 0)
modcount.fit(Y, 1000, xi = None, xu = x, xu_type = ["count","count"], xi_type = None, verbose = False)

modcont = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.01, gamma = 0)
modcont.fit(Y, 1000, xi = None, xu = x, xu_type = ["cont","cont"], xi_type = None, verbose = False)


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,8))
ax.plot(modno.ll[1:][::10], label = "No covariates")
ax.plot(modcateg.ll[1:][::10], label = "Categorical")
ax.plot(modcount.ll[1:][::10], label = "Count-type")
ax.plot(modcont.ll[1:][::10], label = "Countinuous")
ax.legend();

print(np.array(modno.ll[1:]).mean()) # 271.45
print(np.array(modcateg.ll[1:]).mean()) # 270.96
print(np.array(modcount.ll[1:]).mean()) # 271.67
print(np.array(modcont.ll[1:]).mean()) # 271.35

print(np.array(modno.ll[2:]).var()) # 10.04
print(np.array(modcateg.ll[2:]).var()) # 13.16
print(np.array(modcount.ll[2:]).var()) # 8.6
print(np.array(modcont.ll[2:]).var()) # 9.7

#%%
Y = load_data("synth4")
x1 = [100,0,100,0,100,0,100,0,100,0,100,0,100,0,50,50]
x2 = [100,0,100,0,100,0,100,0,100,0,100,0,100,0,50,50]

x3 = [100,100,100,100,100,100,100,100,100,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x4 = [100,0,100,0,100,0,100,0,100,0,100,0,100,0,50,50,100,0,100,0,100,0,100,0,100,0,100]
xi = np.empty(shape= (2,Y.shape[1]))
xu = np.empty(shape= (2,Y.shape[0]))
xi[0] = x1
xi[1] = x2

xu[0] = x3
xu[1] = x4
mod1 = esbmr(prior_u = "PY", prior_i = "PY", beta = 0.1, components = 2, sigma = 0.4, gamma = 0)
mod1.fit(Y, 200, xu = None, xi = xi, xu_type = None, xi_type = ["cont","cont"], verbose = False)
print(mod1.theta_est)
print(mod1.zi_labels)
#%%
import pandas as pd
import numpy as np 
import random
from numpy import matlib
from scipy.special import betaln
from scipy.special import gammaln
from scipy.stats import mode
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from hpf_vi import hpf_vi
import seaborn as sns
from esbmr import esbmr

def load_data(which_one):
    if which_one == "ratings":
        Y = pd.read_csv("/Users/leonardo/Documents/Bocconi ESS/5. Tesi/Codice/Data/ratings.csv", header = None).values[:,1:]
    elif which_one == "sim":
        Y = np.array([[1,1,0,0,0],[0,1,0,0,0],[1,0,0,1,0],[0,0,0,1,1],[0,1,0,1,1],[0,0,0,1,0]])
    elif which_one == "synth":
        type1 = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        type2 = np.array([0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0])
        type3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1])
        type4 = np.array([0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
        Y = np.array([type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type2, type2, type2, type2, type2, type2, type3, type3, type3, type3, type3, type3, type3, type3, type4, type4, type4, type4, type4, type4])
    elif which_one == "synth2":
        type1 = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
        type2 = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
        Y = np.array([type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2])
    elif which_one == "synth3":
        edges1 = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
        edges2 = np.array([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0])
        edges3 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0])
        edges4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1])
        Y = np.array([edges1, edges1, edges1, edges1, edges2, edges2, edges2, edges2, edges3, edges3, edges3, edges3, edges4, edges4, edges4, edges4])
    elif which_one == "synth4":
        type1 = np.array([5,5,5,5,2,1,0,0,0,0,0,0,0,0,0,0])
        type2 = np.array([0,0,0,0,0,0,5,5,2,3,1,4,4,5,5,1])
        Y = np.array([type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2])
    elif which_one == "synth5":
        type1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        type2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        Y = np.array([type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type1, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2, type2])


    return Y
def train_val_split(data, valid_dim=0.2):
        '''
        Creating two additional objects, i.e. training and validation set, which can be used in the fitting process

        Parameters:
        data = np.array
        valid_dim = float
        '''
        if valid_dim >= 1:
          raise ValueError("valid_dim must be lower than 1")

        train = data.copy()
        valid = np.zeros(data.shape)

        for u in np.unique(data.nonzero()[0]):
            ind = data[u].nonzero()[0] 

            if len(ind) > 0: 
                valid_ind = np.random.choice(ind, round(len(ind)*valid_dim), replace=False)
                for i in valid_ind:
                    valid[u,i], train[u,i] = data[u,i], 0
        return train, valid

Y = load_data("ratings")
train, test = train_val_split(Y, valid_dim = 0.2)
#%%
mods = esbmr(prior_u = "GN", prior_i = "GN", components = 10, beta = 0.1, gamma = 0.2)
mods.fit(train, 100, mserror = True, valid = test) # testing if label switching occurs
#%%
mods.co_clustering_matrices()
sns.heatmap(mods.ccmatrix_u, cmap="YlGnBu")
#%%
sns.heatmap(mods.ccmatrix_i, cmap="YlGnBu")
#%%
y_pred = mods.predict(test)
mods.block_recommend(range(0,mods.Y.shape[0]), 2)
#%%
mods.precision_at_top_cluster(test, y_pred, threshold = 0.5)
#%%
mods.roc_curve(test,y_pred)

#%% Co-clustering and true clusters comparisons
zu_true = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]) # true labels for synth3
zi_true = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]) # true labels for synth3
Y_user = np.zeros(shape = (zu_true.shape[0],zu_true.shape[0]))
for i in range(Y_user.shape[0]):
    for j in range(Y_user.shape[1]):
        if zu_true[i] == zu_true[j]:
            Y_user[i,j] = 1
Y_item = np.zeros(shape = (zi_true.shape[0],zi_true.shape[0]))
for i in range(Y_item.shape[0]):
    for j in range(Y_item.shape[1]):
        if zi_true[i] == zi_true[j]:
            Y_item[i,j] = 1

#%% FINAL PLOT FOR CO-CLUSTERING ANALYSIS (EXPLICIT FEEDBACK):
fig, ax = plt.subplots(figsize=(20,10), nrows = 2, ncols = 2)
sns.heatmap(Y_user, cmap="YlGnBu", ax=ax[0,0])
sns.heatmap(modelx.ccmatrix_u, cmap="YlGnBu", ax=ax[0,1])
sns.heatmap(Y_item, cmap="YlGnBu", ax=ax[1,0])
sns.heatmap(modelx.ccmatrix_i, cmap="YlGnBu", ax=ax[1,1])
# %%
y_predicted = model.predict(test)
np.round(model.y_predicted)
#%%
model.mse(train,y_predicted, valid = test)
# %%
from hpf_vi import hpf_vi
# %%
hpf = hpf_vi()
# %%
hpf.fit(train, 100)
# %%
np.round(hpf.predicted)
# %%
hpf.mse_train
# %%
model = esbmr()
model.fit(train, 100)
# %%
model.predict(test)
# %%
model.mse(Y,model.y_predicted)
#%%
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
#%%
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

# %%
plt.plot(fpr,tpr)
plt.show() 

# This is the AUC
auc = np.trapz(tpr,fpr)



# %%
