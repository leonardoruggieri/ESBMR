#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from esbmr import esbmr
from hpf_vi import hpf_vi
#%%
def load_data(which_one):
    if which_one == "sim":
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
# Y = load_data("synth4")
#%% LOAD DATA
#df = pd.read_csv("training_new.csv") # filtro carta e merchant - Gen, Feb, Set 2020
#df = df[df.nm_nome_acc_pulito != 'nan']
df = pd.read_csv("interazioni_sbt_2020_attivi.csv") 
#%% Ottenimento valori binnati per il campo "conteggio" del DataFrame df:
df['bins'] = pd.qcut(df['conteggio'], q = [0, .75, .80, .89, .95, 1],labels=False,
                            precision=0)
df['bins'] = df['bins'] + 1
# Nota: i bin vanno scelti di volta in volta in modo che non si sovrappongano. I valori del parametro q vano bene per questo caso specifico (interazioni_sbt_2020_attivi).

#%% Dataframe "inter" sarà riferimento per recuperare al volo i merchant raccomandati alle carte.
inter = pd.crosstab(df['id_car'],df['nm_nome_cleaned'],df['bins'],aggfunc='first').fillna(0)
#%% Numpy array per explicit feedback datasets
inter_np = np.array(inter) 

#%%
df = np.array(df).astype(str)
#%%
def build_adjacency_matrix(df):
    '''
    It builds a count-type adjacency matrix starting from single user/item interactions
    
    ---------
    Parameter:
    df : numpy.array (2D)
    '''
    y = np.zeros(shape = (np.unique(df[:,0]).shape[0], np.unique(df[:,1]).shape[0]))
    rows, cols = df.shape
    for u in range(np.unique(df[:,0]).shape[0]):
        for i in range(np.unique(df[:,1]).shape[0]):
            if df[u,1] == np.unique(df[:,1])[i]:
                y[u,i] = 1
        print(f"User {u} completed.")
    return y

def f2(df):
    row_name = np.unique(df[:,0], return_index = True)
    col_name = np.unique(df[:,1], return_index = True)
    y = np.zeros(shape = (np.unique(df[:,0]).shape[0], np.unique(df[:,1]).shape[0]))
    user_count = 0
    global row_index
    row_index = 0
    col_index = 0
    y[row_index, col_index] = 1
    df_iniziale = df[0,0]
    for i in range(df.shape[0]):
        if df[i,0] == df[i-1,0] and df[i,1] != df[i-1,1]:
            y[row_index, np.argwhere(col_name[0][:] == df[i,1])] = 1
        if df[i,0] != df[i-1,0]:
            row_index += 1
            y[row_index, np.argwhere(col_name[0][:] == df[i,1])] = 1
        # print(f"Row {i} completed.")
    return y

def f3(df):
    row_name = np.unique(df[:,0], return_index = False)
    col_name = np.unique(df[:,1], return_index = False)
    y = np.zeros(shape = (np.unique(df[:,0]).shape[0], np.unique(df[:,1]).shape[0]))
    for it in range(df.shape[0]):
            y[np.argwhere(row_name == df[it,0])[0][0],np.argwhere(col_name == df[it,1])[0][0]] = 1
    print(f"Until row {it}")
    return y
#%%
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
                valid_ind = np.random.choice(a = ind, size = round(len(ind)*valid_dim), replace=False)
                for i in valid_ind:
                    valid[u,i], train[u,i] = data[u,i], 0
        return train, valid

#%%
#df = df[:,[0,13]]
y = f3(df)
#%%
#train, test = train_val_split(y, valid_dim = 0.3)
#print(y.sum(), train.sum(), test.sum()) # for implicit feedback data only!
#%%
#print("Sparsity degree: ", y.sum() / (y.shape[0] * y.shape[1]))
# %%
mod = esbmr(prior_u = "PY", sigma = 0.8)

# %%
inter_np = inter_np.astype(int)
mod.fit(inter_np,5)
#%%
y_pred = mod.predict(test)
# %%
mod.mse(test,y_pred)
# %%
from hpf_vi import hpf_vi
# %%
hpf = hpf_vi()
hpf.fit(y,100)
# %%
