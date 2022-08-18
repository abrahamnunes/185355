import pandas as pd
import numpy as np

HC_CTRL = pd.read_csv("data/parameters/parameters_HC_CTRL_MSE.csv")
HC_LITM = pd.read_csv("data/parameters/parameters_HC_LITM_MSE.csv")
LR_CTRL = pd.read_csv("data/parameters/parameters_LR_CTRL_MSE.csv")
LR_LITM = pd.read_csv("data/parameters/parameters_LR_LITM_MSE.csv")
NR_CTRL = pd.read_csv("data/parameters/parameters_NR_CTRL_MSE.csv")
NR_LITM = pd.read_csv("data/parameters/parameters_NR_LITM_MSE.csv")

HCC_FI = pd.read_csv("data/parameters/simFIs_HC_CTRL_MSE.csv")
HCL_FI = pd.read_csv("data/parameters/simFIs_HC_LITM_MSE.csv")
LRC_FI = pd.read_csv("data/parameters/simFIs_LR_CTRL_MSE.csv")
LRL_FI = pd.read_csv("data/parameters/simFIs_LR_LITM_MSE.csv")
NRC_FI = pd.read_csv("data/parameters/simFIs_NR_CTRL_MSE.csv")
NRL_FI = pd.read_csv("data/parameters/simFIs_NR_LITM_MSE.csv")

# parameters to be optimized
free_params = {
    'bk': ['gkbar'],  # big conductance, calcium-activated potassium channel
    'ichan2': ['gnatbar', 'vshiftma', 'vshiftmb', 'vshiftha', 'vshifthb', 'vshiftnfa', 'vshiftnfb', 'vshiftnsa',
               'vshiftnsb',
               'gkfbar', 'gksbar', 'gl'],  # KDR channel conductances, sodium conductance
    'ka': ['gkabar'],  # A-type (fast inactivating) Kv channel
    'kir': ['gkbar'],  # inward rectifier potassium (Kir) channel
    'km': ['gbar'],  # KM channel
    'lca': ['glcabar'],  # l-type calcium
    'nca': ['gncabar'],  # n-type calcium
    'sk': ['gskbar'],  # small conductance potassium channel
    'tca': ['gcatbar']  # t-type calcium
}


dfs = [HC_CTRL, HC_LITM, LR_CTRL, LR_LITM, NR_CTRL, NR_LITM]

dfnames = ["HC_CTRL", "HC_LITM", "LR_CTRL", "LR_LITM", "NR_CTRL", "NR_LITM"]

FIs = [HCC_FI, HCL_FI, LRC_FI, LRL_FI, NRC_FI, NRL_FI]
FInames = ["HCC_FI", "HCL_FI", "LRC_FI", "LRL_FI", "NRC_FI", "NRL_FI"]

cellids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def insert_foldchange(df, dfname, free_params, cellids):
    mod = []
    for key, value in free_params.items():
        for i in range(0, len(value)):
            mod.append(key)
    for i, j in zip(cellids, range(4, 24, 2)):
        fc = df['Cell_%s' % cellids[i]] / df['baseline']
        df.insert(loc=j, column="fc_%s" % cellids[i], value=fc)
    df.insert(loc=1, column="mod", value=mod)
    df.to_csv("data/parameters/paramFC_%s.csv" % dfname)


for df, dfname in zip(dfs, dfnames):
    insert_foldchange(df, dfname, free_params, cellids)


def restructureFI(FI, FIname, cellids):
    currs = FI['I']
    freqs = FI['F']
    freqs = freqs.to_numpy()
    restructuredFI = pd.DataFrame({"I": currs[0:12]})
    for i, j in zip(cellids, range(0, 120, 12)):
        singlecell = freqs[0 + j:12 + j]
        restructuredFI.insert(loc=i + 1, column="Cell_%s" % cellids[i], value=singlecell)
    restructuredFI.to_csv("data/parameters/restruct_FIs_%s.csv" % FIname)


for FI, FIname in zip(FIs, FInames):
    restructureFI(FI, FIname, cellids)
