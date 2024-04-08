import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from netpyne import specs, sim 
from clamps import IClamp 
from find_rheobase import ElectrophysiologicalPhenotype
from IVdata import IVdata 
from CurvesFromData import extractFI, extractIV
from hocfromstr import optimizedhoc 

# SET SEED, INDICATE PERCENT NOISE TO ADD 
np.random.seed(123) #123
percentage = 0.02

# IMPORT GC DICT, INDICATE PARAMS OF INTEREST
netparams = specs.NetParams()

gc =  netparams.importCellParams(
        label='GC',
        conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
        fileName="objects/GC.hoc",
        cellName="GranuleCell",
        cellArgs=[1],
        importSynMechs=False
    )

cell_dict = {"secs": gc["secs"]}

free_params = {
    'bk': ['gkbar'],  # big conductance, calcium-activated potassium channel
    'ichan2': ['gnatbar', 'vshiftma', 'vshiftmb', 'vshiftha', 'vshifthb', 'vshiftnfa', 'vshiftnfb', 'vshiftnsa',
               'vshiftnsb',
               'gkfbar', 'gksbar', 'gl'],  # sodium, potassium parameters
    'lca': ['glcabar'],  # l-type calcium
    'nca': ['gncabar'],  # n-type calcium
    'sk': ['gskbar'],  # small conductance potassium channel
    'tca': ['gcatbar']  # t-type calcium

}

# DATA FOR PLOTTING PURPOSES
rawnriv = pd.read_csv("rawdata/NR_NaK_long.csv")  # lithium non-responder IVs
rawnr = pd.read_csv("rawdata/NR_FI.csv")  # lithium non-responder FIs
NR_LITM_FI = extractFI(rawnr, 'LITM').averageFI()
NR_LITM_IV = extractIV(rawnriv, 'LITM').averageIV()
NR_CTRL_FI = extractFI(rawnr, 'CTRL').averageFI()
NR_CTRL_IV = extractIV(rawnriv, 'CTRL').averageIV()

simfi_nrc = pd.read_csv("data/parameters/simFIs_NR_CTRL.csv") 
simiv_nrc = pd.read_csv("data/parameters/simIVs_NR_CTRL.csv") 
# IMPORT PARAMS FROM NR-CTRL
nrctrl = pd.read_csv("data/parameters/parameters_NR_CTRL.csv")
fittedparams = nrctrl['Cell_0']

# ADD NOISE
def addnoise(percentage, rawvec):
    sigma = [percentage * val for val in rawvec]
    noise = [np.random.normal(val, sig) for val, sig in zip(rawvec, sigma)]
    return noise

noiseyparam = addnoise(percentage, fittedparams)

nrctrl['litm'] = noiseyparam
nrctrl.to_csv('data/parameters/parameters_NR_LITMnoise.csv')

# SAVE NOISEY PARAMS TO HOC.
optimizedhoc(noiseyparam, "NR_LITM")


#--------
# FOR PLOTTING 

dend1 = ["gcdend1_0", "gcdend1_1", "gcdend1_2", "gcdend1_3"]
dend2 = ["gcdend2_0", "gcdend2_1", "gcdend2_2", "gcdend2_3"]
sections = ["soma"] + dend1

j = 0
for section in sections:
    for key in free_params.keys():
        for val in free_params[key]:
            cell_dict['secs'][section]['mechs'][key][val] = noiseyparam[j]
            j = j + 1           
for i in range(0,4):
    for k in free_params.keys():
        for v in free_params[k]:
            cell_dict['secs'][dend2[i]]['mechs'][k][v] = cell_dict['secs'][dend1[i]]['mechs'][k][v]

#testing fits 
epgc = ElectrophysiologicalPhenotype(cell_dict, noise=False)
iv = IVdata(cell_dict)  # instantiate class
nrlFI = epgc.compute_fi_curve(ilow=0, ihigh=0.033, n_steps=12, delay=0, duration=1500)
nrlIV = iv.compute_ivdata(vlow=-70, vhigh=20, n_steps=10, delay=10, duration=5)

nrlFI.to_csv('data/parameters/simFIs_NR_LITMnoise.csv')
nrlIV.to_csv('data/parameters/simIVs_NR_LITMnoise.csv')

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)

currents = np.linspace(0, 33, 12)
ax1.plot(currents, simfi_nrc["F"], color = '0.0', label = 'NRC-Opt.')
ax1.plot(currents, nrlFI["F"], color = 'crimson', label = 'NRC+Noise.')
ax1.errorbar(currents, NR_CTRL_FI[:, 1], yerr=NR_CTRL_FI[:, 2], color='0.7', ls = '--', label='NRC-data')
ax1.errorbar(currents, NR_LITM_FI[:, 1], yerr=NR_LITM_FI[:, 2], color='pink', ls = '--', label='NRL-data')
ax1.title.set_text('F-I curves')
ax1.set_xlabel("Current (pA)")
ax1.set_ylabel("Frequency (Hz)")

ax2.plot(simiv_nrc["V"], simiv_nrc["Na"], color = '0.0', label = 'NRC-Opt.')
ax2.plot(nrlIV["V"], nrlIV["Na"], color = 'crimson', label = 'NRC+Noise.')
ax2.errorbar(NR_CTRL_IV[:,0], NR_CTRL_IV[:, 1], yerr=NR_CTRL_IV[:, 2], color='0.7', ls = '--', label='NRC-data')
ax2.errorbar(NR_LITM_IV[:,0], NR_LITM_IV[:, 1], yerr=NR_LITM_IV[:, 2], color='pink', ls = '--', label='NRL-data')
ax2.title.set_text('Sodium I-V curves')
ax2.set_xlabel("Voltage (mV)")
ax2.set_ylabel("Current (nA)")

ax3.plot(simiv_nrc["V"], simiv_nrc["K"], color = '0.0', label = 'NRC-Opt.')
ax3.plot(nrlIV["V"], nrlIV["K"], color = 'crimson', label = 'NRC+Noise.')
ax3.errorbar(NR_CTRL_IV[:,0], (NR_CTRL_IV[:, 3] + NR_CTRL_IV[:, 5]), yerr=(NR_CTRL_IV[:, 4] + NR_CTRL_IV[:, 6]), color='0.7', ls = '--', label='NRC-data')
ax3.errorbar(NR_LITM_IV[:,0], (NR_LITM_IV[:, 3] + NR_LITM_IV[:, 5]) , yerr= (NR_LITM_IV[:, 4] + NR_LITM_IV[:, 6]), color='pink', ls = '--', label='NRL-data')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.title.set_text('Potassium I-V curves')
ax3.set_xlabel("Voltage (mV)")
ax3.set_ylabel("Current (nA)")

fig1.set_figwidth(12)
fig1.set_figheight(4)
fig1.tight_layout()
fig1.savefig('figures/op-output/nrlnoise_plot.pdf', bbox_inches="tight")
