# SUMMARY PLOT FROM OPTIMIZATION PROCEDURE
# ----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CurvesFromData import extractFI, extractIV

## IMPORT RAW DATA 
rawhc = pd.read_csv("rawdata/HC_FI.csv")  # healthy control FIs
rawlr = pd.read_csv("rawdata/LR_FI.csv")  # lithium responder FIs
rawnr = pd.read_csv("rawdata/NR_FI.csv")  # lithium non-responder FIs
rawhciv = pd.read_csv("rawdata/HC_NaK_long.csv")  # healthy control IVs
rawlriv = pd.read_csv("rawdata/LR_NaK_long.csv")  # lithium responder IVs
rawnriv = pd.read_csv("rawdata/NR_NaK_long.csv")  # lithium non-responder IVs

HC_CTRL_FI = extractFI(rawhc, 'CTRL').averageFI()
HC_LITM_FI = extractFI(rawhc, 'LITM').averageFI()
LR_CTRL_FI = extractFI(rawlr, 'CTRL').averageFI()
LR_LITM_FI = extractFI(rawlr, 'LITM').averageFI()
NR_CTRL_FI = extractFI(rawnr, 'CTRL').averageFI()
NR_LITM_FI = extractFI(rawnr, 'LITM').averageFI()

HC_CTRL_IV = extractIV(rawhciv, 'CTRL').averageIV()
HC_LITM_IV = extractIV(rawhciv, 'LITM').averageIV()
LR_CTRL_IV = extractIV(rawlriv, 'CTRL').averageIV()
LR_LITM_IV = extractIV(rawlriv, 'LITM').averageIV()
NR_CTRL_IV = extractIV(rawnriv, 'CTRL').averageIV()
NR_LITM_IV = extractIV(rawnriv, 'LITM').averageIV()

## OPTIMIZED FI CURVES
simfi_hcc = pd.read_csv("data/parameters/simFIs_HC_CTRL.csv") 
simfi_hcl = pd.read_csv("data/parameters/simFIs_HC_LITM.csv") 
simfi_lrc = pd.read_csv("data/parameters/simFIs_LR_CTRL.csv") 
simfi_lrl = pd.read_csv("data/parameters/simFIs_LR_LITM.csv")
simfi_nrc = pd.read_csv("data/parameters/simFIs_NR_CTRL.csv") 
simfi_nrl = pd.read_csv("data/parameters/simFIs_NR_LITM.csv")
#simfi_nrl = simfi_nrc 


## OPTIMIZED IV CURVES 
simiv_hcc = pd.read_csv("data/parameters/simIVs_HC_CTRL.csv") 
simiv_hcl = pd.read_csv("data/parameters/simIVs_HC_LITM.csv") 
simiv_lrc = pd.read_csv("data/parameters/simIVs_LR_CTRL.csv") 
simiv_lrl = pd.read_csv("data/parameters/simIVs_LR_LITM.csv")
simiv_nrc = pd.read_csv("data/parameters/simIVs_NR_CTRL.csv") 
simiv_nrl = pd.read_csv("data/parameters/simIVs_NR_LITM.csv")
#simiv_nrl = simiv_nrc 

## PLOTTING 

## -------------------------------------------------------------------------------------------------------
## FI CURVES
## -------------------------------------------------------------------------------------------------------
currents = np.linspace(0, 33, 12)

fig1, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3, 3)

ax1.plot(currents, simfi_hcc["F"], color = '0.0', label = 'HCC-Opt.')
ax1.plot(currents, simfi_hcl["F"], color = 'crimson', label = 'HCL-Opt.')
ax1.errorbar(currents, HC_CTRL_FI[:, 1], yerr=HC_CTRL_FI[:, 2], color='0.7', ls = '--', label='HCC-data')
ax1.errorbar(currents, HC_LITM_FI[:, 1], yerr=HC_LITM_FI[:, 2], color='pink', ls = '--', label='HCL-data')
ax1.title.set_text('F-I curves')
ax1.set_xlabel("Current (pA)")
ax1.set_ylabel("Frequency (Hz)")

ax2.plot(currents, simfi_lrc["F"], color = '0.0', label = 'LRC-Opt.')
ax2.plot(currents, simfi_lrl["F"], color = 'crimson', label = 'LRL-Opt.')
ax2.errorbar(currents, LR_CTRL_FI[:, 1], yerr=LR_CTRL_FI[:, 2], color='0.7', ls = '--', label='LRC-data')
ax2.errorbar(currents, LR_LITM_FI[:, 1], yerr=LR_LITM_FI[:, 2], color='pink', ls = '--', label='LRL-data')
ax2.set_xlabel("Current (pA)")
ax2.set_ylabel("Frequency (Hz)")

ax3.plot(currents, simfi_nrc["F"], color = '0.0', label = 'NRC-Opt.')
ax3.plot(currents, simfi_nrl["F"], color = 'crimson', label = 'NRL-Opt.')
ax3.errorbar(currents, NR_CTRL_FI[:, 1], yerr=NR_CTRL_FI[:, 2], color='0.7', ls = '--', label='NRC-data')
ax3.errorbar(currents, NR_LITM_FI[:, 1], yerr=NR_LITM_FI[:, 2], color='pink', ls = '--', label='NRL-data')
ax3.set_xlabel("Current (pA)")
ax3.set_ylabel("Frequency (Hz)")


## -------------------------------------------------------------------------------------------------------
## IV CURVES: SODIUM 
## -------------------------------------------------------------------------------------------------------


ax4.plot(simiv_hcc["V"], simiv_hcc["Na"], color = '0.0', label = 'HCC-Opt.')
ax4.plot(simiv_hcl["V"], simiv_hcl["Na"], color = 'crimson', label = 'HCL-Opt.')
ax4.errorbar(HC_CTRL_IV[:,0], HC_CTRL_IV[:, 1], yerr=HC_CTRL_IV[:, 2], color='0.7', ls = '--', label='HCC-data')
ax4.errorbar(HC_LITM_IV[:,0], HC_LITM_IV[:, 1], yerr=HC_LITM_IV[:, 2], color='pink', ls = '--', label='HCL-data')
ax4.title.set_text('Sodium I-V curves')
ax4.set_xlabel("Voltage (mV)")
ax4.set_ylabel("Current (nA)")

ax5.plot(simiv_lrc["V"], simiv_lrc["Na"], color = '0.0', label = 'LRC-Opt.')
ax5.plot(simiv_lrl["V"], simiv_lrl["Na"], color = 'crimson', label = 'LRL-Opt.')
ax5.errorbar(LR_CTRL_IV[:,0], LR_CTRL_IV[:, 1], yerr=LR_CTRL_IV[:, 2], color='0.7', ls = '--', label='LRC-data')
ax5.errorbar(LR_LITM_IV[:,0], LR_LITM_IV[:, 1], yerr=LR_LITM_IV[:, 2], color='pink', ls = '--', label='LRL-data')
ax5.set_xlabel("Voltage (mV)")
ax5.set_ylabel("Current (nA)")

ax6.plot(simiv_nrc["V"], simiv_nrc["Na"], color = '0.0', label = 'NRC-Opt.')
ax6.plot(simiv_nrl["V"], simiv_nrl["Na"], color = 'crimson', label = 'NRL-Opt.')
ax6.errorbar(NR_CTRL_IV[:,0], NR_CTRL_IV[:, 1], yerr=NR_CTRL_IV[:, 2], color='0.7', ls = '--', label='NRC-data')
ax6.errorbar(NR_LITM_IV[:,0], NR_LITM_IV[:, 1], yerr=NR_LITM_IV[:, 2], color='pink', ls = '--', label='NRL-data')
ax6.set_xlabel("Voltage (mV)")
ax6.set_ylabel("Current (nA)")


## -------------------------------------------------------------------------------------------------------
## IV CURVES: POTASSIUM
## -------------------------------------------------------------------------------------------------------


ax7.plot(simiv_hcc["V"], simiv_hcc["K"], color = '0.0', label = 'HCC-Opt.')
ax7.plot(simiv_hcl["V"], simiv_hcl["K"], color = 'crimson', label = 'HCL-Opt.')
ax7.errorbar(HC_CTRL_IV[:,0], (HC_CTRL_IV[:, 3] + HC_CTRL_IV[:, 5]) , yerr= (HC_CTRL_IV[:, 4] + HC_CTRL_IV[:, 6]), color='0.7', ls = '--', label='HCC-data')
ax7.errorbar(HC_LITM_IV[:,0], (HC_LITM_IV[:, 3] + HC_LITM_IV[:, 5]) , yerr= (HC_LITM_IV[:, 4] + HC_LITM_IV[:, 6]), color='pink', ls = '--', label='HCL-data')
ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax7.title.set_text('Potassium I-V curves')
ax7.set_xlabel("Voltage (mV)")
ax7.set_ylabel("Current (nA)")

ax8.plot(simiv_lrc["V"], simiv_lrc["K"], color = '0.0', label = 'LRC-Opt.')
ax8.plot(simiv_lrl["V"], simiv_lrl["K"], color = 'crimson', label = 'LRL-Opt.')
ax8.errorbar(LR_CTRL_IV[:,0], (LR_CTRL_IV[:, 3] + LR_CTRL_IV[:, 5]), yerr=(LR_CTRL_IV[:, 4] + LR_CTRL_IV[:, 6]), color='0.7', ls = '--', label='LRC-data')
ax8.errorbar(LR_LITM_IV[:,0], (LR_LITM_IV[:, 3] + LR_LITM_IV[:, 5]), yerr=(LR_LITM_IV[:, 4] + LR_LITM_IV[:, 6]), color='pink', ls = '--', label='LRL-data')
ax8.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax8.set_xlabel("Voltage (mV)")
ax8.set_ylabel("Current (nA)")

ax9.plot(simiv_nrc["V"], simiv_nrc["K"], color = '0.0', label = 'NRC-Opt.')
ax9.plot(simiv_nrl["V"], simiv_nrl["K"], color = 'crimson', label = 'NRL-Opt.')
ax9.errorbar(NR_CTRL_IV[:,0], (NR_CTRL_IV[:, 3] + NR_CTRL_IV[:, 5]), yerr=(NR_CTRL_IV[:, 4] + NR_CTRL_IV[:, 6]), color='0.7', ls = '--', label='NRC-data')
ax9.errorbar(NR_LITM_IV[:,0], (NR_LITM_IV[:, 3] + NR_LITM_IV[:, 5]), yerr=(NR_LITM_IV[:, 4] + NR_LITM_IV[:, 6]), color='pink', ls = '--', label='NRL-data')
ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax9.set_xlabel("Voltage (mV)")
ax9.set_ylabel("Current (nA)")

fig1.set_figwidth(12)
fig1.set_figheight(10)
fig1.tight_layout()
fig1.savefig('figures/op-output/aggregate_plot.pdf', bbox_inches="tight")

print("Finished plotting summary figure.")