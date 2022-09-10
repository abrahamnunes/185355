""" CODE FOR MANUALLY ADJUSTING PARAMETERS. USED TO IDENTIFY MIN/MAX BOUNDS FOR OPTIMIZER
* temporary file *
"""

import numpy as np
import pandas as pd
import matplotlib

# nicer font options:
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netpyne import specs, sim
from clamps import IClamp
from find_rheobase import ElectrophysiologicalPhenotype
from IVdata import IVdata
from CurvesFromData import extractIV, extractFI

netparams = specs.NetParams()

gc = netparams.importCellParams(
    label='GC',
    conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
    fileName="objects/GC.hoc",
    cellName="GranuleCell",
    cellArgs=[1],
    importSynMechs=False
)

rawhciv = pd.read_csv("rawdata/HC_NaK_long.csv")  # healthy control IVs


with open('gc.txt', 'w') as f:
    f.write(str(gc))


class testparam(object):
    def __init__(self,
                 cell):
        self.cell_dict = {"secs": cell["secs"]}
        self.manual_adjust()

    def curr_inj(self, current, delay=0, duration=1000):
        iclamp = IClamp(self.cell_dict, noise=False, delay=delay, duration=duration, T=duration + delay * 2)
        res = iclamp(current)
        return res

    def volt_inj(self):
        IV = IVdata(self.cell_dict)
        self.testclamp = IV.compute_ivdata(vlow=-70, vhigh=20, n_steps=10, delay=10, duration=5)
        return self.testclamp

    def sim_fi(self):
        ep = ElectrophysiologicalPhenotype(self.cell_dict, noise=False)
        self.simfi = ep.compute_fi_curve(ilow=0, ihigh=0.033, n_steps=12, delay=0, duration=1000)
        return self.simfi

    def data_iv(self):
        eiv = extractIV(rawhciv, 'CTRL')
        self.dataiv = eiv.averageIV()
        return self.dataiv

    def manual_adjust(self):
        baseline = self.sim_fi()
        baselineiv = self.volt_inj()

        #self.cell_dict['secs']['soma']['mechs']['bk']['gkbar'] = 0.0006

        # --- SODIUM
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gnatbar'] = 0.09 #0.12
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftma'] = 70 #43.0
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftmb'] = 20 #15.0
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftha'] = 100 #65.0  # of interest
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshifthb'] = 14.5 #12.5  # of interest
        self.cell_dict['secs']['soma']['ions']['na']['e'] = 60 #50

        # --- POTASSIUM

        self.cell_dict['secs']['soma']['mechs']['ichan2']['gkfbar'] = 0.02 #0.016
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gksbar'] =  0.006
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnfa'] = 45 #18.0
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnfb'] = 20 #43
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnsa'] = 40 #30
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnsb'] = 30 #55

        self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] = 0.012
        #self.cell_dict['secs']['soma']['mechs']['kir']['gkbar'] = 0.0
        self.cell_dict['secs']['soma']['mechs']['km']['gbar'] = 0.001
        self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] = 0.005
        self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] = 0.002
        self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] = 0.001
        self.cell_dict['secs']['soma']['mechs']['tca']['gcatbar'] = 3.7E-05
        shifted = self.sim_fi()
        shiftediv = self.volt_inj()
        aptest = self.curr_inj(0.03)

        dataiv = self.data_iv()

        plt.plot(baseline['I'], baseline['F'], label="baseline")
        plt.plot(shifted['I'], shifted['F'], label="manual fit")
        plt.legend()
        plt.savefig("figures/manual-adjust/ifvsif.jpeg")
        plt.close()

        plt.plot(baselineiv['V'], baselineiv['Na'], label='BL Na')
        plt.plot(baselineiv['V'], baselineiv['K'], label='BL K')
        plt.plot(dataiv[:, 0], dataiv[:, 1], label='D Na')
        plt.plot(dataiv[:, 0], (dataiv[:, 3] + dataiv[:, 5]), label='D K')
        plt.plot(shiftediv['V'], shiftediv['Na'], label='SH Na')
        plt.plot(shiftediv['V'], shiftediv['K'], label='SH K')
        plt.legend()
        plt.savefig("figures/manual-adjust/ivvsiv.jpeg")
        plt.close()

        plt.plot(aptest['t'], aptest['V'])
        plt.savefig("figures/manual-adjust/aptest.jpeg")



TestParam = testparam(gc)
