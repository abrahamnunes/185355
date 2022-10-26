# MANUALLY ADJUST MOSSY CELL PARAMS
# TEMPORARY FILE, USED TO IDENTIFY MIN/MAX BOUNDS FOR OPTIMIZER
# -----------------------------

import matplotlib
import numpy as np

# nicer font options:
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

matplotlib.use('Agg')  # hopefully this works over ssh
import matplotlib.pyplot as plt
import pandas as pd
import pylab
from random import Random  # TODO replace with numpy rand f'n.  pseudorandom number generation
from inspyred import ec  # evolutionary algorithm
from netpyne import specs, sim  # neural network design and simulation
from clamps import IClamp
from IVdata import IVdata
from CurvesFromData import extractIV, extractFI
# from clamps_noise import ICNoise
from find_rheobase import ElectrophysiologicalPhenotype
from scipy.signal import find_peaks
from tabulate import tabulate
# from FI_fromdata import extractFI
import similaritymeasures
import random

netparams = specs.NetParams()

mc = netparams.importCellParams(
    label='MC',
    conds={"cellType": "MossyCell", "cellModel": "MossyCell"},
    fileName="objects/MC.hoc",
    cellName="MossyCell",
    #cellArgs=[1],
    importSynMechs=False
)

free_params = {
    'bk': ['gkbar'],  # big conductance, calcium-activated potassium channel
    'ichan2': ['gnatbar', 'vshiftma', 'vshiftmb', 'vshiftha', 'vshifthb', 'vshiftnfa', 'vshiftnfb', 'vshiftnsa',
               'vshiftnsb',
               'gkfbar', 'gksbar', 'gl'],  # sodium, potassium parameters
    'ka': ['gkabar'],  # A-type (fast inactivating) Kv channel
    'lca': ['glcabar'],  # l-type calcium
    'nca': ['gncabar'],  # n-type calcium
    'sk': ['gskbar'],  # small conductance potassium channel
    'ih': ['ghyfbar', 'ghysbar']  # HCN channel
}

with open('figures/mossycell/mc.txt', 'w') as f:
    f.write(str(mc))


class testparam(object):
    def __init__(self,
                 cell,
                 free_params):
        self.cell_dict = {"secs": cell["secs"]}
        self.free_params = free_params

    def curr_inj(self, current, delay=0, duration=1000):
        iclamp = IClamp(self.cell_dict, delay=delay, duration=duration, T=duration + delay * 2)
        res = iclamp(current)
        return res

    def volt_inj_na(self):
        IV = IVdata(self.cell_dict)
        self.testclampna = IV.compute_ivdata(vlow=-80, vhigh=40, n_steps=13, delay=10, duration=5)
        return self.testclampna

    def volt_inj_k(self):
        IV = IVdata(self.cell_dict)
        self.testclampk = IV.compute_ivdata(vlow=-90, vhigh=0, n_steps=10, delay=10, duration=5)
        return self.testclampk

    def sim_fi(self):
        ep = ElectrophysiologicalPhenotype(self.cell_dict, noise=False)
        self.simfi = ep.compute_fi_curve(ilow=-0.4, ihigh=0.4, n_steps=110, delay=0, duration=1000)
        return self.simfi

    def data_fi(self):
        x = [0., 0.040, 0.080, 0.120, 0.160, 0.200, 0.240, 0.280, 0.320, 0.360, 0.400]
        y = [0, 1, 4, 8.5, 14, 17, 20, 21.5, 22, 25, 26]
        datafi = [x, y]
        self.datafi = np.array(datafi)
        return self.datafi

    def data_iv_na(self):
        v = np.linspace(-80, 40, 13)
        i = [-0.01, -0.01, -0.01, -0.02, -0.07, -0.2, -0.38, -0.36, -0.3, -0.24, -0.19, -0.14, -0.11]
        dataivna = [v, i]
        self.dataivna = np.array(dataivna)
        return self.dataivna

    def data_iv_k(self):
        v = np.linspace(-90, 0, 10)
        ik = [54, 72, 18, 72, 226, 469, 929, 1362, 1796, 2265]
        iknA = [x / 1000 for x in ik]
        dataiv = [v, iknA]
        self.dataiv_k = np.array(dataiv)
        return self.dataiv_k

    def optimized_cell(self):
        mossyopt = pd.read_csv("data/parameters/parameters_mossy.csv")
        mossyparams = mossyopt['Cell_0'].to_numpy()
        j = 0
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = mossyparams[j]
                j = j + 1
        epmc_opt = ElectrophysiologicalPhenotype(self.cell_dict, noise=False)
        mcFI_opt = epmc_opt.compute_fi_curve(ilow=0, ihigh=0.33, n_steps=120, delay=0, duration=1500)
        return mcFI_opt

    def manual_adjust(self):
        baseline = self.sim_fi()
        baselineivna = self.volt_inj_na()
        baselineivk = self.volt_inj_k()

        self.cell_dict['secs']['soma']['mechs']['bk']['gkbar'] = 0.0165 #0.03039610718265275  # 0.0006
        # --- SODIUM
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gnatbar'] =  0.12 #0.055866501682604486
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftma'] = 43 #59  # 25.729832752129596  # 43.0
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftmb'] = 15 #27.03685981762697  # 15.0
        self.cell_dict['secs']['soma']['mechs']['ichan2'][
            'vshiftha'] = 65 #125  # 125.28480038847258 #125.28480038847258  # 65.0  # of interest
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshifthb'] = 12.5 #18.616547074320465  # 12.5  # of interest

        # --- POTASSIUM

        self.cell_dict['secs']['soma']['mechs']['ichan2'][
            'gkfbar'] = 0.0005 #0.0286296712742244  # 0.0005 # 0.001002156114034962 #0.016
        self.cell_dict['secs']['soma']['mechs']['ichan2'][
            'gksbar'] = 0.0 #0.014713327547369214  # 0.006 #
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnfa'] = 18 #30.701653087965823  # 18.0
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnfb'] = 43 #24.156010322453657  # 125.4040070771422 #43
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnsa'] = 30 #44.694885585078325  # 54.1958679173991  # 30
        self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnsb'] = 55 #17.74053364029013  # 76.16308645928503  # 55

        self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] = 1e-05 #1.650234883413953e-05  # 0.012
        self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] = 0.0006 #0.00017469257304634966  # 0.005
        self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] = 8e-05 #3.0246490314670187e-05  # 0.002
        self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] = 0.016 #0.022809187461813662  # 0.001
        self.cell_dict['secs']['soma']['mechs']['ih']['ghyfbar'] = 5e-06 #9.297236447233174e-07
        self.cell_dict['secs']['soma']['mechs']['ih']['ghysbar'] = 5e-06 #5.9517273494970865e-06
        shifted = self.sim_fi()
        shiftedivna = self.volt_inj_na()
        shiftedivk = self.volt_inj_k()
        aptest = self.curr_inj(0)

        dataivna = self.data_iv_na()
        dataivk = self.data_iv_k()

        datafi = self.data_fi()
        optimized_cell = self.optimized_cell()

        plt.plot(baseline['I'], baseline['F'], label="baseline")
        plt.plot(shifted['I'], shifted['F'], label="manual fit")
        plt.plot(optimized_cell['I'], optimized_cell['F'], label="optimized")
        plt.plot(datafi[0, :], datafi[1, :], label="data")
        plt.legend()
        plt.savefig("figures/mossycell/ifvsif.jpeg")
        plt.close()

        plt.plot(baselineivna['V'], baselineivna['Na'], label='BL Na')
        plt.plot(baselineivk['V'], baselineivk['K'], label='BL K')
        plt.plot(dataivna[0, :], dataivna[1, :], label='D Na')
        plt.plot(shiftedivna['V'], shiftedivna['Na'], label='SH Na')
        plt.plot(dataivk[0, :], dataivk[1, :], label='D K')
        plt.plot(shiftedivk['V'], shiftedivk['K'], label='SH K')
        plt.legend()
        plt.savefig("figures/mossycell/ivvsiv.jpeg")
        plt.close()

        plt.plot(aptest['t'], aptest['V'])
        plt.savefig("figures/mossycell/MOSSYcurrinj.jpeg")


TestParam = testparam(mc, free_params)
TestParam.manual_adjust()

'''
goodparams = [0.03039610718265275,  # ['bk']['gkbar']
              0.05753540741645549,  # ['ichan2']['gnatbar']
              59,  # ['ichan2']['vshiftma']
              27.03685981762697,  # ['ichan2']['vshiftmb']
              125,  # ['ichan2']['vshiftha']
              21.692365618111857,  # ['ichan2']['vshifthb']
              0.039537232546728474,  # ['ichan2']['gkfbar']
              0.00813724848118442,  # ['ichan2']['gksbar']
              30.701653087965823,  # ['ichan2']['vshiftnfa']
              24.156010322453657,  # ['ichan2']['vshiftnfb']
              44.694885585078325,  # ['ichan2']['vshiftnsa']
              17.74053364029013,  # ['ichan2']['vshiftnsb']
              1.650234883413953e-05,  # ['ka']['gkabar']
              0.0007649406562098781,  # ['lca']['glcabar']
              0.0001476092734232904,  # ['nca']['gncabar']
              0.022809187461813662,  # ['sk']['gskbar']
              9.297236447233174e-07,  # ['ih']['ghyfbar']
              5.9517273494970865e-06  # ['ih']['ghysbar']
              ]
'''
