import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from netpyne import sim, specs
from scipy.optimize import golden
from clamps import VClamp, IClamp
import random

netparams = specs.NetParams()
gc = netparams.importCellParams(
    label="GC",
    conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
    fileName="objects/GC.hoc",
    cellName="GranuleCell",
    cellArgs=[1],
    importSynMechs=False
)

class ElectrophysiologicalPhenotype(object):
    """ This object computes a wide variety of electrophysiological properties for 
        a given neuron object. 
    """
    def __init__(self, cell, noise):
        self.cell_dict = {"secs": cell["secs"]}
        self.noise = noise
        self.fi_data = {} 
        self.fi_curve = None
        self.rheo_current_brack = None

    def step_current(self, current, delay=250, duration=500):
        """ Injects a level of current and returns the number of spikes emitted
        
        Arguments: 
            current: `float`. Amount of current injected [nA]
            delay: `float`. Time after recording starts where current is injected [ms]
            duration: `float`. Total duration of the current injection [ms]
        
        Returns: 
            `dict`. Results of the step current injection simulation
        
        """
        if self.noise:
            iclamp = IClamp(self.cell_dict, noise=True, delay=delay, duration=duration, T=duration + delay * 2)
            res = iclamp(current)

        else:
            iclamp = IClamp(self.cell_dict, noise=False, delay=delay, duration=duration, T=duration + delay*2)
            res = iclamp(current)
        return res

    def compute_fi_curve(self, ilow=0, ihigh=0.5, n_steps=100, delay=250, duration=500):
        """ Computes the fi curve, and stores the raw data

        Arguments: 
            ilow: `float`. Starting current injection 
            ihigh: `float`. Top current injection 
            n_steps: `int`. Number of current injection steps 
            delay: `float`. Time after recording starts where current is injected [ms]
            duration: `float`. Total duration of the current injection [ms]

        Returns: 
            `pandas.DataFrame`.
        """
        self.fi_curve = pd.DataFrame(np.zeros((n_steps, 2)), columns=["I", "F"])
        current_steps = np.linspace(ilow, ihigh, n_steps)
        self.fi_curve["I"] = current_steps 

        for j, current in enumerate(current_steps):
            self.fi_data[j] = self.step_current(current, delay=delay, duration=duration)
            self.fi_data[j]["current"] = current
            self.fi_curve.iloc[j, 0] = current
            self.fi_curve.iloc[j, 1] = self.fi_data[j]["rate"]

        #self._get_rheobase_bracket()

        return self.fi_curve

    def _get_rheobase_bracket(self):
        """ Finds the initial bracket for the rheobase calculation based on the fI curve """
        ilow = np.max(self.fi_curve.loc[self.fi_curve.F == 0, "I"].values)
        ihigh = np.min(self.fi_curve.loc[self.fi_curve.F > 0, "I"].values)
        self.rheo_current_brack = (ilow, ihigh)

    def find_rheobase(self, current_brack=(0, 1), delay=250, duration=500, tol=1e-3):
        """ Finds the rheobase of the clamped neuron using Golden Section Search, 
            minimizing the loss function ((n_spikes-1) + I)^2, where I is the 
            injected current level 
        
        Arguments: 
            current_brack: `tuple` or `None`. (low, mid, high) for golden section search
            delay: `float`. Time after recording starts where current is injected [ms]
            duration: `float`. Total duration of the current injection [ms]
            tol: `float`. The tolerance level 

        Returns: 
            `float`. Rheobase 
        """
        # Use current bracket defined from F-I curve preferentially
        if self.rheo_current_brack is None:
            self.rheo_current_brack = current_brack 

        # Define the loss function 
        rheo_loss = lambda i: (( len(self.step_current(i, delay, duration)["spkt"]) - 1) + i)**2
        
        # Perform golden section search
        self.rheobase = golden(rheo_loss, brack=self.rheo_current_brack, tol=tol)

        return self.rheobase 



'''
gcpheno = ElectrophysiologicalPhenotype(gc)
volt = gcpheno.step_current(0.2)
plt.plot(volt['t'], volt['V'])
#fi = gcpheno.compute_fi_curve(n_steps=20, delay=500, duration=1000)
# gcpheno.find_rheobase()

i = 2
fig, ax = plt.subplots(nrows=2)
ax[0].plot(gcpheno.fi_data[i]["t"], gcpheno.fi_data[i]["V"])
ax[1].plot(gcpheno.fi_data[i]["t"], gcpheno.fi_data[i]["i"])
plt.show()
plt.close()

fi.plot('I','F') #plotting FI curve
'''



