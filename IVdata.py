#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:48:16 2022

vclamp test: working with abe's code to create iv curves
IVdata was built similarly to ElectrophysiologicalPhenotype

@author: selenasingh

June 23:
- need to update/fill out docstrings 
- check within 'compute_ivdata' if begin avg at 100ms appropriate 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netpyne import sim, specs
from scipy.optimize import golden
from clamps import VClamp, IClamp 

netparams = specs.NetParams()

#import granule cell info from hoc 
gc = netparams.importCellParams(
    label='GC',
    conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
    fileName="objects/GC.hoc",
    cellName="GranuleCell",
    cellArgs=[1], # [CHECK] what does this mean?
    importSynMechs=False
)

class IVdata(object) :
    """returns IV curve for various injected voltage vals"""
    def __init__(self, cell):
        self.cell_dict = {"secs": cell["secs"]}
        self.iv_interm = {}
        self.iv_data = None 
        
    def step_voltage(self, voltage, delay=250, duration=500):
        """
        
        Injects voltage, returns current

        Parameters
        ----------
        voltage : FLOAT
            voltage at which membrane is clamped to [mV]
        delay : FLOAT, optional
            Time after recording starts where voltage is clamped [ms]. The default is 250.
        duration : FLOAT, optional
            Total duration of simulation [ms]. The default is 500.

        Returns
        -------
        DICT
        results of voltage clamp

        """
        vclamp = VClamp(self.cell_dict, delay=delay, duration=duration, T=duration + delay*2)
        results = vclamp(voltage)
        return results 
    
    def compute_ivdata(self, vlow = -70, vhigh = 20, n_steps = 10, delay = 250, duration =500):
        """
        computes iv curve and stores it in a dataframe that can be used for plotting after.
        default vlow and vhigh vals chosen to match experimental volt clamp data 

        Parameters
        ----------
        vlow : FLOAT, optional
            DESCRIPTION. The default is -70.
        vhigh : FLOAT, optional
            DESCRIPTION. The default is 20.
        n_steps : INT, optional
            DESCRIPTION. The default is 10.
        delay : FLOAT, optional
            DESCRIPTION. The default is 250.
        duration : FLOAT, optional
            DESCRIPTION. The default is 500.

        Returns
        -------
        'pandas.DataFrame'

        """
        self.iv_data = pd.DataFrame(np.zeros((n_steps, 3)), columns=["V", "Na", "K"]) 
        voltage_steps = np.linspace(vlow, vhigh, n_steps) #creates same volt clamp steps used in experiments
        self.iv_data["V"] = voltage_steps 

        for j, voltage in enumerate(voltage_steps): 
            self.iv_interm[j] = self.step_voltage([voltage], delay=delay, duration=duration)
            self.iv_interm[j]["voltage"]=voltage 
            self.iv_data.iloc[j,0] = voltage
            mean_ina = np.mean(self.iv_interm[j]["i_na"][100:duration]) #CHECK if 100 is appropriate, or should use 'delay'
            mean_ik = np.mean(self.iv_interm[j]["i_k"][100:duration]) # ''
            self.iv_data.iloc[j,1] = mean_ina 
            self.iv_data.iloc[j,2] = mean_ik 
            
        return self.iv_data
            

###### RUN SIMULATIONS
#IV = IVdata(gc) # instantiate class 
#voltclamp = IV.step_voltage([0]) #testing if voltclamp works
#testclamp = IV.compute_ivdata(vlow = -70, vhigh = 20, n_steps = 10, delay = 250, duration = 500)

###### PLOTTING 
#plt.plot(testclamp['V'], testclamp['Na'])
               