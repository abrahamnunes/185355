#deeply unsure about this

import numpy as np 
import pandas as pd

class extractFI(object):
    def __init__(self, celldata, condition):
        self.FI_curve = celldata
        self.condition = condition 
        self.CTRL_APs = []
        self.LITM_APs = []
        
    def averageFI(self):
        aggregate = self.FI_curve.groupby(['Condition', 'Current']).agg({'AP':['mean']}).round() 
        ## separate out CTRL and LITM conditions
        self.CTRL_APs = aggregate.AP['mean']['CTRL'].values.astype(int)
        self.LITM_APs = aggregate.AP['mean']['LITM'].values.astype(int)
        ## add current vals as column 
        self.CTRL_APs = np.c_[np.linspace(0,0.33,12), self.CTRL_APs] 
        self.LITM_APs = np.c_[np.linspace(0,0.33,12), self.LITM_APs]
        
        if self.condition == 'CTRL':
            return self.CTRL_APs
        if self.condition == 'LITM':
            return self.LITM_APs
    

