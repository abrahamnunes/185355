# deeply unsure about this

import numpy as np
import pandas as pd


class extractFI(object):
    def __init__(self, celldata, condition):
        self.FI_curve = celldata
        self.condition = condition
        self.CTRL_APs = []
        self.LITM_APs = []

    def averageFI(self):
        aggregate = self.FI_curve.groupby(['Condition', 'Current']).agg({'AP': ['mean']})
        stdv = self.FI_curve.groupby(['Condition', 'Current']).agg({'AP': ['std']})
        counts = self.FI_curve.groupby(['Condition']).agg({'Cell': ['nunique']}) #computes number of cells/condition

        ## separate out CTRL and LITM conditions
        self.CTRL_APs = aggregate.AP['mean']['CTRL'].values
        self.CTRL_SEM = stdv.AP['std']['CTRL'].values/np.sqrt(counts.Cell['nunique']['CTRL'])
        self.LITM_APs = aggregate.AP['mean']['LITM'].values
        self.LITM_SEM = stdv.AP['std']['LITM'].values/np.sqrt(counts.Cell['nunique']['LITM'])

        ## add current vals as column
        self.CTRL_APs = np.c_[np.linspace(0, 0.33, 12), self.CTRL_APs, self.CTRL_SEM]
        self.LITM_APs = np.c_[np.linspace(0, 0.33, 12), self.LITM_APs, self.LITM_SEM]

        if self.condition == 'CTRL':
            return self.CTRL_APs
        if self.condition == 'LITM':
            return self.LITM_APs
