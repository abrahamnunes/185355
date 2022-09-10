# Objects for extracting average FI and IV curves from raw data

import numpy as np
import pandas as pd

class extractFI(object):
    """ object for extracting average FI curves from DataFrame objects
    Arguments
        celldata: 'DataFrame' of raw FI curves
        condition: 'str'. Experimental condition. Can either be 'CTRL' or 'LITM'

    Returns
        'DataFrame'. Average and SEM statistics for specified condition
    """
    def __init__(self, celldata, condition):
        self.FI_curve = celldata
        self.condition = condition
        self.CTRL_APs = []
        self.LITM_APs = []

    def averageFI(self):
        """ Computes average FIs and SEM, returns DataFrame"""
        aggregate = self.FI_curve.groupby(['Condition', 'Current']).agg({'AP': ['mean']})
        stdv = self.FI_curve.groupby(['Condition', 'Current']).agg({'AP': ['std']})
        counts = self.FI_curve.groupby(['Condition']).agg({'Cell': ['nunique']})  # computes number of cells/condition

        ## separate out CTRL and LITM conditions
        self.CTRL_APs = aggregate.AP['mean']['CTRL'].values
        self.CTRL_SEM = stdv.AP['std']['CTRL'].values / np.sqrt(counts.Cell['nunique']['CTRL'])
        self.LITM_APs = aggregate.AP['mean']['LITM'].values
        self.LITM_SEM = stdv.AP['std']['LITM'].values / np.sqrt(counts.Cell['nunique']['LITM'])

        ## add current vals as column
        self.CTRL_APs = np.c_[np.linspace(0, 0.033, 12), self.CTRL_APs, self.CTRL_SEM]
        self.LITM_APs = np.c_[np.linspace(0, 0.033, 12), self.LITM_APs, self.LITM_SEM]

        if self.condition == 'CTRL':
            return self.CTRL_APs
        if self.condition == 'LITM':
            return self.LITM_APs


class extractIV(object):
    """ object for extracting average IV curves from DataFrame objects
        Arguments
            celldata: 'DataFrame' of raw IV curves
            condition: 'str'. Experimental condition. Can either be 'CTRL' or 'LITM'

        Returns
            'DataFrame'. Average and SEM statistics for specified condition
    """
    def __init__(self, celldata, condition):
        self.IV_data = celldata
        self.condition = condition
        self.aggregateIVs = []

    def averageIV(self):
        """ Computes average IVs for Na, Kfast and Kslow currents and SEM, returns DataFrame"""
        avgIV = self.IV_data.groupby(['Condition', 'Ion', 'Voltage']).agg({'Current_nA': ['mean']})
        stdv = self.IV_data.groupby(['Condition', 'Ion', 'Voltage']).agg({'Current_nA': ['std']})
        counts = self.IV_data.groupby(['Condition', 'Ion']).agg(
            {'Cell': ['nunique']})  # computes number of cells/condition

        ions = ['Na', 'Kfast', 'Kslow']
        avgstats = []
        for ion in ions:
            IV = avgIV.Current_nA['mean'][self.condition][ion].values
            SEM = stdv.Current_nA['std'][self.condition][ion].values / np.sqrt(
                counts.Cell['nunique'][self.condition][ion])
            avgstatstempct = np.c_[IV, SEM]
            avgstats.append(avgstatstempct)  # append avg IV+SEM/ion next to each other
        avgstats = np.hstack(avgstats)
        self.aggregateIVs = np.c_[np.linspace(-70,20,10), avgstats]

        return self.aggregateIVs
