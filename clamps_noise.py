import numpy as np
from netpyne import specs, sim

class ICNoise(object):
    def __init__(self, cell, delay=100, duration=200, T=400, dt=0.025,
                 record_step=0.1, verbose=False):
        """ Runs a current-clamp experiment stimulating and recording at the soma, with low-level background noise

        Arguments:
            cell: `dict`. Cellular properties specified in NetPyNE dictionary syntax
            delay: `float`. Time after which current starts [ms]
            duration: `float`. Duration of current injection [ms]
            T: `float`. Total duration of simulation [ms]
            dt: `float`. Integration timestep [ms]
            record_step: `float`. Step size at which to save data [mS]

        """
        self.cell = cell
        self.duration = duration
        self.delay = delay
        self.T = T
        self.dt = dt
        self.record_step = record_step
        self.verbose = verbose

        self.netparams = specs.NetParams()
        self._set_netparams_neuron()
        self._set_netparams_stim()
        self._set_simparams()
        self._set_netparams_synmech()

    def _set_netparams_neuron(self):
        self.netparams.cellParams['neuron'] = self.cell
        self.netparams.popParams['pop'] = {'cellType': 'neuron', 'numCells': 1}

    def _set_netparams_synmech(self):
        self.netparams.synMechParams['exc'] = {'mod': 'Exp2Syn',
                                               'tau1': 0.1,
                                               'tau2': 5.0,
                                               'e': 0}

    def _set_netparams_stim(self):
        self.netparams.stimSourceParams['iclamp'] = {'type': 'IClamp',
                                                     'del': self.delay,
                                                     'dur': self.duration}
        self.netparams.stimSourceParams['bkg'] = {'type': 'NetStim',
                                                  'rate': 10}
        # 'noise': 0.5}
        self.netparams.stimTargetParams['iclamp->neuron'] = {
            'source': 'iclamp',
            'sec': 'soma',
            'loc': 0.5,
            'conds': {'pop': 'pop', 'cellList': [0]},
        }
        self.netparams.stimTargetParams['bkg->neuron'] = {
            'source': 'bkg',
            'sec': 'soma',
            'loc': 0.5,
            'conds': {'pop': 'pop', 'cellList': [0]},
            'delay': 5,
            'synMech': 'exc'
        }

    def _set_simparams(self):
        self.simconfig = specs.SimConfig()
        self.simconfig.duration = self.T
        self.simconfig.dt = self.dt
        self.simconfig.verbose = self.verbose
        self.simconfig.recordCells = ["all"]
        self.simconfig.recordTraces = {
            'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}
        }
        self.simconfig.recordStep = self.record_step

    def __call__(self, amp, noise, weight):
        """
        Arguments:
            amp: `float`. Current to be injected [nA]
            noise: 'float'. Amount of noise, where 0 is completely deterministic, 1 is completely random
            weight: 'float'. Weight of bkg population simulation onto neuron

        Returns:
            `dict`. Simulation data with the following key-value pairs
                - `t`: List representing time [ms]
                - `V`: List representing membrane voltage [mV]
                - `spkt`: List of spike times [ms]
                - `avg_rate`: Average firing rate across whole recording [Hz]
                - `rate`: Firing rate only during current injection [Hz]
        """
        self.netparams.stimSourceParams['iclamp']['amp'] = amp
        self.netparams.stimSourceParams['bkg']['noise'] = noise
        self.netparams.stimTargetParams['bkg->neuron']['weight'] = weight
        sim.createSimulateAnalyze(self.netparams, self.simconfig)

        print(list(sim.allSimData.keys()))
        results = {
            't': np.array(sim.allSimData['t']),
            'V': np.array(sim.allSimData['V_soma']['cell_0']),
            'spkt': np.array(sim.allSimData['spkt']),
            'avg_rate': sim.allSimData['avgRate'],
            'rate': len(sim.allSimData['spkt']) / (self.duration / 1000),
        }
        return results
