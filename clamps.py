# ========================================================================
#   DEFINES ELECTRODES FOR EVALUATING NEURONAL PROPERTIES
#
# ========================================================================
import numpy as np
from netpyne import specs, sim


class VClamp(object):
    def __init__(self,
                 cell,
                 delay=100,
                 duration=400,
                 T=600,
                 dt=0.025,
                 record_step=0.1,
                 verbose=False):
        """ Defines a voltage clamp object for stimulating and recording at the soma
        
        Arguments: 
            cell: `dict`. Cellular properties specified in NetPyNE dictionary syntax
            delay: `float`. Delay until voltage step is taken
            duration: `float`. Duration of current injection [ms]
            T: `float`. Total duration of simulation [ms]
            dt: `float`. Integration timestep [ms]
            record_step: `float`. Step size at which to save data [mS]

        TODO: 
            - [ ] Allow for changing the stimulation/recording location 
        """
        self.cell = cell
        self.delay = delay
        self.duration = duration
        self.T = T
        self.dt = dt
        self.record_step = record_step
        self.verbose = verbose

        self.netparams = specs.NetParams()
        self._set_netparams_neuron()
        self._set_netparams_stim()
        self._set_simparams()

    def _set_netparams_neuron(self):
        self.netparams.cellParams['neuron'] = self.cell
        self.netparams.popParams['pop'] = {'cellType': 'neuron', 'numCells': 1}

    def _set_netparams_stim(self):
        self.netparams.stimSourceParams['vclamp'] = {
            'type': 'VClamp',
            'dur': [
                self.delay,
                self.duration,
                self.T - (self.delay + self.duration),
            ],
        }
        self.netparams.stimTargetParams['vclamp->neuron'] = {
            'source': 'vclamp',
            'sec': 'soma',
            'loc': 0.5,
            'conds': {'pop': 'pop', 'cellList': [0]},
        }

    def _set_simparams(self):
        """
        TODO: 
            - [ ] Make it so that you can record many other types of currents
        """
        self.simconfig = specs.SimConfig()
        self.simconfig.duration = self.T
        self.simconfig.dt = self.dt
        self.simconfig.verbose = self.verbose
        self.simconfig.recordCells = ["all"]
        self.simconfig.recordTraces = {
            'i_na': {'sec': 'soma', 'loc': 0.5, 'var': 'ina'},
            'i_k': {'sec': 'soma', 'loc': 0.5, 'var': 'ik'},
        }
        self.simconfig.recordStep = self.record_step

    def __call__(self, amp):
        """ 
        Arguments: 
            amp: `list` of `float`. Voltage at which membrane is to be maintained [nA]

        Returns: 
            `dict`. Simulation data with the following key-value pairs 
                - `t`: List representing time [ms]
                - `i_na`: array of sodium currents 
                - `i_k`: array of potassium currents
                
        """
        self.netparams.stimSourceParams['vclamp']['amp'] = amp
        sim.createSimulateAnalyze(self.netparams, self.simconfig)
        results = {
            't': sim.allSimData['t'],
            'i_na': np.array(sim.allSimData['i_na']['cell_0']),
            'i_k': np.array(sim.allSimData['i_k']['cell_0']),
        }
        return results


class IClamp(object):
    def __init__(self, cell, delay=100, duration=200, T=400, dt=0.025,
                 record_step=0.1, verbose=False):
        """ Runs a current-clamp experiment stimulating and recording at the soma
        
        Arguments: 
            cell: `dict`. Cellular properties specified in NetPyNE dictionary syntax
            delay: `float`. Time after which current starts [ms]
            duration: `float`. Duration of current injection [ms]
            T: `float`. Total duration of simulation [ms]
            dt: `float`. Integration timestep [ms]
            record_step: `float`. Step size at which to save data [mS]

        TODO: 
            - [ ] Allow for changing the stimulation/recording location 
        """
        self.cell = cell
        self.delay = delay
        self.duration = duration
        self.T = T
        self.dt = dt
        self.record_step = record_step
        self.verbose = verbose

        self.netparams = specs.NetParams()
        self._set_netparams_neuron()
        self._set_netparams_stim()
        self._set_simparams()

    def _set_netparams_neuron(self):
        self.netparams.cellParams['neuron'] = self.cell
        self.netparams.popParams['pop'] = {'cellType': 'neuron', 'numCells': 1}

    def _set_netparams_stim(self):
        self.netparams.stimSourceParams['iclamp'] = {'type': 'IClamp',
                                                     'del': self.delay,
                                                     'dur': self.duration}
        self.netparams.stimTargetParams['iclamp->neuron'] = {
            'source': 'iclamp',
            'sec': 'soma',
            'loc': 0.5,
            'conds': {'pop': 'pop', 'cellList': [0]},
        }

    def _set_simparams(self):
        self.simconfig = specs.SimConfig()
        self.simconfig.duration = self.T
        self.simconfig.dt = self.dt
        self.simconfig.verbose = self.verbose
        self.simconfig.recordCells = ["all"]
        self.simconfig.recordTraces = {
            'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'},
            'iclamp': {'sec': 'soma', 'loc': 0.5, 'stim': 'iclamp->neuron', 'var': 'i'}
        }
        self.simconfig.recordStep = self.record_step

    def __call__(self, amp):
        """ 
        Arguments: 
            amp: `float`. Current to be injected [nA]

        Returns: 
            `dict`. Simulation data with the following key-value pairs 
                - `t`: List representing time [ms]
                - `V`: List representing membrane voltage [mV]
                - `spkt`: List of spike times [ms]
                - `avg_rate`: Average firing rate across whole recording [Hz]
                - `rate`: Firing rate only during current injection [Hz]
        """
        self.netparams.stimSourceParams['iclamp']['amp'] = amp
        sim.createSimulateAnalyze(self.netparams, self.simconfig)
        
        print(list(sim.allSimData.keys()))
        results = {
            't': np.array(sim.allSimData['t']),
            'V': np.array(sim.allSimData['V_soma']['cell_0']),
            'spkt': np.array(sim.allSimData['spkt']),
            'avg_rate': sim.allSimData['avgRate'],
            'rate': len(sim.allSimData['spkt']) / (self.duration / 1000),
            'i': sim.allSimData['iclamp']['cell_0']
        }
        return results