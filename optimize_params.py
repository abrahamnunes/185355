# imports
import time
start = time.time()

import numpy as np
import pandas as pd
import matplotlib

# nicer font options:
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

matplotlib.use('Agg')  # hopefully this works over ssh
import matplotlib.pyplot as plt
from random import Random
from inspyred import ec  # evolutionary algorithm
from netpyne import specs, sim  # neural network design and simulation
from clamps import IClamp
from find_rheobase import ElectrophysiologicalPhenotype
from CurvesFromData import extractFI, extractIV
from IVdata import IVdata
import random

netparams = specs.NetParams()

"""IMPORTS used by optimizeparams 
    gc: import granule cell from NEURON .hoc file 
    free_params: parameters of interest to be optimized
    raw_: imported csv to DataFrame objects containing raw data in long-table format 
"""


# import granule cell info from hoc
def importgc():
    gc = netparams.importCellParams(
        label='GC',
        conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
        fileName="objects/GC.hoc",
        cellName="GranuleCell",
        cellArgs=[1],
        importSynMechs=False
    )
    return gc


# parameters to be optimized
free_params = {
    'bk': ['gkbar'],  # big conductance, calcium-activated potassium channel
    'ichan2': ['gnatbar', 'vshiftma', 'vshiftmb', 'vshiftha', 'vshifthb', 'vshiftnfa', 'vshiftnfb', 'vshiftnsa',
               'vshiftnsb',
               'gkfbar', 'gksbar', 'gl'],  # sodium, potassium parameters
    'lca': ['glcabar'],  # l-type calcium
    'nca': ['gncabar'],  # n-type calcium
    'sk': ['gskbar'],  # small conductance potassium channel
    'tca': ['gcatbar']  # t-type calcium
}

# import raw data
rawhc = pd.read_csv("rawdata/HC_FI.csv")  # healthy control IFs
rawlr = pd.read_csv("rawdata/LR_FI.csv")  # lithium responder IFs
rawnr = pd.read_csv("rawdata/NR_FI.csv")  # lithium non-responder IFs
rawhciv = pd.read_csv("rawdata/HC_NaK_long.csv")  # healthy control IVs
rawlriv = pd.read_csv("rawdata/LR_NaK_long.csv")  # lithium responder IVs
rawnriv = pd.read_csv("rawdata/NR_NaK_long.csv")  # lithium non-responder IVs


class optimizeparams(object):
    """
    This object uses an evolutionary algorithm to optimize the parameters of a biophysical neuronal model of a
    hippocampal granule cell to frequency-current (FI) and current-voltage (IV) curves from iPSC-derived granule cells.

    Arguments

        cell: 'dict'. Model parameters imported from NEURON .hoc file
        free_params: 'dict'. Parameters to be optimized
        freqdata: 'DataFrame'. FI curve data imported from csv
        currdata: 'DataFrame'. IV curve data imported from csv (Na, Kfast and Kslow currents)
        population: 'str'. Participant group. Can be either: 'HC' or 'LR' or 'NR'
        condition: 'str'. Experimental condition. Can be either: 'CTRL' or 'LITM'
        pop_size: 'int', optional. Number of parameter sets per evaluation. Default = 10
        max_evaluations: 'int', optional. Terminate evolutionary iterations after max_evaluations. Default = 100
        num_selected: 'int', optional. Indicates how many parameter sets are selected for next evol'n iteration. Default = 10
        mutation_rate: 'float', optional. Rate of mutation of parameters. Default = 0.03

    Output
        Results from optimization algorithm: saved as .csv and .pdf figures in the folders 'data/parameters' and
        'figures/op-output'.
    """

    def __init__(self,
                 cell,
                 free_params,
                 freqdata,  # FI data from iPSC
                 currdata,  # IV data from iPSC
                 population,  # str, either 'HC', 'LR', 'NR'
                 condition,  # str , either 'CTRL' or 'LITM'
                 pop_size=10,
                 max_evaluations=350,
                 num_selected=10,
                 mutation_rate=0.03,
                 ):

        self.cell_dict = {"secs": cell["secs"]}
        self.baseline_dict = {"secs": cell["secs"]}
        self.free_params = free_params
        self.FI_curve = freqdata
        self.IV_curve = currdata
        self.condition = condition
        self.population = population
        self.initialParams = []
        self.minParamValues = []
        self.maxParamValues = []
        self.num_inputs = len(sum(self.free_params.values(), []))
        self.free_params = free_params
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.num_selected = num_selected
        self.mutation_rate = mutation_rate
        self.num_elites = 1
        self.flag = str(self.population + '_' + self.condition + '_' + 'wdends')
        self.n_simcells = 1  # number of simulated cells

        self.plot_results()  # run optimization upon class instantiation

    def retrieve_baseline_params(self):
        """ Saves baseline parameters from cell_dict

        Returns:
            'list'. List of baseline parameter values.
        """
        self.baseline = []
        dend1 = ["gcdend1_0", "gcdend1_1", "gcdend1_2", "gcdend1_3"]
        sections = ["soma"] + dend1
        for section in sections:
            for key in self.free_params.keys():
                for val in self.free_params[key]:
                    self.baseline.append(self.baseline_dict['secs'][section]['mechs'][key][val])
        self.num_inputs = len(self.baseline)

        return self.baseline

    def curr_inj(self, current, delay=0, duration=1000):
        """Injects current, returns number of action potentials

        Parameters:

            current : 'float'. Current at which membrane is clamped to [nA]
            delay : 'float', optional. Time after recording starts when current is clamped [ms]. The default is 0.
            duration : 'float', optional. Total duration of simulation [ms]. The default is 1000.

        Returns:
            'dict'. Results of current clamp.

        """
        iclamp = IClamp(self.cell_dict, noise=False, delay=delay, duration=duration, T=duration + delay * 2)
        res = iclamp(current)
        return res

    def sim_fi(self, noise):
        """Computes simulated FI curve, and stores the raw data

        Parameters:
            noise: 'bool'. Indicate whether to include background noise.

        Returns:
            'pandas.DataFrame'. Simulated FI curve.
        """
        ep = ElectrophysiologicalPhenotype(self.cell_dict, noise=noise)
        self.simfi = ep.compute_fi_curve(ilow=0, ihigh=0.033, n_steps=12, delay=0, duration=1500)
        return self.simfi

    def data_fi(self):
        """ Computes the average FI curve and accompanying SEM from imported iPSC data

        Returns:
            'ndarray'
        """
        self.avgFI = extractFI(self.FI_curve, self.condition).averageFI()
        return self.avgFI

    def sim_iv(self):
        """Computes simulated IV curves for Na and K currents, and stores results

        Returns:
            'pandas.DataFrame'. Simulated IV curves for Na and K currents.
        """
        iv = IVdata(self.cell_dict)  # instantiate class
        self.simiv = iv.compute_ivdata(vlow=-70, vhigh=20, n_steps=10, delay=10, duration=5)
        return self.simiv

    def data_iv(self):
        """ Computes the average IV curves for Na, Kfast and Kslow currents and accompanying SEM from imported iPSC data

        Returns:
            'ndarray'
        """
        self.avgIV = extractIV(self.IV_curve, self.condition).averageIV()
        return self.avgIV

    def generate_netparams(self, random, args):
        """
        Initialize set of random initial parameter values selected from uniform distribution within min-max bounds.

        Returns
            'list'. initialParams
        """
        self.initialParams = [random.uniform(self.minParamValues[i], self.maxParamValues[i]) for i in
                              range(self.num_inputs)]
        return self.initialParams

    def evaluate_netparams(self, candidates, args):
        """
        Fitness function that evaluates the fitness of candidate parameters by quantifying the difference between
        simulated FI and IV curves to the FI and IV curves from data using mean squared error.

        Returns
            'list'. Fitness values for sets of candidates
        """
        self.fitnessCandidates = []

        for cand in candidates:
            # TODO: find cleaner way of doing this
            i = 0
            dend1 = ["gcdend1_0", "gcdend1_1", "gcdend1_2", "gcdend1_3"]
            dend2 = ["gcdend2_0", "gcdend2_1", "gcdend2_2", "gcdend2_3"]
            sections = ["soma"] + dend1
            for section in sections:
                for k in free_params.keys():
                    for v in free_params[k]:
                        self.cell_dict['secs'][section]['mechs'][k][v] = cand[i]
                        i += 1
            # dendrite 1 params == dendrite 2 params
            for i in range(0,4):
                for k in free_params.keys():
                    for v in free_params[k]:
                        self.cell_dict['secs'][dend2[i]]['mechs'][k][v] = self.cell_dict['secs'][dend1[i]]['mechs'][k][v]
            
            FI_data = self.data_fi()
            FI_sim = self.sim_fi(noise=False).to_numpy()
            IV_data = self.data_iv()
            IV_sim = self.sim_iv().to_numpy()

            ficurves = np.sum([((x1 - x2) ** 2) for (x1, x2) in zip(FI_data[:, 1], FI_sim[:, 1])]) / len(FI_data[:, 1])
            na_currs = np.sum([((x1 - x2) ** 2) for (x1, x2) in zip(IV_data[:, 1], IV_sim[:, 1])]) / len(IV_data[:, 1])
            k_currs = np.sum(
                [((x1 - x2) ** 2) for (x1, x2) in zip((IV_data[:, 3] + IV_data[:, 5]), IV_sim[:, 2])]) / len(
                IV_data[:, 1]
            )

            fitness = (0.45*na_currs + 0.45*k_currs + 0.1*ficurves)

            self.fitnessCandidates.append(fitness)

        return self.fitnessCandidates

    def find_bestcandidate(self):
        """
        Sets up custom evolutionary computation and returns list of optimized parameters.

        Components of EC
            gc_ec : instantiate evolutionary computation with random.Random object
            selector: method used to select best candidate based on fitness value
            variator: method used to determine how mutations (variations) are made to each generation of params
            replacer: method used to determine if/how individuals are replaced in pool of candidates after selection
            terminator: method used to specify how to terminate evolutionary algorithm
            observer: method that allows for supervision of evolutionary computation across all evaluations
            evolve: pulls together all components of custom algorithm, iteratively calls evaluate_netparams, returns
                    parameters that minimize fitness function.

        Returns
            'list'. bestCand (list of optimized parameters)
        """

        # TODO: Potentially write custom variator function to be compatible with np.random.RandomState
        # rand = np.random.RandomState(self.setseed)

        rand = Random()
        rand.seed(self.setseed)  # will take cell # as seed (n_simcells = 1, seed = 0. n_simcells = 2, seed = 1, etc).

        # SET UP MIN/MAX BOUNDS FOR PARAMETERS ------------------
        # TODO: find cleaner way of dealing with these lists, allow for easier modification

        if self.population == "HC":
            scalemax = 1.898 # optimial initial conditions for HC dendrites have a smaller upper bound than BD conditions. 
            scalemin = 0.3
        elif self.flag == "LR_CTRL_wdends": 
            scalemax = 2.232 #2.1 
            scalemin = 0.30 #0.3
        elif self.flag == "LR_LITM_wdends": 
            scalemax = 2.233 #2.1 
            scalemin = 0.3165 #0.3165 w 350 iterations good. 
        else:
            scalemax = 2.234 
            scalemin = 0.25  

        #soma min/max bounds determined from single optimization
        soma_minbounds = [(0.0006 * 0.1), (0.3 * 0.9), (68 * 0.9), (22 * 0.9), (120 * 0.9), (20 * 0.9),
                          (33 * 0.9), (78 * 0.9), (41 * 0.9), (100 * 0.9), (0.020 * 0.9), (0.001 * 0.9),
                          (1.44E-05 * 1.0), (0.005 * 0.1), (0.002 * 0.1), (0.001 * 0.1), (3.70E-05 * 0.05)]

        soma_maxbounds = [(0.0006 * 2.0), (0.3 * 1.3), (68 * 1.1), (22 * 1.1), (120 * 1.1), (20 * 1.1),
                          (33 * 1.1), (78 * 1.1), (41 * 1.1), (100 * 1.1), (0.020 * 1.5), (0.001 * 1.5),
                          (1.44E-05 * 2.0), (0.005 * 2.0), (0.002 * 2.0), (0.001 * 2.0), (3.70E-05 * 1.0)]

        dendrite_minbounds = [scalemin * param for param in self.baseline[len(soma_minbounds):]]
        dendrite_maxbounds = [scalemax * param for param in self.baseline[len(soma_minbounds):]] #2.1 for LR and NRs

        self.minParamValues = soma_minbounds + dendrite_minbounds
        self.maxParamValues = soma_maxbounds + dendrite_maxbounds

        # SET UP EVOLUTIONARY COMPUTATION ----------------------
        self.gc_ec = ec.EvolutionaryComputation(rand)
        self.gc_ec.selector = ec.selectors.truncation_selection  # purely deterministic
        self.gc_ec.variator = [ec.variators.uniform_crossover, ec.variators.gaussian_mutation]
        self.gc_ec.replacer = ec.replacers.generational_replacement
        self.gc_ec.terminator = ec.terminators.evaluation_termination  # terminates after max number of evals is met
        self.gc_ec.observer = ec.observers.plot_observer  # save to file, use observers.file_observer

        self.final_pop = self.gc_ec.evolve(generator=self.generate_netparams,  # f'n for initializing params
                                           evaluator=self.evaluate_netparams,  # f'n for evaluating fitness values
                                           pop_size=self.pop_size,  # number of parameter sets per evaluation
                                           maximize=False,  # best fitness corresponds to minimum value
                                           bounder=ec.Bounder(  # set min/max param bounds
                                               self.minParamValues,
                                               self.maxParamValues
                                           ),
                                           max_evaluations=self.max_evaluations,
                                           num_selected=self.num_selected,
                                           mutation_rate=self.mutation_rate,
                                           num_inputs=self.num_inputs,
                                           num_elites=self.num_elites
                                           )

        self.final_pop.sort(reverse=True)  # sort final population so best fitness is first in list
        self.bestCand = self.final_pop[0].candidate  # bestCand <-- individual @ start of list

        plt.savefig('figures/op-output/observer_%s.pdf' % self.flag)  # save fitness vs. iterations graph
        plt.close()

        # save candidate list for debugging purposes
        file = open('data/parameters/bestCand.txt','w')
        for param in self.bestCand:
            file.write(str(param)+"\n")
        file.close()

        return self.bestCand

    def build_optimizedcell(self):
        """ Replaces baseline parameters with parameters from best candidate, then uses current injection experiment
            to build 'optimized' cell.

        Returns
            'dict'. Results of current clamp from optimized cell.
        """
        dend1 = ["gcdend1_0", "gcdend1_1", "gcdend1_2", "gcdend1_3"]
        dend2 = ["gcdend2_0", "gcdend2_1", "gcdend2_2", "gcdend2_3"]
        sections = ["soma"] + dend1
        j = 0
        for section in sections:
            for key in self.free_params.keys():
                for val in self.free_params[key]:
                    self.cell_dict['secs'][section]['mechs'][key][val] = self.bestCand[j]
                    j = j + 1
        for i in range(0,4):
            for k in free_params.keys():
                for v in free_params[k]:
                    self.cell_dict['secs'][dend2[i]]['mechs'][k][v] = self.cell_dict['secs'][dend1[i]]['mechs'][k][v]

        finalclamp = self.curr_inj(0.33)

        # save dictionary used to build optimized cell, for debugging purposes
        with open('data/parameters/build-cell-dict.txt', 'w') as f:
            print(self.cell_dict, file=f)
        f.close()        
        
        return finalclamp

    def revert_to_baseline(self):
        """ Replaces optimized parameters with baseline parameters to ensure each simulated neuron starts with the same
            baseline parameters.

        Returns
            'dict'. Results of current clamp.
        """
        # TODO: fix this. Doesn't seem to do anything currently.
        j = 0
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = self.baseline[j]
                j = j + 1
        clamptobuild = self.curr_inj(0.33)
        return clamptobuild

    def store_curves(self):
        """ Generates set of n optimized neurons (n_simcells), stores baseline and optimized parameters,
            IF and IV curves.

        Returns
            'tuple' of two DataFrames, (sim_fi_store, sim_iv_store)
        """
        # initialize empty DataFrames, populate with baseline parameters

        sections = 17*['soma'] + 17*['gcdend1_0'] + 17*['gcdend1_1'] + 17*['gcdend1_2'] + 17*['gcdend1_3']
        baselineparams = self.retrieve_baseline_params()
        self.param_store = pd.DataFrame({"sec": sections, "param": sum(free_params.values(), []) * 5,
                                         "baseline": baselineparams})
        self.sim_fi_store = pd.DataFrame([])
        self.sim_iv_store = pd.DataFrame([])

        # generate set of n_simcells, populate DataFrames above with FI, IV, params
        for cell_n in range(0, self.n_simcells):
            self.setseed = cell_n  # set new seed for evol'n computation
            newparams = self.find_bestcandidate()  # find optimized parameters
            newparamdf = pd.DataFrame({"Cell_%s" % cell_n: newparams})  # store those params with a label
            self.param_store = pd.concat([self.param_store, newparamdf], axis=1)  # append params to DF
            self.build_optimizedcell()  # build the optimized cell
            newcellfi = self.sim_fi(noise=False)  # generate simulated FI curve
            newcelliv = self.sim_iv()  # generate simulated IV curves
            self.sim_fi_store = pd.concat([newcellfi, self.sim_fi_store])  # append FI curve to DF
            self.sim_iv_store = pd.concat([newcelliv, self.sim_iv_store])  # append IV curve to DF
            self.revert_to_baseline()  # revert parameters back to baseline

        # save dataframes to .csv
        self.sim_fi_store.to_csv('data/parameters/simFIs_%s.csv' % self.flag)
        self.sim_iv_store.to_csv('data/parameters/simIVs_%s.csv' % self.flag)
        self.param_store.to_csv('data/parameters/parameters_%s.csv' % self.flag)

        return self.sim_fi_store, self.sim_iv_store

    def compute_avg_curves(self):
        """ Computes average simulated FI and IV curves and SEM

        Returns
            'tuple' of two DataFrames, (avg_FI, avg_IV)
        """
        sim_stores = self.store_curves()
        sim_fi_store = sim_stores[0]
        sim_iv_store = sim_stores[1]

        # average simulated FI curve:
        avgfi = sim_fi_store.groupby(['I']).agg({'F': ['mean']}).values
        semfi = sim_fi_store.groupby(['I']).agg({'F': ['std']}).values / np.sqrt(self.n_simcells)
        self.avg_FI = np.c_[np.linspace(0, 0.033, 12), avgfi, semfi]

        # average simulated IV curves:
        iv_na = sim_iv_store.groupby(['V']).agg({'Na': ['mean']}).values
        iv_k = sim_iv_store.groupby(['V']).agg({'K': ['mean']}).values
        stdv_na = sim_iv_store.groupby(['V']).agg({'Na': ['std']}).values / np.sqrt(self.n_simcells)
        stdv_k = sim_iv_store.groupby(['V']).agg({'K': ['std']}).values / np.sqrt(self.n_simcells)
        self.avg_IV = np.c_[np.linspace(-70, 20, 10), iv_na, stdv_na, iv_k, stdv_k]

        return self.avg_FI, self.avg_IV

    def plot_results(self):
        """ Plots average simulated IV and FI curves from optimized neurons against avg curves from data. Saves
        figure to 'figures/op-output'. Automatically called when optimizeparams is instantiated.
        """

        # Generate and collect all data for plotting
        currentvals = np.linspace(0, 0.033, 12)
        baselineparams = self.retrieve_baseline_params()
        baselinecellfi = self.sim_fi(noise=False).to_numpy()
        baselinecelliv = self.sim_iv().to_numpy()
        exp_fi = self.data_fi()
        exp_iv = self.data_iv()
        simcurves = self.compute_avg_curves()
        avg_fi = simcurves[0]
        avg_iv = simcurves[1]
        self.revert_to_baseline()

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
        # FI curves
        ax1.plot(baselinecellfi[:, 0], baselinecellfi[:, 1], color='0.7', linestyle='dashed', label='Baseline')
        ax1.errorbar(exp_fi[:, 0], exp_fi[:, 1], yerr=exp_fi[:, 2], color='0.5', label='Data')
        ax1.errorbar(avg_fi[:, 0], avg_fi[:, 1], yerr=avg_fi[:, 2], color='0.0', label='Optimized')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel("Current (nA)")
        ax1.set_ylabel("Frequency (Hz)")

        # IV curve: Na
        ax2.plot(baselinecelliv[:, 0], baselinecelliv[:, 1], color='0.7', linestyle='dashed', label='Baseline Na')
        ax2.errorbar(exp_iv[:, 0], exp_iv[:, 1], yerr=exp_iv[:, 2], color='0.5', label='Data Na')
        ax2.errorbar(avg_iv[:, 0], avg_iv[:, 1], yerr=avg_iv[:, 2], color='0.0', label='Optimized Na')
        ax2.axhline(0, lw=0.25, color='0.0')  # x = 0
        ax2.axvline(0, lw=0.25, color='0.0')  # y = 0
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel("Voltage (mV)")
        ax2.set_ylabel("Current (nA)")

        # IV curve: K
        ax3.plot(baselinecelliv[:, 0], baselinecelliv[:, 2], color='0.7', linestyle='dashed', label='Baseline K')
        ax3.errorbar(exp_iv[:, 0], (exp_iv[:, 3] + exp_iv[:, 5]), yerr=(exp_iv[:, 4] + exp_iv[:, 6]),
                     color='0.5', label='Data K')
        ax3.errorbar(avg_iv[:, 0], avg_iv[:, 3], yerr=avg_iv[:, 4], color='0.0', label='Optimized K')
        ax3.axhline(0, lw=0.25, color='0.0')  # x = 0
        ax3.axvline(0, lw=0.25, color='0.0')  # y = 0
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_xlabel("Voltage (mV)")
        ax3.set_ylabel("Current (nA)")

        fig1.tight_layout()
        fig1.savefig('figures/op-output/optimizationresults_%s.pdf' % self.flag, bbox_inches="tight")


# TODO: test reverttobaseline, see if we can eliminate the gc init
'''

opt_results = [
    optimizeparams(importgc(), free_params, rawnrn, rawnrniv, group, condition, )
    for (rawnrn, rawnrniv, group) in [(rawlr, rawlriv, "LR")]
    for condition in ["LITM"]
]

'''
opt_results = [
    optimizeparams(importgc(), free_params, rawnrn, rawnrniv, group, condition, )
    for (rawnrn, rawnrniv, group) in [(rawlr, rawlriv, "LR"), (rawnr, rawnriv, "NR")]
    for condition in ["CTRL", "LITM"]
]

# [(rawhc, rawhciv, "HC"),
import aggregate_plots 
aggregate_plots

end = time.time()
time_consumed=(end-start)/60

print("Optimization took %s minutes." %time_consumed)
