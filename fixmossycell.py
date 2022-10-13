# PARAMETER OPTIMIZATION TO FIX MOSSY CELL FI CURVES, Na and K I-V RELATIONSHIPS
# TEMPORARY FILE: Will unify with optimize_params.py in future
# --------------------
# DATA USED FOR FITTING:
# Howard AL, Neu A, Morgan RJ, Echegoyen JC, Soltesz I. Opposing Modifications in Intrinsic Currents and
# Synaptic Inputs in Post-Traumatic Mossy Cells: Evidence for Single-Cell Homeostasis in a Hyperexcitable Network.
# Journal of Neurophysiology. 2007;97(3):2394–2409. doi:10.1152/jn.00509.2006
# --------------------
import matplotlib
import numpy as np

# nicer font options:
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

matplotlib.use('Agg')  # hopefully this works over ssh
import matplotlib.pyplot as plt
import pylab
import pandas as pd
from random import Random
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
    cellArgs=[1],
    importSynMechs=False
)

# parameters to be optimized
free_params = {
    'bk': ['gkbar'],  # big conductance, calcium-activated potassium channel
    'ichan2': ['gnatbar', 'vshiftma', 'vshiftmb', 'vshiftha', 'vshifthb', 'vshiftnfa', 'vshiftnfb', 'vshiftnsa',
               'vshiftnsb',
               'gkfbar', 'gl'],  # sodium, potassium parameters
    'ka': ['gkabar'],  # A-type (fast inactivating) Kv channel
    'lca': ['glcabar'],  # l-type calcium
    'nca': ['gncabar'],  # n-type calcium
    'sk': ['gskbar'],  # small conductance potassium channel
    'ih': ['ghyfbar', 'ghysbar']  # HCN channel
}


class testparam(object):
    def __init__(self,
                 cell,
                 free_params,
                 pop_size=10,
                 max_evaluations=60,
                 num_selected=10,
                 mutation_rate=0.03,
                 ):
        self.cell_dict = {"secs": cell["secs"]}
        self.free_params = free_params
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
        self.flag = str('mossy')
        self.n_simcells = 1  # number of simulated cells

        # self.manual_adjust()
        self.plot_results()

    def retrieve_baseline_params(self):
        """ Saves baseline parameters from cell_dict

        Returns:
            'list'. List of baseline parameter values.
        """
        self.baseline = []
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.baseline.append(self.cell_dict['secs']['soma']['mechs'][key][val])
        self.num_inputs = len(self.baseline)
        return self.baseline

    def curr_inj(self, current, delay=0, duration=1000):
        iclamp = IClamp(self.cell_dict, delay=delay, duration=duration, T=duration + delay * 2)
        res = iclamp(current)
        return res

    def sim_fi(self):
        ep = ElectrophysiologicalPhenotype(self.cell_dict, noise=False)
        self.simfi = ep.compute_fi_curve(ilow=0, ihigh=0.4, n_steps=11, delay=0, duration=1000)
        return self.simfi

    def sim_iv_na(self):
        iv = IVdata(self.cell_dict)  # instantiate class
        self.simivna = iv.compute_ivdata(vlow=-80, vhigh=40, n_steps=13, delay=10, duration=5)
        return self.simivna

    def sim_iv_k(self):
        iv = IVdata(self.cell_dict)  # instantiate class
        self.simivk = iv.compute_ivdata(vlow=-90, vhigh=0, n_steps=10, delay=10, duration=5)
        return self.simivk

    def data_fi(self):
        x = [0, 0.040, 0.080, 0.120, 0.160, 0.200, 0.240, 0.280, 0.320, 0.360, 0.400]
        y = [0, 1, 4, 8.5, 14, 17, 20, 21.5, 22, 25, 26]
        datafi = [x, y]
        self.datafi = np.array(datafi)
        return self.datafi

    def data_iv_na(self):
        v = np.linspace(-80, 40, 13)
        ina = [-0.01, -0.01, -0.01, -0.02, -0.07, -0.2, -0.38, -0.36, -0.3, -0.24, -0.19, -0.14, -0.11]
        dataiv = [v, ina]
        self.dataiv_na = np.array(dataiv)
        return self.dataiv_na

    def data_iv_k(self):
        v = np.linspace(-90, 0, 10)
        ik = [54, 72, 18, 72, 226, 469, 929, 1362, 1796, 2265]
        iknA = [x / 1000 for x in ik]
        dataiv = [v, iknA]
        self.dataiv_k = np.array(dataiv)
        return self.dataiv_k

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
            i = 0
            for k in free_params.keys():
                for v in free_params[k]:
                    self.cell_dict['secs']['soma']['mechs'][k][v] = cand[i]
                    i += 1
            FI_data = self.data_fi()
            FI_sim = self.sim_fi().to_numpy()
            Na_data = self.data_iv_na()
            Na_sim = self.sim_iv_na().to_numpy()
            K_data = self.data_iv_k()
            K_sim = self.sim_iv_k().to_numpy()

            ficurves = np.sum([((x1 - x2) ** 2) for (x1, x2) in zip(FI_data[1, :], FI_sim[:, 1])]) / len(FI_data[1, :])
            na_currs = np.sum([((x1 - x2) ** 2) for (x1, x2) in zip(Na_data[1, :], Na_sim[:, 1])]) / len(Na_data[1, :])
            k_currs = np.sum([((x1 - x2) ** 2) for (x1, x2) in zip(K_data[1, :], K_sim[:, 2])]) / len(K_data[1, :])

            fitness = (1 / 3 * ficurves) + (1 / 3 * na_currs) + (1 / 3 * k_currs)

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

        # self.minParamValues = [0.5 * param for param in self.baseline]  # 0.5 best for IF, Na
        # self.maxParamValues = [3.0 * param for param in self.baseline]  # 3.0 best for IF, Na

        self.minParamValues = [(0.0165 * 0.1), (0.05 * 0.8), (30.0 * 0.8), (22 * 0.8), (100 * 0.8), (15 * 0.8),
                               (18.0 * 0.1), (43.0 * 0.1), (30.0 * 0.1), (55.0 * 0.1), (0.020 * 0.8), (1.1e-05 * 0.1),
                               (1e-05 * 0.1), (0.0006 * 0.1), (8e-05 * 0.1), (0.016 * 0.1),
                               (5e-06 * 0.1), (5e-06 * 0.1),
                               ]

        self.maxParamValues = [(0.0165 * 2.0), (0.05 * 1.2), (30.0 * 1.5), (22 * 1.5), (100 * 1.5), (15 * 1.5),
                               (18.0 * 2.0), (43.0 * 2.0), (30.0 * 2.0), (55.0 * 2.0), (0.03 * 1.5), (1.1e-05 * 2.0),
                               (1e-05 * 2.0), (0.0006 * 2.0), (8e-05 * 2.0), (0.016 * 2.0),
                               (5e-06 * 2.0), (5e-06 * 2.0),
                               ]

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
        return self.bestCand

    def build_optimizedcell(self):
        """ Replaces baseline parameters with parameters from best candidate, then uses current injection experiment
            to build 'optimized' cell.

        Returns
            'dict'. Results of current clamp from optimized cell.
        """
        j = 0
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = self.bestCand[j]
                j = j + 1
        finalclamp = self.curr_inj(0.33)
        # self.ep_opt = ElectrophysiologicalPhenotype(self.cell_dict)
        return finalclamp

    def store_curves(self):
        """ Generates set of n optimized neurons (n_simcells), stores baseline and optimized parameters,
            IF and IV curves.

        Returns
            'tuple' of two DataFrames, (sim_fi_store, sim_iv_store)
        """
        # initialize empty DataFrames, populate with baseline parameters
        baselineparams = self.retrieve_baseline_params()
        self.param_store = pd.DataFrame({"param": sum(free_params.values(), []), "baseline": baselineparams})
        self.sim_fi_store = pd.DataFrame([])
        self.sim_ivna_store = pd.DataFrame([])
        self.sim_ivk_store = pd.DataFrame([])

        # generate set of n_simcells, populate DataFrames above with FI, IV, params
        for cell_n in range(0, self.n_simcells):
            self.setseed = cell_n  # set new seed for evol'n computation
            newparams = self.find_bestcandidate()  # find optimized parameters
            newparamdf = pd.DataFrame({"Cell_%s" % cell_n: newparams})  # store those params with a label
            self.param_store = pd.concat([self.param_store, newparamdf], axis=1)  # append params to DF
            self.build_optimizedcell()  # build the optimized cell
            newcellfi = self.sim_fi()  # generate simulated FI curve
            newcellivna = self.sim_iv_na()  # generate simulated IV curves
            newcellivk = self.sim_iv_k()  # generate simulated IV curves
            self.sim_fi_store = pd.concat([newcellfi, self.sim_fi_store])  # append FI curve to DF
            self.sim_ivna_store = pd.concat([newcellivna, self.sim_ivna_store])  # append IV curve to DF
            self.sim_ivk_store = pd.concat([newcellivk, self.sim_ivk_store])  # append IV curve to DF

        # save dataframes to .csv
        self.sim_fi_store.to_csv('data/parameters/simFIs_%s.csv' % self.flag)
        self.sim_ivna_store.to_csv('data/parameters/simIVsNa_%s.csv' % self.flag)
        self.sim_ivk_store.to_csv('data/parameters/simIVsK_%s.csv' % self.flag)
        self.param_store.to_csv('data/parameters/parameters_%s.csv' % self.flag)

        return self.sim_fi_store, self.sim_ivna_store, self.sim_ivk_store

    def compute_avg_curves(self):
        """ Computes average simulated FI and IV curves and SEM

        Returns
            'tuple' of two DataFrames, (avg_FI, avg_IV)
        """
        sim_stores = self.store_curves()
        sim_fi_store = sim_stores[0]
        sim_ivna_store = sim_stores[1]
        sim_ivk_store = sim_stores[2]

        # average simulated FI curve:
        avgfi = sim_fi_store.groupby(['I']).agg({'F': ['mean']}).values
        semfi = sim_fi_store.groupby(['I']).agg({'F': ['std']}).values / np.sqrt(self.n_simcells)
        self.avg_FI = np.c_[np.linspace(0, 0.4, 11), avgfi, semfi]

        iv_na = sim_ivna_store.groupby(['V']).agg({'Na': ['mean']}).values
        iv_k = sim_ivk_store.groupby(['V']).agg({'K': ['mean']}).values
        stdv_na = sim_ivna_store.groupby(['V']).agg({'Na': ['std']}).values / np.sqrt(self.n_simcells)
        stdv_k = sim_ivk_store.groupby(['V']).agg({'K': ['std']}).values / np.sqrt(self.n_simcells)
        self.avg_IV_Na = np.c_[np.linspace(-80, 40, 13), iv_na, stdv_na]
        self.avg_IV_K = np.c_[np.linspace(-90, 0, 10), iv_k, stdv_k]

        return self.avg_FI, self.avg_IV_Na, self.avg_IV_K

    def plot_results(self):
        """ Plots average simulated IV and FI curves from optimized neurons against avg curves from data. Saves
        figure to 'figures/op-output'. Automatically called when optimizeparams is instantiated.
        """

        # Generate and collect all data for plotting
        baselinecellfi = self.sim_fi().to_numpy()
        baselinecellivna = self.sim_iv_na().to_numpy()
        baselinecellivk = self.sim_iv_k().to_numpy()
        exp_fi = self.data_fi()
        exp_ivna = self.data_iv_na()
        exp_ivk = self.data_iv_k()
        simcurves = self.compute_avg_curves()
        avg_fi = simcurves[0]
        avg_ivna = simcurves[1]
        avg_ivk = simcurves[2]

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
        # FI curves
        ax1.plot(baselinecellfi[:, 0], baselinecellfi[:, 1], color='0.7', linestyle='dashed', label='Baseline')
        ax1.plot(exp_fi[0, :], exp_fi[1, :], color='0.5', label='Data')
        ax1.errorbar(avg_fi[:, 0], avg_fi[:, 1], yerr=avg_fi[:, 2], color='0.0', label='Optimized')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel("Current (nA)")
        ax1.set_ylabel("Frequency (Hz)")

        # IV curve: Na
        ax2.plot(baselinecellivna[:, 0], baselinecellivna[:, 1], color='0.7', linestyle='dashed', label='Baseline Na')
        ax2.plot(exp_ivna[0, :], exp_ivna[1, :], color='0.5', label='Data Na')
        ax2.plot(avg_ivna[:, 0], avg_ivna[:, 1], color='0.0', label='Optimized Na')
        ax2.axhline(0, lw=0.25, color='0.0')  # x = 0
        ax2.axvline(0, lw=0.25, color='0.0')  # y = 0
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel("Voltage (mV)")
        ax2.set_ylabel("Current (nA)")

        ax3.plot(baselinecellivk[:, 0], baselinecellivk[:, 2], color='0.7', linestyle='dashed', label='Baseline K')
        ax3.plot(exp_ivk[0, :], exp_ivk[1, :], color='0.5', label='Data K')
        ax3.plot(avg_ivk[:, 0], avg_ivk[:, 1], color='0.0', label='Optimized K')
        ax3.axhline(0, lw=0.25, color='0.0')  # x = 0
        ax3.axvline(0, lw=0.25, color='0.0')  # y = 0
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_xlabel("Voltage (mV)")
        ax3.set_ylabel("Current (nA)")

        # IV curve: K
        # ax3.plot(baselinecelliv[:, 0], baselinecelliv[:, 2], color='0.7', linestyle='dashed', label='Baseline K')
        # ax3.errorbar(exp_iv[:, 0], (exp_iv[:, 3] + exp_iv[:, 5]), yerr=(exp_iv[:, 4] + exp_iv[:, 6]),
        #             color='0.5', label='Data K')
        # ax3.errorbar(avg_iv[:, 0], avg_iv[:, 3], yerr=avg_iv[:, 4], color='0.0', label='Optimized K')
        # ax3.axhline(0, lw=0.25, color='0.0')  # x = 0
        # ax3.axvline(0, lw=0.25, color='0.0')  # y = 0
        # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax3.set_xlabel("Voltage (mV)")
        # ax3.set_ylabel("Current (nA)")

        fig1.tight_layout()
        fig1.savefig('figures/mossycell/optimizationresults_%s.pdf' % self.flag, bbox_inches="tight")

    def manual_adjust(self):
        baseline = self.sim_fi()


TestParam = testparam(mc, free_params)
