# imports
import numpy as np
import pandas as pd
import matplotlib
# nicer font options:
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

matplotlib.use('Agg')  # hopefully this works over ssh
import matplotlib.pyplot as plt
import pylab
from random import Random  # TODO replace with numpy rand f'n.  pseudorandom number generation
from inspyred import ec  # evolutionary algorithm
from netpyne import specs, sim  # neural network design and simulation
from clamps import IClamp
from find_rheobase import ElectrophysiologicalPhenotype
from scipy.signal import find_peaks
from tabulate import tabulate
from FI_fromdata import extractFI
import similaritymeasures

netparams = specs.NetParams()

# import granule cell info from hoc
gc = netparams.importCellParams(
    label='GC',
    conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
    fileName="objects/GC.hoc",
    cellName="GranuleCell",
    cellArgs=[1],
    importSynMechs=False
)

# parameters to be optimized
free_params = {
    'bk': ['gkbar'],  # big conductance, calcium-activated potassium channel
    'ichan2': ['gnatbar', 'gkfbar', 'gksbar', 'gl'], # KDR channel conductances, sodium conductance
    'ka': ['gkabar'],  # A-type (fast inactivating) Kv channel
    'kir': ['gkbar'],  # inward rectifier potassium (Kir) channel
    'km': ['gbar'],  # KM channel
    'lca': ['glcabar'],  # l-type calcium
    'nca': ['gncabar'],  # n-type calcium
    'sk': ['gskbar'],  # small conductance potassium channel
    'tca': ['gcatbar']  # t-type calcium
}

# import raw data
rawhc = pd.read_csv("rawdata/HC_FI.csv")
rawlr = pd.read_csv("rawdata/LR_FI.csv")
rawnr = pd.read_csv("rawdata/NR_FI.csv")


class optimizeparams(object):
    def __init__(self,
                 cell,
                 free_params,
                 celldata,
                 population,  # str, either 'HC', 'LR', 'NR'
                 condition,  # str , either 'CTRL' or 'LITM'
                 difference_method,  # str options: 'Area', 'Frechet', 'DTW', 'PCM', 'MSE', 'CL'
                 pop_size=10,
                 max_evaluations=150,
                 num_selected=10,
                 mutation_rate=0.03,
                 num_elites=1,
                 targetRate=12,
                 # current=0.33
                 ):

        self.cell_dict = {"secs": cell["secs"]}
        self.baseline_dict = {"secs": cell["secs"]}
        self.free_params = free_params
        self.FI_curve = celldata
        self.condition = condition
        self.population = population
        self.difference_method = difference_method
        self.initialParams = []
        self.minParamValues = []
        self.maxParamValues = []
        self.num_inputs = 12
        self.free_params = free_params
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.num_selected = num_selected
        self.mutation_rate = mutation_rate
        self.num_elites = num_elites
        self.targetRate = targetRate
        self.flag = str(self.population + '_' + self.condition + '_' + self.difference_method)
        self.n_stimcells = 10

    def curr_inj(self, current, delay=0, duration=1000):
        iclamp = IClamp(self.cell_dict, delay=delay, duration=duration, T=duration + delay * 2)
        res = iclamp(current)
        return res

    def retrieve_baseline_params(self):
        self.baseline = []
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.baseline.append(self.baseline_dict['secs']['soma']['mechs'][key][val])
        self.num_inputs = len(self.baseline)

        return self.baseline

    def sim_fi(self):
        ep = ElectrophysiologicalPhenotype(self.cell_dict)
        self.simfi = ep.compute_fi_curve(ilow=0, ihigh=0.33, n_steps=12, delay=0, duration=2000)
        return self.simfi

    def data_fi(self):
        efi = extractFI(self.FI_curve, self.condition)
        self.avgFI = efi.averageFI()
        return self.avgFI

    def generate_netparams(self, random, args):
        self.initialParams = [random.uniform(self.minParamValues[i], self.maxParamValues[i]) for i in
                              range(self.num_inputs)]
        self.initialParams
        return self.initialParams

    # design fitness function, used in the ec evolve function --> final_pop = my_ec.evolve(...,evaluator=evaluate_netparams,...)
    def evaluate_netparams(self, candidates, args):
        self.fitnessCandidates = []

        for icand, cand in enumerate(candidates):
            # TODO find way to use free_params here to remove this ugly situation
            self.cell_dict['secs']['soma']['mechs']['bk']['gkbar'] = cand[0]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gnatbar'] = cand[1]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gkfbar'] = cand[2]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gksbar'] = cand[3]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gl'] = cand[4]
            self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] = cand[5]
            self.cell_dict['secs']['soma']['mechs']['kir']['gkbar'] = cand[6]
            self.cell_dict['secs']['soma']['mechs']['km']['gbar'] = cand[7]
            self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] = cand[8]
            self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] = cand[9]
            self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] = cand[10]
            self.cell_dict['secs']['soma']['mechs']['tca']['gcatbar'] = cand[11]

            FI_data = self.data_fi()
            FI_sim = self.sim_fi().to_numpy()

            if self.difference_method == "Area":
                fitness = abs(similaritymeasures.area_between_two_curves(FI_data, FI_sim))
            elif self.difference_method == "Frechet":
                fitness = abs(similaritymeasures.frechet_dist(FI_data, FI_sim))
            elif self.difference_method == "CL":  # curve length (i.e., arc-length)
                fitness = abs(similaritymeasures.curve_length_measure(FI_data, FI_sim))
            elif self.difference_method == "PCM":  # partial curve mapping
                fitness = abs(similaritymeasures.pcm(FI_data, FI_sim))
            elif self.difference_method == "DTW":  # dynamic time warping
                fitness = abs(similaritymeasures.dtw(FI_data, FI_sim)[0])
            elif self.difference_method == "MSE":  # for sim fi with 12 steps
                fitness = np.sum([((x1 - x2) ** 2) for (x1, x2) in zip(FI_data[:, 1], FI_sim[:, 1])]) / 12

            self.fitnessCandidates.append(fitness)

        return self.fitnessCandidates

    def find_bestcandidate(self):
        rand = Random()
        rand.seed()  # will take current time as seed

        normal_vals = [0.0006, 0.12, 0.016, 0.006, 1.44e-05, 0.012, 0.0, 0.001, 0.005,
                       0.002, 0.001, 3.7e-5]  # consistent with hoc file, removed HT, LT, GABA-A, Kir initialized to 0, so will stay 0
        self.minParamValues = list(np.array(normal_vals) * 0.2)  # allow min vals to be 80% lower
        self.maxParamValues = list(np.array(normal_vals) * 1.8)  # allow max vals to be 80% higher

        self.gc_ec = ec.EvolutionaryComputation(rand)
        self.gc_ec.selector = ec.selectors.truncation_selection  # truncation selection is purely deterministic. Choses param populations based on absol fitness
        self.gc_ec.variator = [ec.variators.uniform_crossover, ec.variators.gaussian_mutation]
        self.gc_ec.replacer = ec.replacers.generational_replacement
        self.gc_ec.terminator = ec.terminators.evaluation_termination  # terminates after max number of evals is met
        self.gc_ec.observer = ec.observers.plot_observer
        # self.gc_ec.observer = ec.observers.file_observer  # use to save optimizer data

        self.final_pop = self.gc_ec.evolve(generator=self.generate_netparams,
                                           # assign design parameter generator to iterator parameter generator
                                           evaluator=self.evaluate_netparams,
                                           # assign fitness function to iterator evaluator
                                           pop_size=self.pop_size,
                                           # each generation of parameter sets will consist of 10 individuals
                                           maximize=False,  # best fitness corresponds to minimum value
                                           bounder=ec.Bounder(self.minParamValues, self.maxParamValues),
                                           # boundaries for parameter set ([probability, weight, delay])
                                           max_evaluations=self.max_evaluations,
                                           num_selected=self.num_selected,
                                           # number of generated parameter sets to be selected for next generation
                                           mutation_rate=self.mutation_rate,  # rate of mutation
                                           num_inputs=self.num_inputs,  # len([probability, weight, delay])
                                           num_elites=self.num_elites)  # 1 existing individual will survive to next generation if it has better fitness than an individual selected by the tournament selection

        self.final_pop.sort(reverse=True)  # sort final population so best fitness (minimum difference) is first in list
        self.bestCand = self.final_pop[0].candidate  # bestCand <-- individual @ start of list

        plt.savefig('figures/op-output/observer_%s.pdf' % self.flag)
        plt.close()
        return self.bestCand

    def build_optimizedcell(self, currentval=0.33):
        j = 0
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = self.bestCand[j]
                j = j + 1
        finalclamp = self.curr_inj(currentval)
        self.ep_opt = ElectrophysiologicalPhenotype(self.cell_dict)
        return finalclamp

    def revert_to_baseline(self):
        baseline = self.retrieve_baseline_params()
        j = 0
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = baseline[j]
                j = j + 1

    def store_fis(self):
        baselineparams = self.retrieve_baseline_params()
        self.sim_fi_store = pd.DataFrame([])
        self.param_store = pd.DataFrame({"param": sum(free_params.values(), []), "baseline": baselineparams})
        for cell_n in range(0, self.n_stimcells):
            newparams = self.find_bestcandidate()
            newparamdf = pd.DataFrame({"Cell_%s" % cell_n: newparams})
            self.build_optimizedcell()
            newcellfi = self.sim_fi()
            self.sim_fi_store = pd.concat([newcellfi, self.sim_fi_store])
            self.param_store = pd.concat([self.param_store, newparamdf], axis=1)
            self.revert_to_baseline()
        self.sim_fi_store.to_csv('data/parameters/simFIs_%s.csv' % self.flag)
        self.param_store.to_csv('data/parameters/parameters_%s.csv' % self.flag)

        return self.sim_fi_store

    def compute_avgFI(self):
        sim_fi_store = self.store_fis()
        aggregate = sim_fi_store.groupby(['I']).agg({'F': ['mean']})
        stdv = sim_fi_store.groupby(['I']).agg({'F': ['std']})
        self.avg = aggregate.F['mean'].values
        self.sem = stdv.F['std'].values / np.sqrt(self.n_stimcells)
        self.avg_FI = np.c_[np.linspace(0, 0.33, 12), self.avg, self.sem]

        return self.avg_FI

    def return_summarydata(self):
        currentvals = np.linspace(0, 0.33, 12)
        baselineparams = self.retrieve_baseline_params()
        baselinecellfi = self.sim_fi().to_numpy()
        exp_fi = self.data_fi()
        avg_fi = self.compute_avgFI()

        fig1 = plt.figure("fivsfi")
        plt.plot(baselinecellfi[:, 0], baselinecellfi[:, 1], color='0.7', linestyle='dashed', label='Baseline')
        plt.errorbar(exp_fi[:, 0], exp_fi[:, 1], yerr=exp_fi[:, 2], color='0.5', label='Data')
        plt.errorbar(avg_fi[:, 0], avg_fi[:, 1], yerr=avg_fi[:, 2], color='0.0', label='Optimized')
        plt.xlabel("Current (nA)")
        plt.ylabel("Frequency (Hz)")
        plt.legend(loc="upper left")
        fig1.savefig('figures/op-output/fivsfi_simavg_%s.pdf' % self.flag)

        '''
        fig2, axis = plt.subplots(12, 1)
        j = 0
        for currentval in currentvals:
            baselinecell = self.curr_inj(currentval)
            axis[j].plot(baselinecell['t'], baselinecell['V'], color='0.0', label='$%.2f nA$' % currentval)
            box = axis[j].get_position()
            axis[j].set_position([box.x0, box.y0, box.width * 0.7, box.height])
            axis[j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis[j].yaxis.set_visible(False)
            j = j + 1
        fig2.supxlabel('Time (ms)')
        fig2.supylabel('Membrane Potential')
        fig2.suptitle('Baseline: %s' % self.flag)
        fig2.savefig('figures/op-output/baseline_%s.pdf' % self.flag)
        plt.close(fig2)

        newparams = self.find_bestcandidate()
        newcell = self.build_optimizedcell()
        newcellfi = self.sim_fi()

        fig3, axis = plt.subplots(12, 1)
        i = 0
        for currentval in currentvals:
            newcell = self.curr_inj(currentval)
            axis[i].plot(newcell['t'], newcell['V'], color='0.0', label='$%.2f nA$' % currentval)
            box = axis[i].get_position()
            axis[i].set_position([box.x0, box.y0, box.width * 0.7, box.height])
            axis[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis[i].yaxis.set_visible(False)
            i = i + 1
        fig3.supxlabel('Time (ms)')
        fig3.supylabel('Membrane Potential')
        fig3.suptitle('Optimized: %s' % self.flag)
        fig3.savefig('figures/op-output/optimized_%s.pdf' % self.flag)
        plt.close(fig3)
        '''

op_hcc = optimizeparams(gc, free_params, rawhc, 'HC', 'CTRL', 'MSE')
op_hcc.return_summarydata()

op_hcl = optimizeparams(gc, free_params, rawhc, 'HC', 'LITM', 'MSE')
op_hcl.return_summarydata()

op_lrc = optimizeparams(gc, free_params, rawlr, 'LR', 'CTRL', 'MSE')
op_lrc.return_summarydata()

op_lrl = optimizeparams(gc, free_params, rawlr, 'LR', 'LITM', 'MSE')
op_lrl.return_summarydata()

op_nrc = optimizeparams(gc, free_params, rawnr, 'NR', 'CTRL', 'MSE')
op_nrc.return_summarydata()

op_nrl = optimizeparams(gc, free_params, rawnr, 'NR', 'LITM', 'MSE')
op_nrl.return_summarydata()