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
from random import Random  # TODO replace with numpy rand f'n.  pseudorandom number generation
from inspyred import ec  # evolutionary algorithm
from netpyne import specs, sim  # neural network design and simulation
from clamps import IClamp
from clamps_noise import ICNoise
from find_rheobase import ElectrophysiologicalPhenotype
from FI_fromdata import extractFI
import similaritymeasures
import random

netparams = specs.NetParams()


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
               'gkfbar', 'gksbar', 'gl'],  # KDR channel conductances, sodium conductance
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
                 max_evaluations=100,
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
        self.num_inputs = 20
        self.free_params = free_params
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.num_selected = num_selected
        self.mutation_rate = mutation_rate
        self.num_elites = num_elites
        self.targetRate = targetRate
        self.flag = str(self.population + '_' + self.condition + '_' + self.difference_method)
        self.n_stimcells = 10
        self.noiselvl = random.uniform(0.8, 1)
        self.weight = 0.0001  # random.uniform(0.01, 0.1)

    def curr_inj(self, current, delay=0, duration=1000, noise=False):
        if noise:
            iclamp = ICNoise(self.cell_dict, delay=delay, duration=duration, T=duration + delay * 2)
            res = iclamp(current, self.noiselvl, self.weight)
        else:
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

    def sim_fi(self, noise):
        ep = ElectrophysiologicalPhenotype(self.cell_dict, noise=noise)
        self.simfi = ep.compute_fi_curve(ilow=0, ihigh=0.33, n_steps=12, delay=0, duration=1000)
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
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftma'] = cand[2]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftmb'] = cand[3]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftha'] = cand[4]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshifthb'] = cand[5]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnfa'] = cand[6]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnfb'] = cand[7]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnsa'] = cand[8]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['vshiftnsb'] = cand[9]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gkfbar'] = cand[10]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gksbar'] = cand[11]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gl'] = cand[12]
            self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] = cand[13]
            self.cell_dict['secs']['soma']['mechs']['kir']['gkbar'] = cand[14]
            self.cell_dict['secs']['soma']['mechs']['km']['gbar'] = cand[15]
            self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] = cand[16]
            self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] = cand[17]
            self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] = cand[18]
            self.cell_dict['secs']['soma']['mechs']['tca']['gcatbar'] = cand[19]

            FI_data = self.data_fi()
            FI_sim = self.sim_fi(noise=False).to_numpy()

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

        normal_vals = [0.0006, 0.12, 43.0, 15.0, 65.0, 12.5, 18.0, 43.0, 30.0, 55.0, 0.016, 0.006, 1.44e-05, 0.012, 0.0,
                       0.001, 0.005, 0.002, 0.001, 3.7e-5]
        # consistent with hoc file, removed HT, LT, GABA-A, Kir initialized to 0, so will stay 0
        # self.minParamValues = list(np.array(normal_vals) * 0.2)  # allow min vals to be 80% lower
        # self.maxParamValues = list(np.array(normal_vals) * 1.8)  # allow max vals to be 80% higher

        self.minParamValues = [0.00012, 0.024, 39.99, 13.95, 60.45, 11.625, 12.0, 35.0, 25.0, 50.0, 0.0032, 0.0012,
                               0.00000288, 0.0024, 0, 0.0002, 0.001, 0.0004, 0.0002, 0.0000074]

        self.maxParamValues = [0.00108, 0.216, 46.01, 16.05, 79.55, 15.0, 25.0, 50.0, 40.0, 60.0, 0.0288, 0.0108,
                               0.00002592, 0.0216, 0, 0.0018, 0.009, 0.0036, 0.0018, 0.0000666]

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
            # for j, val in zip(range(0,len(self.bestCand)), self.free_params[key]):
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = self.bestCand[j]
                j = j + 1
        finalclamp = self.curr_inj(currentval)
        # self.ep_opt = ElectrophysiologicalPhenotype(self.cell_dict)
        return finalclamp

    def revert_to_baseline(self):
        # TODO: fix this. Doesn't seem to do anything currently.
        j = 0
        for key in self.free_params.keys():
            # for j, val in zip(range(0,len(self.bestCand)), self.free_params[key]):
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = self.baseline[j]
                j = j + 1
        clamptobuild = self.curr_inj(0.33)
        # self.ep_opt = ElectrophysiologicalPhenotype(self.cell_dict)
        return clamptobuild

    def store_fis(self):
        baselineparams = self.retrieve_baseline_params()
        self.sim_fi_store = pd.DataFrame([])
        self.param_store = pd.DataFrame({"param": sum(free_params.values(), []), "baseline": baselineparams})
        for cell_n in range(0, self.n_stimcells):
            newparams = self.find_bestcandidate()
            newparamdf = pd.DataFrame({"Cell_%s" % cell_n: newparams})
            self.build_optimizedcell()
            newcellfi = self.sim_fi(noise=False)
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
        baselinecellfi = self.sim_fi(noise=False).to_numpy()
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

        self.revert_to_baseline()

        fig2, axis = plt.subplots(12, 1)
        i = 0
        for currentval in currentvals:
            newcell = self.curr_inj(currentval, noise=True)
            axis[i].plot(newcell['t'], newcell['V'], color='0.0', label='$%.2f nA$' % currentval)
            box = axis[i].get_position()
            axis[i].set_position([box.x0, box.y0, box.width * 0.7, box.height])
            axis[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis[i].yaxis.set_visible(False)
            i = i + 1
        fig2.supxlabel('Time (ms)')
        fig2.supylabel('Membrane Potential')
        fig2.suptitle('Optimized: %s' % self.flag)
        fig2.savefig('figures/op-output/optimized_%s.pdf' % self.flag)
        plt.close(fig2)


gc = importgc()  # re-loading to ensure baseline params are the same each time
op_hcc = optimizeparams(gc, free_params, rawhc, 'HC', 'CTRL', 'MSE')
op_hcc.return_summarydata()

gc1 = importgc()
op_hcl = optimizeparams(gc1, free_params, rawhc, 'HC', 'LITM', 'MSE')
op_hcl.return_summarydata()

gc2 = importgc()
op_lrc = optimizeparams(gc2, free_params, rawlr, 'LR', 'CTRL', 'MSE')
op_lrc.return_summarydata()

gc3 = importgc()
op_lrl = optimizeparams(gc3, free_params, rawlr, 'LR', 'LITM', 'MSE')
op_lrl.return_summarydata()

gc4 = importgc()
op_nrc = optimizeparams(gc4, free_params, rawnr, 'NR', 'CTRL', 'MSE')
op_nrc.return_summarydata()

gc5 = importgc()
op_nrl = optimizeparams(gc5, free_params, rawnr, 'NR', 'LITM', 'MSE')
op_nrl.return_summarydata()
