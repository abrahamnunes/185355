# imports
import numpy as np
import pandas as pd
import matplotlib

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
    'HT': ['gbar'],
    'LT': ['gbar'],
    'bk': ['gkbar'],
    'ichan2': ['gnatbar', 'gkfbar', 'gksbar', 'gl', 'ggabaa'],
    'ka': ['gkabar'],
    'kir': ['gkbar', 'kl', 'at', 'bt'],
    'km': ['gbar'],
    'lca': ['glcabar'],
    'nca': ['gncabar'],
    'sk': ['gskbar'],
    'tca': ['gcatbar']
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
                 difference_method,  # str
                 pop_size=10,
                 max_evaluations=170,
                 num_selected=10,
                 mutation_rate=0.1,
                 num_elites=1,
                 targetRate=12,
                 current=0.33
                 ):

        self.cell_dict = {"secs": cell["secs"]}
        self.free_params = free_params
        self.FI_curve = celldata
        self.condition = condition
        self.population = population
        self.difference_method = difference_method
        self.initialParams = []
        self.minParamValues = []
        self.maxParamValues = []
        self.num_inputs = 18
        self.free_params = free_params
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.num_selected = num_selected
        self.mutation_rate = mutation_rate
        self.num_elites = num_elites
        self.targetRate = targetRate
        self.flag = str(self.population + '_' + self.condition + '_' + self.difference_method)
        # self.current = current

    def curr_inj(self, current, delay=0, duration=1000):
        iclamp = IClamp(self.cell_dict, delay=delay, duration=duration, T=duration + delay * 2)
        res = iclamp(current)
        return res

    def retrieve_baseline_params(self):
        self.baseline = []
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.baseline.append(self.cell_dict['secs']['soma']['mechs'][key][val])
        self.num_inputs = len(self.baseline)

        return self.baseline

    def sim_fi(self):
        ep = ElectrophysiologicalPhenotype(self.cell_dict)
        self.simfi_df = ep.compute_fi_curve(ilow=0, ihigh=0.33, n_steps=24, delay=0, duration=1000)
        self.simfi = self.simfi_df.to_numpy()
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
            self.cell_dict['secs']['soma']['mechs']['HT']['gbar'] = cand[0]
            self.cell_dict['secs']['soma']['mechs']['LT']['gbar'] = cand[1]
            self.cell_dict['secs']['soma']['mechs']['bk']['gkbar'] = cand[2]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gnatbar'] = cand[3]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gkfbar'] = cand[4]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gksbar'] = cand[5]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['gl'] = cand[6]
            self.cell_dict['secs']['soma']['mechs']['ichan2']['ggabaa'] = cand[7]
            self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] = cand[8]
            self.cell_dict['secs']['soma']['mechs']['kir']['gkbar'] = cand[9]
            self.cell_dict['secs']['soma']['mechs']['kir']['kl'] = cand[10]
            self.cell_dict['secs']['soma']['mechs']['kir']['at'] = cand[11]
            self.cell_dict['secs']['soma']['mechs']['kir']['bt'] = cand[12]
            self.cell_dict['secs']['soma']['mechs']['km']['gbar'] = cand[13]
            self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] = cand[14]
            self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] = cand[15]
            self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] = cand[16]
            self.cell_dict['secs']['soma']['mechs']['tca']['gcatbar'] = cand[17]

            # clamp = self.curr_inj(self.current)
            FI_data = self.data_fi()
            FI_sim = self.sim_fi()

            # find number of spikes, 'rate' is highly inaccurate for spikes with peaks < ~10mV
            # spikes = find_peaks(clamp['V'], 0)
            # num_spikes = len(spikes[0])
            if self.difference_method == "Area":
                fitness = abs(similaritymeasures.area_between_two_curves(FI_data, FI_sim))
            elif self.difference_method == "Frechet":
                fitness = abs(similaritymeasures.frechet_dist(FI_data, FI_sim))
            elif self.difference_method == "CL":
                fitness = abs(similaritymeasures.curve_length_measure(FI_data, FI_sim))
            elif self.difference_method == "PCM":
                fitness = abs(similaritymeasures.pcm(FI_data, FI_sim))
            elif self.difference_method == "DTW":
                fitness = abs(similaritymeasures.dtw(FI_data, FI_sim)[0])
            # fitness = abs(self.targetRate - num_spikes)
            self.fitnessCandidates.append(fitness)

        return self.fitnessCandidates

    def find_bestcandidate(self):
        rand = Random()
        rand.seed(1)

        # TODO find biologically plausible vals for min max bounds
        self.minParamValues = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.maxParamValues = [0.0001, 0.0001, 0.001, 0.6, 0.005, 0.01, 1.44e-05, 7.22e-05, 0.05, 0.0001, 15.0, 0.012,
                               0.1, 0.005, 0.01, 0.004, 0.05, 6e-05]

        self.gc_ec = ec.EvolutionaryComputation(rand)
        self.gc_ec.selector = ec.selectors.tournament_selection
        self.gc_ec.variator = [ec.variators.uniform_crossover,
                               # biased coin flip to determine whether 'mom' or 'dad' element is passed to offspring design
                               ec.variators.gaussian_mutation]
        self.gc_ec.replacer = ec.replacers.generational_replacement
        self.gc_ec.terminator = ec.terminators.evaluation_termination
        self.gc_ec.observer = ec.observers.plot_observer
        #self.gc_ec.observer = ec.observers.file_observer

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
                                           # evolutionary algorithm termination at 50 evaluations
                                           num_selected=self.num_selected,
                                           # number of generated parameter sets to be selected for next generation
                                           mutation_rate=self.mutation_rate,  # rate of mutation
                                           num_inputs=self.num_inputs,  # len([probability, weight, delay])
                                           num_elites=self.num_elites)  # 1 existing individual will survive to next generation if it has better fitness than an individual selected by the tournament selection

        self.final_pop.sort(reverse=True)  # sort final population so best fitness (minimum difference) is first in list
        self.bestCand = self.final_pop[0].candidate  # bestCand <-- individual @ start of list

        plt.savefig('figures/op-output/observer_%s.png' % self.flag)
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
        self.fi = self.ep_opt.compute_fi_curve(ilow=0, ihigh=0.33, n_steps=12, delay=0, duration=1000)

        return finalclamp

    def return_summarydata(self):
        currentvals = np.linspace(0, 0.33, 12)
        baselineparams = self.retrieve_baseline_params()
        baselinecellfi = self.sim_fi()

        fig1, axis = plt.subplots(12, 1)
        j = 0
        for currentval in currentvals:
            baselinecell = self.curr_inj(currentval)
            axis[j].plot(baselinecell['t'], baselinecell['V'], label='$%.2f nA$' % currentval)
            box = axis[j].get_position()
            axis[j].set_position([box.x0, box.y0, box.width * 0.7, box.height])
            axis[j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis[j].yaxis.set_visible(False)
            j = j + 1
        fig1.supxlabel('Time (ms)')
        fig1.supylabel('Membrane Potential')
        fig1.suptitle('Baseline: %s' % self.flag)
        fig1.savefig('figures/op-output/baseline_%s.png' % self.flag)
        plt.close(fig1)

        newparams = self.find_bestcandidate()
        newcell = self.build_optimizedcell()
        newcellfi = self.sim_fi()

        fig2, axis = plt.subplots(12, 1)
        i = 0
        for currentval in currentvals:
            newcell = self.curr_inj(currentval)
            axis[i].plot(newcell['t'], newcell['V'], label='$%.2f nA$' % currentval)
            box = axis[i].get_position()
            axis[i].set_position([box.x0, box.y0, box.width * 0.7, box.height])
            axis[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axis[i].yaxis.set_visible(False)
            i = i + 1
        fig2.supxlabel('Time (ms)')
        fig2.supylabel('Membrane Potential')
        fig2.suptitle('Optimized: %s' % self.flag)
        fig2.savefig('figures/op-output/optimized_%s.png' % self.flag)
        plt.close(fig2)

        exp_fi = self.data_fi()

        diffs = [x1 - x2 for (x1, x2) in zip(newparams, baselineparams)]
        paramnames = sum(free_params.values(), [])
        headers = ['paramname', 'baseline', 'optimized', 'diff']
        summtable = zip(paramnames, baselineparams, newparams, diffs)

        with open('data/parameters/sumtable_%s.txt' % self.flag, 'w') as f:
            f.write(tabulate(summtable, headers=headers))

        fig3 = plt.figure("fivsfi")
        plt.plot(baselinecellfi[:, 0], baselinecellfi[:, 1], label='Baseline')
        plt.plot(newcellfi[:, 0], newcellfi[:, 1], label='Optimized')
        plt.plot(exp_fi[:, 0], exp_fi[0:, 1], label='Data')
        plt.xlabel("Current (nA)")
        plt.ylabel("Number of Spikes")
        plt.legend(loc="upper left")
        fig3.savefig('figures/op-output/fivsfi_%s.png' % self.flag)



#op_c = optimizeparams(gc, free_params, rawhc, 'HC', 'CTRL', 'Area')
#op_c.return_summarydata()

#op_li = optimizeparams(gc, free_params, rawhc, 'HC', 'LITM','Area')
#op_li.return_summarydata()

#op_lr_ctrl = optimizeparams(gc, free_params, rawnr, 'NR', 'CTRL', 'PCM')
#op_lr_ctrl.return_summarydata()

op_lr_litm = optimizeparams(gc, free_params, rawlr, 'LR', 'CTRL', 'DTW')
op_lr_litm.return_summarydata()