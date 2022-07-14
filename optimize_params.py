#imports
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('Agg') #hopefully this works over ssh 
import matplotlib.pyplot as plt
import pylab              
from random import Random #TODO replace with numpy rand f'n.  pseudorandom number generation
from inspyred import ec   # evolutionary algorithm
from netpyne import specs, sim   # neural network design and simulation
from clamps import IClamp
#from find-rheobase import ElectrophysiologicalPhenotype
from scipy.signal import find_peaks
from tabulate import tabulate
from FI_fromdata import extractFI

#from IVdata import IVdata # import voltage clamp data from simulated GC 

netparams = specs.NetParams()

#import granule cell info from hoc 
gc = netparams.importCellParams(
    label='GC',
    conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
    fileName="objects/GC.hoc",
    cellName="GranuleCell",
    cellArgs=[1], 
    importSynMechs=False
)

#parameters to be optimized 
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

#import raw data 
#rawhc = pd.read_csv("rawdata/HC_FI.csv")

class optimizeparams(object):
    def __init__(self, 
                 cell,
                 #celldata,
                 free_params,
                 pop_size = 10,
                 max_evaluations = 20, 
                 num_selected = 10, 
                 mutation_rate = 0.2, 
                 num_elites = 1,
                 targetRate = 25,
                 current = 0.33
                 ):
        
        self.cell_dict = {"secs": cell["secs"]}
        self.free_params = free_params
        #self.FI_curve = celldata
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
        self.current = current

        
    def curr_inj(self, delay=0, duration=1000):
        """ Injects a level of current and returns the number of spikes emitted
        
        Arguments: 
            current: `float`. Amount of current injected [nA]
            delay: `float`. Time after recording starts where current is injected [ms]
            duration: `float`. Total duration of the current injection [ms]
        
        Returns: 
            `dict`. Results of the step current injection simulation
        
        """
        iclamp = IClamp(self.cell_dict, delay=delay, duration=duration, T=duration + delay*2)
        res = iclamp(self.current)
        return res 
    

    def retrieve_baseline_params(self):
        self.baseline = []
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.baseline.append(self.cell_dict['secs']['soma']['mechs'][key][val])
        
        self.num_inputs = len(self.baseline)
        
        return self.baseline 
            
    def generate_netparams(self, random, args):
        self.initialParams = [random.uniform(self.minParamValues[i], self.maxParamValues[i]) for i in range(self.num_inputs)]
        self.initialParams
        return self.initialParams
        
    # design fitness function, used in the ec evolve function --> final_pop = my_ec.evolve(...,evaluator=evaluate_netparams,...)
    def evaluate_netparams(self, candidates, args):
        self.fitnessCandidates = []

        for icand,cand in enumerate(candidates):
            #TODO find way to use free_params here to remove this ugly situation
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
            
            clamp = self.curr_inj(self.current)
            
            #find number of spikes, 'rate' is highly inaccurate for spikes with peaks < ~10mV
            spikes = find_peaks(clamp['V'], 0) 
            num_spikes = len(spikes[0]) 
            
            #targetRate = 45
            fitness = abs(self.targetRate - num_spikes)
            self.fitnessCandidates.append(fitness)
        
        return self.fitnessCandidates
    
    def find_bestcandidate(self):
        rand = Random()
        rand.seed(1)
        
        #TODO find biologically plausible vals for min max bounds        
        self.minParamValues = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.maxParamValues = [0.0001, 0.0001, 0.001, 0.6, 0.005, 0.01, 1.44e-05, 7.22e-05, 0.05, 0.0001, 15.0, 0.012, 0.1, 0.005, 0.01, 0.004, 0.05, 6e-05]
        
        self.gc_ec = ec.EvolutionaryComputation(rand)
        self.gc_ec.selector = ec.selectors.tournament_selection
        self.gc_ec.variator = [ec.variators.uniform_crossover,   # biased coin flip to determine whether 'mom' or 'dad' element is passed to offspring design
                         ec.variators.gaussian_mutation] 
        self.gc_ec.replacer = ec.replacers.generational_replacement  
        self.gc_ec.terminator = ec.terminators.evaluation_termination
        
        self.gc_ec.observer = ec.observers.plot_observer#[ec.observers.stats_observer,  # print evolutionary computation statistics
                  #ec.observers.plot_observer,   # plot output of the evolutionary computation as graph
                  #ec.observers.file_observer,
                  #ec.observers.best_observer]
                
        self.final_pop = self.gc_ec.evolve(generator=self.generate_netparams,  # assign design parameter generator to iterator parameter generator
                              evaluator=self.evaluate_netparams,     # assign fitness function to iterator evaluator
                              pop_size=self.pop_size,                      # each generation of parameter sets will consist of 10 individuals
                              maximize=False,                   # best fitness corresponds to minimum value
                              bounder=ec.Bounder(self.minParamValues, self.maxParamValues), # boundaries for parameter set ([probability, weight, delay])
                              max_evaluations=self.max_evaluations,               # evolutionary algorithm termination at 50 evaluations
                              num_selected=self.num_selected,                  # number of generated parameter sets to be selected for next generation
                              mutation_rate=self.mutation_rate,                # rate of mutation
                              num_inputs=self.num_inputs,                     # len([probability, weight, delay])
                              num_elites=self.num_elites)                     # 1 existing individual will survive to next generation if it has better fitness than an individual selected by the tournament selection

        self.final_pop.sort(reverse=True)                            # sort final population so best fitness (minimum difference) is first in list
        self.bestCand = self.final_pop[0].candidate                       # bestCand <-- individual @ start of list

        plt.savefig('figures/op_output/observer.png')
        return self.bestCand
    
    def build_optimizedcell(self):
        j = 0 
        
        for key in self.free_params.keys():
            for val in self.free_params[key]:
                self.cell_dict['secs']['soma']['mechs'][key][val] = self.bestCand[j]
                j = j + 1

        finalclamp = self.curr_inj(self.current)
        
        return finalclamp 
    
    def return_summarydata(self):
        
        baselineparams = self.retrieve_baseline_params()
        baselinecell = self.curr_inj(self.current)
        
        newparams = self.find_bestcandidate()
        newcell = self.build_optimizedcell()
        
        
        diffs = [x1 - x2 for (x1, x2) in zip(newparams, baselineparams)]

        paramnames = sum(free_params.values(), [])

        headers = ['paramname', 'baseline', 'optimized', 'diff']

        summtable = zip(paramnames, baselineparams, newparams, diffs)
        
        with open ('data/parameters/sumtable.txt', 'w') as f:    
           f.write(tabulate(summtable, headers=headers))
        
        figure, axis = plt.subplots(2,1, constrained_layout=True)

        axis[0].plot(baselinecell['t'], baselinecell['V'])
        axis[0].set_title("Baseline cell")

        axis[1].plot(newcell['t'], newcell['V'])
        axis[1].set_title("Optimized cell")
        
        figure.savefig('figures/op_output/baseline_optimized.png')
    

op = optimizeparams(gc, free_params)
op.return_summarydata() 




