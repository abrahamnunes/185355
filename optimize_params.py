#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:32:02 2022

evol'n algorithm to optimize params
testing with small subset of data first. 

@author: selenasingh

July 4: successfully tuned 1 gc to fire at specific freq.

- figure out how to deal with the parameters in a nice way
"""

#imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pylab              
from random import Random # pseudorandom number generation
from inspyred import ec   # evolutionary algorithm
from netpyne import specs, sim   # neural network design and simulation
from clamps import IClamp
#from find-rheobase import ElectrophysiologicalPhenotype
from scipy.signal import find_peaks
from tabulate import tabulate

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

#import raw data 
#FI_curve = pd.read_csv("rawdata/firsttest.csv")

class optimizeparams(object):
    def __init__(self, 
                 cell,
                 num_inputs = 18, #number of parameters to be optimized
                 pop_size = 10,
                 max_evaluations = 100, 
                 num_selected = 10, 
                 mutation_rate = 0.2, 
                 num_elites = 1,
                 targetRate = 25,
                 current = 0.33):
        
        self.cell_dict = {"secs": cell["secs"]}
        self.iv_interm = {}
        self.initialParams = []
        self.iv_data = None 
        self.minParamValues = []
        self.maxParamValues = []
        self.num_inputs = num_inputs
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
        baseline = [0]*self.num_inputs
        
        baseline[0] =  self.cell_dict['secs']['soma']['mechs']['HT']['gbar']
        baseline[1] = self.cell_dict['secs']['soma']['mechs']['LT']['gbar'] 
        baseline[2] = self.cell_dict['secs']['soma']['mechs']['bk']['gkbar'] 
        baseline[3] = self.cell_dict['secs']['soma']['mechs']['ichan2']['gnatbar'] 
        baseline[4] = self.cell_dict['secs']['soma']['mechs']['ichan2']['gkfbar'] 
        baseline[5] = self.cell_dict['secs']['soma']['mechs']['ichan2']['gksbar'] 
        baseline[6] = self.cell_dict['secs']['soma']['mechs']['ichan2']['gl'] 
        baseline[7] = self.cell_dict['secs']['soma']['mechs']['ichan2']['ggabaa'] 
        baseline[8] = self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] 
        baseline[9] = self.cell_dict['secs']['soma']['mechs']['kir']['gkbar'] 
        baseline[10] = self.cell_dict['secs']['soma']['mechs']['kir']['kl'] 
        baseline[11] = self.cell_dict['secs']['soma']['mechs']['kir']['at'] 
        baseline[12] = self.cell_dict['secs']['soma']['mechs']['kir']['bt'] 
        baseline[13] = self.cell_dict['secs']['soma']['mechs']['km']['gbar'] 
        baseline[14] = self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] 
        baseline[15] = self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] 
        baseline[16] = self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] 
        baseline[17] = self.cell_dict['secs']['soma']['mechs']['tca']['gcatbar']
        
        return baseline
            

    def generate_netparams(self, random, args):
        self.initialParams = [random.uniform(self.minParamValues[i], self.maxParamValues[i]) for i in range(self.num_inputs)]
        self.initialParams
        return self.initialParams
        
# design fitness function, used in the ec evolve function --> final_pop = my_ec.evolve(...,evaluator=evaluate_netparams,...)
    def evaluate_netparams(self, candidates, args):
        self.fitnessCandidates = []

        for icand,cand in enumerate(candidates):
            
            # [TODO] find a better way to do this:
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
            
            ## find number of spikes, 'rate' is highly inaccurate for spikes with peaks < ~10mV
            spikes = find_peaks(clamp['V'], 0) 
            num_spikes = len(spikes[0]) 
            
            #targetRate = 45
            fitness = abs(self.targetRate - num_spikes)
            self.fitnessCandidates.append(fitness)
        
        return self.fitnessCandidates
    
    def find_bestcandidate(self):
        rand = Random()
        rand.seed(1)
        
                                ## [TODO] find biologically plausible vals for these min max bounds.
        #self.minParamValues = [0.05, 0.01, 0.001] #0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #self.maxParamValues = [0.6, 0.005, 0.01] #0.6, 0.06, 0.01, 1.44e-05, 7.22e-05, 0.05, 15.0, 0.007, 0.1, 0.005, 0.01, 0.004, 0.05, 6e-05]
        
        self.minParamValues = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.maxParamValues = [0.0001, 0.0001, 0.001, 0.6, 0.005, 0.01, 1.44e-05, 7.22e-05, 0.05, 0.0001, 15.0, 0.012, 0.1, 0.005, 0.01, 0.004, 0.05, 6e-05]
        
        self.gc_ec = ec.EvolutionaryComputation(rand)
        self.gc_ec.selector = ec.selectors.tournament_selection
        self.gc_ec.variator = [ec.variators.uniform_crossover,   # biased coin flip to determine whether 'mom' or 'dad' element is passed to offspring design
                         ec.variators.gaussian_mutation] 
        self.gc_ec.replacer = ec.replacers.generational_replacement  
        self.gc_ec.terminator = ec.terminators.evaluation_termination
        
        self.gc_ec.observer = [ec.observers.stats_observer,  # print evolutionary computation statistics
                  ec.observers.plot_observer,   # plot output of the evolutionary computation as graph
                  ec.observers.best_observer] 
        
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

        return self.bestCand
    
    def build_optimizedcell(self):
        
        self.cell_dict['secs']['soma']['mechs']['HT']['gbar'] = self.bestCand[0]
        self.cell_dict['secs']['soma']['mechs']['LT']['gbar'] = self.bestCand[1]
        self.cell_dict['secs']['soma']['mechs']['bk']['gkbar'] = self.bestCand[2]
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gnatbar'] = self.bestCand[3]
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gkfbar'] = self.bestCand[4]
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gksbar'] = self.bestCand[5]
        self.cell_dict['secs']['soma']['mechs']['ichan2']['gl'] = self.bestCand[6]
        self.cell_dict['secs']['soma']['mechs']['ichan2']['ggabaa'] = self.bestCand[7]
        self.cell_dict['secs']['soma']['mechs']['ka']['gkabar'] = self.bestCand[8]
        self.cell_dict['secs']['soma']['mechs']['kir']['gkbar'] = self.bestCand[9]
        self.cell_dict['secs']['soma']['mechs']['kir']['kl'] = self.bestCand[10]
        self.cell_dict['secs']['soma']['mechs']['kir']['at'] = self.bestCand[11]
        self.cell_dict['secs']['soma']['mechs']['kir']['bt'] = self.bestCand[12]
        self.cell_dict['secs']['soma']['mechs']['km']['gbar'] = self.bestCand[13]
        self.cell_dict['secs']['soma']['mechs']['lca']['glcabar'] = self.bestCand[14]
        self.cell_dict['secs']['soma']['mechs']['nca']['gncabar'] = self.bestCand[15]
        self.cell_dict['secs']['soma']['mechs']['sk']['gskbar'] = self.bestCand[16]
        self.cell_dict['secs']['soma']['mechs']['tca']['gcatbar'] = self.bestCand[17]
        
        finalclamp = self.curr_inj(self.current)
        #findspikes = find_peaks(finalclamp['V'],0)
        #finalfreq = len(findspikes[0])
        
        return finalclamp #, finalfreq
    
    

  

#----------run simulations 
init = optimizeparams(gc)
baselineparams = init.retrieve_baseline_params()
baselinecell = init.curr_inj(0.33)

op = optimizeparams(gc)
newparams = op.find_bestcandidate()
newcell = op.build_optimizedcell()

#----------CONSTRUCT PARAM COMPARISON TABLE 
diffs = [x1 - x2 for (x1, x2) in zip(newparams, baselineparams)]

paramnames = ['HTgbar', 'LTgbar', 'bkgbar', 'gnatbar', 'gkfbar', 'gksbar', 'gl', 'ggabaa', 'gkabar', 'gkbar', 'kl', 'at', 'bt', 'gkmbar', 'glcabar', 'gncabar', 'gskbar', 'gcatbar']

headers = ['paramname', 'baseline', 'optimized', 'diff']

summtable = zip(paramnames, baselineparams, newparams, diffs)

print(tabulate(summtable, headers=headers))


#----------PLOTTING
figure, axis = plt.subplots(2,1, constrained_layout=True)

axis[0].plot(baselinecell['t'], baselinecell['V'])
axis[0].set_title("Baseline cell")

axis[1].plot(newcell['t'], newcell['V'])
axis[1].set_title("Optimized cell")






