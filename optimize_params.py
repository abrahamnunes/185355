#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:32:02 2022

evol'n algorithm to optimize params
- building from tutorial file first (which is why this is gross to look at)
testing with small subset of data first. 

@author: selenasingh

GOAL [June 28]: optimize scaling conductance vals to shift I-V curve of one granule cell to fit I-V curve data from ONE cell (starting with HC, NA currents only for now)
NEXT GOAL [July 5]: Use average of all Na, K currents, HC first. Will require transforming tables provided by AK 

"""

#imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pylab              
from random import Random # pseudorandom number generation
from inspyred import ec   # evolutionary algorithm
from netpyne import sim   # neural network design and simulation
from IVdata import IVdata # import voltage clamp data from simulated GC 

netparams = specs.NetParams()

#import granule cell info from hoc 
gc = netparams.importCellParams(
    label='GC',
    conds={"cellType": "GranuleCell", "cellModel": "GranuleCell"},
    fileName="objects/GC.hoc",
    cellName="GranuleCell",
    cellArgs=[1], # [CHECK] what does this mean?
    importSynMechs=False
)

class optimizeparams(object):
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


    def generate_netparams(self, random, args): 
        size = args.get('num_inputs') 
        initialParams = [random.uniform(minParamValues[i], maxParamValues[i]) for i in range(size)]
        return initialParams

# design fitness function, used in the ec evolve function --> final_pop = my_ec.evolve(...,evaluator=evaluate_netparams,...)
    def evaluate_netparams(candidates, args):
        fitnessCandidates = []

        for icand,cand in enumerate(candidates):
            # OUTLINE CANDIDATE PARAMS 
            
            # create cell: UPDATE THIS
            sim.createSimulate(netParams=tut2.netParams, simConfig=tut2.simConfig)
    
            # calculate iv curve
            IV = IVdata(gc)
            cellIV = IV.compute_ivdata(vlow = -70, vhigh = 20, n_steps = 10, delay = 250, duration = 500)
    
            # calculate fitness for this candidate
            fitness = abs(targetIV - cellIV['Na'])  # minimize absolute difference in currents/voltage step 
    
            # add to list of fitness for each candidate
            fitnessCandidates.append(fitness)
    
            # print candidate parameters, IV curves
            # print('\n CHILD/CANDIDATE %d: Cell with ...)
            #%(icand, cand[0], cand[1], cand[2], cellIV, fitness))


        return fitnessCandidates




'''
#main

# create random seed for evolutionary computation algorithm --> my_ec = ec.EvolutionaryComputation(rand)
rand = Random()
rand.seed(1)

# target mean firing rate in Hz
 targetIV = ... #IMPORT FROM FILE 

# min and max allowed value for each param optimized: 
#                 prob, weight, delay
minParamValues = [0.01,  0.001,  1]
maxParamValues = [0.5,   0.1,   20]

# instantiate evolutionary computation algorithm with random seed
my_ec = ec.EvolutionaryComputation(rand)

# establish parameters for the evolutionary computation algorithm, additional documentation can be found @ pythonhosted.org/inspyred/reference.html
my_ec.selector = ec.selectors.tournament_selection  # tournament sampling of individuals from population (<num_selected> individuals are chosen based on best fitness performance in tournament)

#toggle variators
my_ec.variator = [ec.variators.uniform_crossover,   # biased coin flip to determine whether 'mom' or 'dad' element is passed to offspring design
                 ec.variators.gaussian_mutation]    # gaussian mutation which makes use of bounder function as specified in --> my_ec.evolve(...,bounder=ec.BOunder(minParamValues, maxParamValues),...)

my_ec.replacer = ec.replacers.generational_replacement    # existing generation is replaced by offspring, with elitism (<num_elites> existing individuals will survive if they have better fitness than offspring)

my_ec.terminator = ec.terminators.evaluation_termination  # termination dictated by number of evaluations that have been run

#toggle observers
my_ec.observer = [ec.observers.stats_observer,  # print evolutionary computation statistics
                  ec.observers.plot_observer,   # plot output of the evolutionary computation as graph
                  ec.observers.best_observer]   # print the best individual in the population to screen

#call evolution iterator
final_pop = my_ec.evolve(generator=generate_netparams,  # assign design parameter generator to iterator parameter generator
                      evaluator=evaluate_netparams,     # assign fitness function to iterator evaluator
                      pop_size=10,                      # each generation of parameter sets will consist of 10 individuals
                      maximize=False,                   # best fitness corresponds to minimum value
                      bounder=ec.Bounder(minParamValues, maxParamValues), # boundaries for parameter set ([probability, weight, delay])
                      max_evaluations=50,               # evolutionary algorithm termination at 50 evaluations
                      num_selected=10,                  # number of generated parameter sets to be selected for next generation
                      mutation_rate=0.2,                # rate of mutation
                      num_inputs=3,                     # len([probability, weight, delay])
                      num_elites=1)                     # 1 existing individual will survive to next generation if it has better fitness than an individual selected by the tournament selection

#configure plotting
pylab.legend(loc='best')
pylab.show()
'''

