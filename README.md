# Biophysical Model of Dentate Gyrus 

This is a reimplementation of the dentate gyrus model from which the following papers are based: 

1. Santhakumar et al. (2005) 
2. Yim et al. (2015) 

Our model is modified primarily from the Yim et al. (2005) implementation. Changes are listed at the end of this file.  

## Contributing 

Please see the `CONTRIBUTING.md` file.

## Usage 

Scripts were tested on Windows 10 and Arch Linux. 

__I recommend using Linux, and using the shell scripts, since this will create the necessary results/output directories.__

### Unix machines 

``` bash
sh run.sh

```
### Windows 

Note that here you will need to create the directories to store results. See `run.sh` for the directory names.

```powershell
nrnivmodl mods 
.\main.hoc 
```

### Writing a Script 

```hoc 
...[load templates here (see main.hoc) for example ]...

strdef netlabel
netlabel = "MyDG"

random_seed = 1

objref dg
dg = new DentateGyrus(netlabel, random_seed)
dg.run() 

```

## Changes 

Here we do not list all specific changes implemented, but rather a high-level overview of how the Yim et al. (2015) code was modified. 

1. Network is now encapsulated as a `template` to facilitate batch simulations 
2. NEURON code was (largely) rewritten to achieve several goals
	- Making code more modular, packing routines into smaller functions 
	- Making variable names more expressive (still work to do here) 
	- Attempting to maximize local scope for variables, rather than allowing for global definitions 
3. Development of functions to "transform" network in order to test different effects
4. Analysis code written in Julia  
5. Added capability for spontaneous action potential discharge of granule cells 
