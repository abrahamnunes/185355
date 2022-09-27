# Make directory structure for outputs
mkdir data
mkdir data/dgnetwork 
mkdir data/fi-curves
mkdir data/parameters

mkdir figures 
mkdir figures/raster-plots
mkdir figures/pattern-separation
mkdir figures/voltage-tracings
mkdir figures/fi-curves
mkdir figures/op-output
mkdir figures/manual-adjust

# Get FI Curves

# Run DG Model 
#if wanting to run natively on m1
nrnivmodl mods
#python optimize_params.py
#python computefoldchange.py
nrniv main.hoc

#if wanting to run using Rosetta
#arch -arch x86_64 nrnivmodl mods
#arch -arch x86_64 nrniv main.hoc

# Plot Network Structure

# Analyze Data 
julia analysis.jl


