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

# Get FI Curves

# Run DG Model 
arch -arch x86_64 nrnivmodl mods #this works w apple m1
python3 optimize_params.py
#arch -arch x86_64 nrniv main.hoc

# Plot Network Structure

# Analyze Data 
#julia analysis.jl


