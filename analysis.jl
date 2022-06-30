
####################################################################
# SCRIPT TO RUN ANALYSIS 
####################################################################
include("utilities.jl");
default(show=false)

# HYPERPARAMS
n_runs = 10
patterns = 0:12
labels = ["HC", "LR", "NR"]
fig_ext = ".png"

# CREATE NECESSARY DIRECTORIES 
create_directories_if_not_exist()

# IDENTIFY NEURON POPULATION RANGES
populations = Dict(
    "DG" => [0, 500],
    "BC" => [500,506],
    "MC" => [506, 521],
    "HIPP" => [521, 527]
)

for run_ ∈ 1:n_runs
    for i ∈ 1:length(labels)
        spikes = load_spike_files(patterns, labels[i]*"-$run_", populations)

        # CREATE RASTER PLOTS
        for p ∈ unique(spikes.Pattern)
            stimin = spikes[(spikes.Population .== "PP") .& (spikes.Pattern .== p), :]
            plots = []
            append!(plots, [raster_plot(stimin; ylab="PP")])
            for pop ∈ keys(populations)
                lb, ub = populations[pop]
                popspikes = spikes[(spikes.Population .== pop) .& (spikes.Pattern .== p),:]
                append!(plots, [raster_plot(popspikes; xlab="", ylab=pop)])
            end
            fig = plot(reverse(plots)..., layout=grid(5, 1, heights=[0.15, 0.15, 0.15, 0.4, 0.15]), size=(400, 500))
            savefig(fig, "figures/raster-plots/raster-"*string(p)*"-"*labels[i]*"-$run_"*fig_ext)
        end
    end 
end 


# PATTERN SEPARATION CURVES
colors=[:blue, :red, :green]
global psfig = plot([0;1], [0;1], ls=:dash, c=:black, 
                        xlabel="Input Correlation "*L"(r_{in})", 
                        ylabel="Output Correlation "*L"(r_{out})", 
                        size=(300, 300),
                        label=nothing, legend=:topleft)

psc = Dict("HC"=>[], "LR"=>[], "NR"=>[])
for i ∈ 1:length(labels)
    for run ∈ 1:n_runs
        spikes = load_spike_files(patterns, labels[i]*"-$run", populations)
        
        out = pattern_separation_curve(spikes, 100, 500)
        x, y = out[:,"Input Correlation"], out[:, "Output Correlation"]
        
        # Remove NaNs before fitting
        idx_ = (.!isnan.(x) .& .!isnan.(y))
        x = x[idx_]
        y = y[idx_]

        f = fit_power_law(x, y)
        append!(psc[labels[i]], f(0.6))

        if (run == n_runs) 
            psm = round(mean(psc[labels[i]]), digits=2)
            psse = std(psc[labels[i]])/sqrt(n_runs)
            pslci = round(psm - 1.96*psse, digits=2)
            psuci = round(psm + 1.96*psse, digits=2)
            if n_runs > 1
                psc_label = labels[i]*" (PS="*string(psm)*" ["*string(pslci)*", "*string(psuci)*"])"
            else 
                psc_label = labels[i]*" (PS="*string(psm)
            end
        else
            psc_label = nothing
        end
        global psfig = scatter!(x, y, c=colors[i], alpha=1/(2*n_runs), label=nothing)
        global psfig = plot!(0:0.01:1, x -> f(x), c=colors[i], label=psc_label)
    end 
end 
psfig
savefig(psfig, "figures/pattern-separation/pattern-separation-curve"*fig_ext)
