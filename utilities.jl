#######################################################################
#   UTILITIES.JL 
#       Functions for analysis of data, I/O, plotting, etc.
#
#######################################################################
using DataFrames, CSV, Plots, PyCall, LsqFit, LaTeXStrings, Statistics, DataStructures, StatsPlots
np = pyimport("numpy")
ss = pyimport("scipy.stats")

#######################################################################
#   I/O Functions
#######################################################################
"""
    create_directories_if_not_exist()

Creates necessary directories if they don't yet exist.
"""
function create_directories_if_not_exist()
    if !("figures" ∈ readdir()) 
        mkdir("figures")
    end

    for d ∈ ["raster-plots", "pattern-separation", "voltage-tracings", "fi-curves"]
        if !(d ∈ readdir("figures"))
            mkdir("figures/"*d)
        end
    end
end

""" 
    load_spike_files(patterns, model_label, neuron_ids; neurons_per_pattern, data_dir)

Loads spike time files and concatenates them. File names from simulation are structured as follows:
    - "DGsp-{p}-{neurons_per_pattern}-{model_label}.txt"
    - "StimIn-{p}-{neurons_per_pattern}-{model_label}.txt"

The `neuron_ids` object is a `Dict` which includes lower and upper bound ranges on neuron IDs:
    ```
    neuron_ids = Dict(
        "DG" => [0, 500],
        "BC" => [500,506],
        "MC" => [506, 521],
        "HIPP" => [521, 527]
    )
    ```
"""
function load_spike_files(
    patterns::Union{UnitRange{Int64}, Vector{Int64}}, 
    model_label::String,
    neuron_ids::Dict;
    neurons_per_pattern::Int64=6, 
    data_dir::String="data/dgnetwork/")

    df = DataFrame()
    for p ∈ patterns
        spike_fname = data_dir*"DGsp-"*string(p)*"-"*string(neurons_per_pattern)*"-"*model_label*".txt"
        stims_fname = data_dir*"StimIn-"*string(p)*"-"*string(neurons_per_pattern)*"-"*model_label*".txt"
        spikes = CSV.read(spike_fname, delim="\t", header=0, DataFrame)
        stimin = CSV.read(stims_fname, delim="\t", header=0, DataFrame)

        if size(spikes, 1) > 0
            rename!(spikes, ["Time", "Neuron"])
            spikes[:,"Population"] .= "" 
            spikes[:,"Pattern"] .= p
            spikes[:,"NeuronsPerPattern"] .= neurons_per_pattern 
            spikes[:,"Model"] .= model_label
            for k ∈ keys(neuron_ids) 
                lb, ub = neuron_ids[k]
                spikes[lb .<= spikes[:, "Neuron"] .< ub, "Population"] .= k
            end

            df = [df; spikes]
        end 

        if size(stimin, 1) > 0
            rename!(stimin, ["Time", "Neuron"])
            stimin[:,"Population"] .= "PP"
            stimin[:,"Pattern"] .= p
            stimin[:,"NeuronsPerPattern"] .= neurons_per_pattern 
            stimin[:,"Model"] .= model_label
            df = [df; stimin]
        end
    end

    return df
end

#######################################################################
#   FUNCTIONS FOR STATISTICAL ANALYSIS
#######################################################################
""" 
    spike_train_correlation(spike_train1, spike_train2, n_neurons, delta, n_bins, duration)

Computes the similarity score for two spike trains using the Pearson
    correlation coefficient. This function was copied from that in
    Yim et al. (2015), but modified in a few ways. We added documentation,
    switched to using DataFrames (for clarity/brevity of code), and computed
    the correlation coefficient using the `scipy.pearsonr` function.
"""
function spike_train_correlation(
    spike_train1::DataFrame, 
    spike_train2::DataFrame, 
    n_neurons::Int64;
    delta::Int64=20, 
    n_bins::Int64=1, 
    duration::Float64=200.0)
    
    # Create kernel
    tri_rng = collect(1:round(Int64, delta/n_bins))
    triangular_kernel = [tri_rng; reverse(tri_rng)[2:end]]'

    # Create bins for histograms 
    bins = collect(1:n_bins:(duration + n_bins))

    x = zeros(round(Int64, n_neurons * duration / n_bins))
    y = zeros(round(Int64, n_neurons * duration / n_bins))
    for i ∈ 1:n_neurons
        s1 = spike_train1[spike_train1.Neuron .== (i-1), "Time"]
        s2 = spike_train2[spike_train2.Neuron .== (i-1), "Time"]
        timeseries1, _ = np.histogram(s1, bins)
        timeseries2, _ = np.histogram(s2, bins)
        idx_low = round(Int64, (i-1)*duration/n_bins + 1)
        idx_high = round(Int64, i*duration/n_bins)
        x[idx_low:idx_high] = np.convolve(timeseries1, triangular_kernel, "same")
        y[idx_low:idx_high] = np.convolve(timeseries2, triangular_kernel, "same")
    end
    return ss.pearsonr(x, y)
end


""" 
    pattern_separation_curve(spike_times, n_pp_cells, n_granule_cells; delta, n_bins, duration)

Computes pairwise correlations between input and output patterns
"""
function pattern_separation_curve(
    spike_times::DataFrame,
    n_pp_cells::Int64,
    n_granule_cells::Int64;
    delta::Int64=20,
    n_bins::Int64=1,
    duration::Float64=200.0)

    out = DataFrame()
    n_patterns = unique(spike_times.Pattern) |> length
    for i ∈ 0:(n_patterns - 2)
        for j ∈ i:(n_patterns-1) 
            pp_a = spike_times[(spike_times.Population .== "PP") .& (spike_times.Pattern .== i), ["Neuron", "Time"]]
            pp_b = spike_times[(spike_times.Population .== "PP") .& (spike_times.Pattern .== j), ["Neuron", "Time"]]
            r_in, p_in = spike_train_correlation(pp_a, pp_b, n_pp_cells; delta=delta, n_bins=n_bins, duration=duration)

            gc_a = spike_times[(spike_times.Population .== "DG") .& (spike_times.Pattern .== i), ["Neuron", "Time"]]
            gc_b = spike_times[(spike_times.Population .== "DG") .& (spike_times.Pattern .== j), ["Neuron", "Time"]]
            r_out, p_out = spike_train_correlation(gc_a, gc_b, n_granule_cells; delta=delta, n_bins=n_bins, duration=duration)

            out_ij = DataFrame(
                "Pattern A"=>[i],
                "Pattern B"=>[j],
                "Input Correlation"=>[r_in],
                "Output Correlation"=>[r_out],
                "Input Correlation p-Value"=>[p_in],
                "Output Correlation p-Value"=>[p_out],
            )
            out = [out; out_ij]
        end
    end
    return out 
end

"""
    fit_power_law(x, y)

Fits `a + (1-a)*(x^b)` to the pattern separation data. 
"""
function fit_power_law(x::Vector, y::Vector)
    # Clip for tractability
    x = max.(x, 0.0001)
    y = max.(y, 0.0001)

    @. model(x, w) = w[1] + (1-w[1])*(x^w[2])
    res = curve_fit(model, x, y, [0., 1.])
    a, b = res.param 
    f(x) = a + (1-a)*(x^b)
    return f
end

function compute_auc(x::Vector, y::Vector)
    # Clip for tractability
    x = max.(x, 0.0001)
    y = max.(y, 0.0001)

    @. model(x, w) = w[1] + (1-w[1])*(x^w[2])
    res = curve_fit(model, x, y, [0., 1.])
    a, b = res.param
    auc = - ((2*a*b) - b + 1)/(2*(1+b))
    return auc
end

#######################################################################
#   FUNCTIONS FOR PLOTTING
#######################################################################

"""
    raster_plot(spikes; xlims, xlab, ylab)

Returns a raster plot
"""
function raster_plot(
    spikes::DataFrame; 
    xlims::Vector=[0,200],
    xlab::String="Time (ms)",
    ylab::String="Neuron ID"
    )

    p = plot(xlab=xlab, ylab=ylab, xlims=xlims, yticks=false)
    p = scatter!(spikes[:,1], spikes[:,2], markershape=:vline, 
                label=nothing, c=:black, msc=:black, msa=1, ma=1, 
                 markerstrokewidth=1)
    return p
end