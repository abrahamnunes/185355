using CSV, DataFrames, Plots, LaTeXStrings, Peaks, Glob


V = DataFrame()
for i ∈ 0:2
    v0 = CSV.read("data/fi-curves/vm-$i.000000.txt", delim="\t", header=0, DataFrame)
    v0[:,"ID"] .= i
    V = [V; v0]
end


pks, vals = findmaxima(V[V.ID .== 0,20])
pks = pks[vals .> 0.0]
ndrs, vals = findminima(V[V.ID .== 0, 20])

v0 = V[V.ID .== 0, :]
plot(v0[:,1], v0[:,20])
scatter!(v0[pks,1], v0[pks,20])
scatter!(v0[ndrs,1], v0[ndrs,20])



"""
    find_spikes(v;thresh)

Finds the spike peaks in a voltage tracing. Spikes are defined as those maxima that exceed some value `thresh` where default is that `thresh=0.0` 
"""
function find_spikes(v::Vector{Float64};thresh::Float64=0.0)

end


""" 
    mean_frequency(v)

Mean frequency calculated as number of action potentials during
stimulation divided by time between stimulus onset and last spike in
Hz.
"""
function mean_frequency(v::Vector{Float64})
end

""" 
    isi_log_slope(v) 

Slope of loglog interspike intervals (ISI).
"""
function isi_log_slope(v::Vector{Float64})
end

""" 
    adaptation_index2(v)

Normalized average difference of two consecutive ISI starting from
second ISI.
""" 
function adaptation_index2(v::Vector{Float64})
end

""" 
    time_to_first_spike(v)

Time from stimulus onset to peak time of first spike in ms.
""" 
function time_to_first_spike(v::Vector{Float64})
end

""" 
    time_to_last_spike(v)

Time from stimulus onset to peak time of last spike in ms.
""" 
function time_to_last_spike(v::Vector{Float64})
end


""" 
    AP_width(v)

Mean of width at -20 mV of action potential (AP) in ms. Mean for all AP.
""" 
function AP_width(v::Vector{Float64})
end


""" 
    AP_height(v)

Height at peak of action potential in mV. Mean for all AP.
""" 
function AP_height(v::Vector{Float64})
end

""" 
    min_voltage_between_spikes(v)

Minimum voltage between two action potentials in mV. Mean for all ISI.
""" 
function min_voltage_between_spikes(v::Vector{Float64})
end

""" 
    steady_state_voltage_stimend(v)

The average voltage during the last 90% of the stimulus duration in mV.
""" 
function steady_state_voltage_stimend(v::Vector{Float64})
end

""" 
    voltage_base(v)

The average voltage during the last 90% before stimulus onset in mV.
""" 
function voltage_base(v::Vector{Float64})
end
 
""" 
    voltage_after_stim(v)

The average voltage between 25% and 75% between end of stimulus and end of recording in mV.
""" 
function voltage_after_stim(v::Vector{Float64})
end
 
 