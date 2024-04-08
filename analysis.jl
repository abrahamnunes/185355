
####################################################################
# SCRIPT TO RUN ANALYSIS 
####################################################################
include("utilities.jl");
default(show=false)


# HYPERPARAMS
n_runs = 14
patterns = 0:24
spontaneousactivity = true


labels = ["HC_CTRL", "LR_CTRL", "NR_CTRL", "HC_LITM", "LR_LITM", "NR_LITM"]
psc = Dict("HC_CTRL"=>[], "LR_CTRL"=>[], "NR_CTRL"=>[], "HC_LITM"=>[], "LR_LITM"=>[], "NR_LITM"=>[])


if spontaneousactivity
    fig_ext = "-wSA.png"
    csv_ext = "-wSA.csv"
else
    fig_ext = "-nSA.png"
    csv_ext = "-nSA.csv"
end

# CREATE NECESSARY DIRECTORIES 
create_directories_if_not_exist()

# IDENTIFY NEURON POPULATION RANGES
populations = Dict(
    "DG" => [0, 500],
    "BC" => [500,506],
    "MC" => [506, 521],
    "HIPP" => [521, 527]
)

println("finished set up. beginning raster plot.")

for run ∈ 1:n_runs
    for i ∈ 1:length(labels)
        spikes = load_spike_files(patterns, labels[i]*"-$run", populations)
        println(i)
        # CREATE RASTER PLOTS
        for p ∈ unique(spikes.Pattern)
            stimin = spikes[(spikes.Population .== "PP") .& (spikes.Pattern .== p), :]
            plots = []
            append!(plots, [raster_plot(stimin; ylab="PP")])
            println(p)
            for pop ∈ keys(populations)
                lb, ub = populations[pop]
                popspikes = spikes[(spikes.Population .== pop) .& (spikes.Pattern .== p),:]
                #if size(popspikes,1) > 0
                append!(plots, [raster_plot(popspikes; xlab="", ylab=pop)])
                #end
                println(pop)
            end
            fig = plot(reverse(plots)..., layout=grid(5, 1, heights=[0.15, 0.15, 0.15, 0.4, 0.15]), size=(400, 500))
            savefig(fig, "figures/raster-plots/raster-"*string(p)*"-"*labels[i]*"-$run"*"-blSA"*".png")
        end
    end 
end 

println("Finished printing raster plots.")

println("Starting PS analysis.")
# PATTERN SEPARATION CURVES
colors=[:blue, :red, :green, :black, :gray, :purple]
global psfig = plot([0;1], [0;1], ls=:dash, c=:black, 
                        xlabel="Input Correlation "*L"(r_{in})", 
                        ylabel="Output Correlation "*L"(r_{out})", 
                        size=(400, 400),
                        label=nothing, legend=:outerbottom)

#psc = Dict("HC_CTRL"=>[], "LR_CTRL"=>[], "NR_CTRL"=>[]) #, "HC_LITM"=>[], "LR_CTRL"=>[], "LR_LITM"=>[], "NR_CTRL"=>[], "NR_LITM"=>[])

for i ∈ 1:length(labels)
    println(labels[i])
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
            psc_label = labels[i]*" (PS="*string(psm)*" ["*string(pslci)*", "*string(psuci)*"])"
        else
            psc_label = nothing
        end
        global psfig = scatter!(x, y, c=colors[i], alpha=1/(2*n_runs), label=nothing)
        global psfig = plot!(0:0.01:1, x -> f(x), c=colors[i], label=psc_label)
        println(run)
    end 
end 
psfig
savefig(psfig, "figures/pattern-separation/pattern-separation-curve"*fig_ext)

println("Finished plotting PS curves.")

println("Beginning AUC calculation. This will take .5 hours.")

auc_save = OrderedDict("HC_CTRL"=>[], "LR_CTRL"=>[], "NR_CTRL"=>[], "HC_LITM"=>[], "LR_LITM"=>[], "NR_LITM"=>[])
auc_means = OrderedDict("HC_CTRL"=>[], "LR_CTRL"=>[], "NR_CTRL"=>[], "HC_LITM"=>[], "LR_LITM"=>[], "NR_LITM"=>[])
auc_ses = OrderedDict("HC_CTRL"=>[], "LR_CTRL"=>[], "NR_CTRL"=>[], "HC_LITM"=>[], "LR_LITM"=>[], "NR_LITM"=>[])

for i ∈ 1:length(labels)
    println(i)
    for run ∈ 1:n_runs
        println(run)
        spikes = load_spike_files(patterns, labels[i]*"-$run", populations)
        
        out = pattern_separation_curve(spikes, 100, 500)
        x, y = out[:,"Input Correlation"], out[:, "Output Correlation"]
        
        # Remove NaNs before fitting
        idx_ = (.!isnan.(x) .& .!isnan.(y))
        x = x[idx_]
        y = y[idx_]

        auc = compute_auc(x, y)
        append!(auc_save[labels[i]], auc)
        if (run == n_runs) 
            aucm = round(mean(auc_save[labels[i]]), digits=2)
            append!(auc_means[labels[i]], aucm)
            aucse = std(auc_save[labels[i]])/sqrt(n_runs)
            append!(auc_ses[labels[i]], aucse)
        end
    end 
end 

df_aucsave = DataFrame(auc_save)
df_aucmeans = DataFrame(auc_means)
df_aucses = DataFrame(auc_ses)
CSV.write("figures/pattern-separation/auc_raw"*csv_ext, df_aucsave)
CSV.write("figures/pattern-separation/auc_means"*csv_ext, df_aucmeans)
CSV.write("figures/pattern-separation/auc_ses"*csv_ext, df_aucses)

println("CSVs saved.")

unpack(a) = eltype(a[1])[el[1] for el in a]

data = unpack(collect(values(auc_means)))
data_err = unpack(collect(values(auc_ses)))

pltlabels = ["HC", "LR", "NR"]
auc_fig = groupedbar(pltlabels, 
                [data[1:3] data[4:6]], 
                xlabel = "Group",
                #xtickfont=font(12),
                ylabel = L"AUC_{PS}",
                ylimits = (-0.15, 0.4),
                c = [:gray :white], 
                markerstrokewidth = 1,
                yerror = [data_err[1:3] data_err[4:6]], 
                dpi=300, size=(350,350),
                label=["Baseline" "Lithium"],
                grid = :none
                )
savefig(auc_fig, "figures/pattern-separation/auc-curve"*fig_ext)

println("Finished plotting AUC bars.")