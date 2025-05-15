using Distributions, StatsPlots, DataFrames, CSV, Random
include("./auxiliary_functions.jl")

# Load coal mine disaster data
data = CSV.read(joinpath(@__DIR__, "coal-mine.csv"), DataFrame, header=false)[:, 1]
τ = data
t1, t_d1 = 1851.0, 1963.0
n = length(τ)

# Function to compute n_i(τ)
function compute_ni(t, τ)
    d = length(t) - 1
    ni = zeros(Int, d)
    for τj in τ
        for i in 1:d
            if t[i] <= τj < t[i+1]
                ni[i] += 1
                break
            end
        end
    end
    return ni
end

# Hybrid MCMC function
function hybrid_mcmc(n_iter, d, ρ=0.2, ϑ=2.0)
    # Initialize parameters
    t = collect(range(t1, stop=t_d1, length=d+1))
    λ = rand(Gamma(2, 1.0), d)
    θ = rand(Gamma(2, ϑ))
    
    # Storage for samples
    θ_samples = zeros(n_iter)
    λ_samples = zeros(n_iter, d)
    t_samples = zeros(n_iter, d+1)
    
    for k in 1:n_iter
        # Gibbs for θ
        θ = rand(Gamma(2d + 2, 1.0 / (ϑ + sum(λ))))
        
        # Gibbs for λ
        ni = compute_ni(t, τ)
        for i in 1:d
            λ[i] = rand(Gamma(ni[i] + 2, 1.0 / (t[i+1] - t[i] + θ)))
        end
        
        # Metropolis-Hastings for t (random walk)
        i = rand(2:d)  # breakpoint to update
        R = ρ * (t[i+1] - t[i-1])
        t_star = t[i] + rand(Uniform(-R, R))
        
        # Simple check if t is ordered
        if t_star > t[i-1] && t_star < t[i+1]
            t_prop = copy(t)
            t_prop[i] = t_star
            ni_prop = compute_ni(t_prop, τ)
            
            # Log posterior ratio (up to constant)
            log_p_curr = sum(ni .* log.(λ)) - sum(λ .* (t[2:end] - t[1:end-1]))
            log_p_prop = sum(ni_prop .* log.(λ)) - sum(λ .* (t_prop[2:end] - t_prop[1:end-1]))
            log_alpha = log_p_prop - log_p_curr
            
            if log(rand()) < log_alpha
                t[i] = t_star
            end
        end
        
        # Store samples
        θ_samples[k] = θ
        λ_samples[k, :] = λ
        t_samples[k, :] = t
    end
    
    return θ_samples, λ_samples, t_samples
end

# Run for different numbers of breakpoints
function run_part_one()
    Random.seed!(77)
    n_iter = 77000
    for d in 2:5  # Corresponds to the breakpoints
        breakpoints_string = get_breakpoints_string(d-1)
        println("Running with $d intervals (d-1 = $(d-1) $breakpoints_string)")
        θ_s, λ_s, t_s = hybrid_mcmc(n_iter, d)

        # Generate labels using the function
        t_labels, λ_labels = generate_labels(t_s, d, t1, t_d1)
        
        # Create a single figure with 3 subplots
        p = plot(layout=(1, 3), size=(1200, 300), plot_title="Posteriors for $(d-1) $breakpoints_string",
                left_margin=21Plots.px, bottom_margin=25Plots.px, top_margin=25Plots.px)
        
        # Histogram for θ
        bin_length = 77 # standarized arbitrary length for clearer comparison
        bin_edges_θ = range(0, 4, length=bin_length)
        histogram!(p[1], θ_s[1001:end], label="", xlabel="Rate parameter θ", ylabel="Frequency", #title="Posterior θ", 
                bins=bin_edges_θ)
        
        # Histogram for λ
        λ_flat = vec(λ_s[1001:end, :])  # Flatten the λ samples
        λ_groups = repeat(1:d, inner=n_iter-1000)  # Create group labels for each λ_i
        λ_max = maximum(λ_s[1001:end, :])
        bin_edges_λ = range(0, 8, length=bin_length)
        histogram!(p[2], λ_flat, group=λ_groups, label=permutedims(λ_labels), xlabel="Disaster Intensities λ", ylabel="", alpha=0.6, #title="Disaster Intensities λ", 
                bins=bin_edges_λ, legend = :topright)
        
        # Histogram for t
        t_flat = vec(t_s[1001:end, 2:d])  # Flatten the t samples
        t_groups = repeat(1:(d-1), inner=n_iter-1000)  # Create group labels for each breakpoint
        bin_edges_t = range(1851, 1963, length=bin_length)
        _breakpoints_string = get_breakpoints_string(d-1, true)
        histogram!(p[3], t_flat, group=t_groups, label=permutedims(t_labels), xlabel="$_breakpoints_string t", ylabel="", alpha=0.6, #title="Breakpoints t", 
                bins=bin_edges_t, legend = :topleft)
        
        display(p)
        # savefig(p, "figure_for_$(d-1)_$breakpoints_string.png")
    end
    println("Processing part 1 finished \U0001F973")
end