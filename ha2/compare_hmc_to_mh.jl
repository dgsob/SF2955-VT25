using Random

include("./hamiltonian_mc.jl")
include("./metropolis_hastings.jl")

# Autocorrelation function
function autocorrelation(samples, max_lag)
    n = size(samples, 1)
    d = size(samples, 2)
    acf = zeros(max_lag + 1, d)
    
    for dim in 1:d
        x = samples[:, dim]
        μ = mean(x)
        var = mean((x .- μ).^2)  # Sample variance
        
        for lag in 0:max_lag
            if lag == 0
                acf[lag + 1, dim] = 1.0
            else
                # Compute covariance for lag
                cov = mean((x[1:(n-lag)] .- μ) .* (x[(lag+1):n] .- μ))
                acf[lag + 1, dim] = cov / var
            end
        end
    end
    
    return acf
end

# Set parameters
θ_init = [0.0, 0.0]  # Initial arbitrary θ 
N = 10000  # Number of samples
max_lag = 50 # Max time steps for autocorrelation
# HMC-specific params
ε = 0.01  # Step size <------------------------------------- variable for tuning
L = 50   # Number of leapfrog steps <----------------------- variable for tuning
# MH-specific params
ζ = 0.2  # Proposal standard deviation <-------------------- variable for tuning

function run_part_two()
    # Set seed
    Random.seed!(77)

    # Run HMC
    hmc_samples, hmc_acceptance_rate = hmc(y, σ, Σ_inv, θ_init, ε, L, N)

    # Run MH
    mh_samples, mh_acceptance_rate = metropolis_hastings(y, σ, Σ_inv, θ_init, ζ, N)

    # Compare acceptance rate
    println("HMC Acceptance rate: ", hmc_acceptance_rate)
    println("MH Acceptance rate: ", mh_acceptance_rate)

    # Compare heatmaps
    p_heatmap = plot(layout=(1, 2), size=(1000, 300),
                    left_margin=20Plots.px, bottom_margin=20Plots.px, top_margin=10Plots.px)
    histogram2d!(p_heatmap[1], hmc_samples[:, 1], hmc_samples[:, 2], bins=50, 
                title="HMC Posterior Samples", xlabel="θ₁", ylabel="θ₂", colorbar=true)
    histogram2d!(p_heatmap[2], mh_samples[:, 1], mh_samples[:, 2], bins=50, 
                title="MH Posterior Samples", xlabel="θ₁", ylabel="θ₂", colorbar=true)

    # Compute autocorrelation for HMC and MH
    hmc_acf = autocorrelation(hmc_samples, max_lag)
    mh_acf = autocorrelation(mh_samples, max_lag)

    # Compare autocorrelation
    p_acf = plot(layout=(1, 2), size=(700, 300),
                left_margin=10Plots.px, bottom_margin=10Plots.px, top_margin=10Plots.px)
    plot!(p_acf[1], 0:max_lag, hmc_acf[:, 1], label="HMC θ₁", title="Autocorrelation for θ₁", 
        xlabel="Lag", ylabel="ACF", color=:blue)
    plot!(p_acf[1], 0:max_lag, mh_acf[:, 1], label="MH θ₁", linestyle=:dash, color=:red)
    plot!(p_acf[2], 0:max_lag, hmc_acf[:, 2], label="HMC θ₂", title="Autocorrelation for θ₂", 
        xlabel="Lag", ylabel="ACF", color=:blue)
    plot!(p_acf[2], 0:max_lag, mh_acf[:, 2], label="MH θ₂", linestyle=:dash, color=:red)

    # Display figures
    display(p_heatmap)
    display(p_acf)

    # Save figures
    # savefig(p_heatmap, "heatmap.png")
    # savefig(p_acf, "autocorrelation.png")

    println("Processing part 2 finished \U0001F973")
end


