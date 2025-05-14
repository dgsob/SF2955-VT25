using CSV
using DataFrames
using LinearAlgebra
using Random
using Distributions
using Plots
using Statistics

# Load data from hmc-observations.csv
data = CSV.read(joinpath(@__DIR__, "data/hmc-observations.csv"), DataFrame, header=false)
y = data[:, 1]

# Define model parameters
σ = 2.0
Σ = [5.0 0.0; 0.0 0.5]
Σ_inv = inv(Σ)  # Σ⁻¹ = [0.2 0; 0 2]

# Log-posterior function (same as HMC)
function log_posterior(θ, y, σ, Σ_inv)
    n = length(y)
    θ_norm = dot(θ, θ)  # θ₁² + θ₂²
    # Likelihood term
    likelihood = -0.5 / (σ^2) * sum((y[i] - θ_norm)^2 for i in 1:n)
    # Prior term
    prior = -0.5 * dot(θ, Σ_inv * θ)
    return likelihood + prior
end

# Metropolis-Hastings algorithm
function metropolis_hastings(y, σ, Σ_inv, θ_init, ζ, N)
    d = length(θ_init)  # Dimension (2 for θ = [θ₁, θ₂])
    samples = zeros(N + 1, d)  # Store samples (including initial)
    samples[1, :] = θ_init  # Set initial θ
    θ = copy(θ_init)
    accept_count = 0

    # Proposal distribution: N₂(θ, ζ²I)
    proposal_dist = MvNormal(zeros(d), ζ^2 * I)

    for t in 1:N
        # Propose new θ*
        θ_proposed = θ + rand(proposal_dist)
        
        # Compute log-posterior for current and proposed states
        log_p_current = log_posterior(θ, y, σ, Σ_inv)
        log_p_proposed = log_posterior(θ_proposed, y, σ, Σ_inv)

        # Metropolis acceptance probability
        # Proposal is symmetric, so q(θ*|θ) = q(θ|θ*), ratio cancels
        α = min(1, exp(log_p_proposed - log_p_current))

        if rand() < α
            θ = copy(θ_proposed) # Accept proposal
            accept_count += 1
        end
        # Store sample (whether accepted or rejected)
        samples[t + 1, :] = θ
    end

    acceptance_rate = accept_count / N
    return samples, acceptance_rate
end