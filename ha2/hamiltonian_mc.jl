using CSV
using DataFrames
using LinearAlgebra
using Random
using Distributions
using Plots

# Load data from hmc-observations.csv
data = CSV.read(joinpath(@__DIR__, "data/hmc-observations.csv"), DataFrame, header=false)
y = data[:, 1]

# Define model parameters
σ = 2.0
Σ = [5.0 0.0; 0.0 0.5]
Σ_inv = inv(Σ)  # Σ⁻¹ = [0.2 0; 0 2]

# Log-posterior function (up to a constant)
function log_posterior(θ, y, σ, Σ_inv)
    n = length(y)
    θ_norm = dot(θ, θ)  # θ₁² + θ₂²
    # Likelihood term
    likelihood = -0.5 / (σ^2) * sum((y[i] - θ_norm)^2 for i in 1:n)
    # Prior term
    prior = -0.5 * dot(θ, Σ_inv * θ)
    return likelihood + prior
end

# Gradient of log-posterior
function grad_log_posterior(θ, y, σ, Σ_inv)
    n = length(y)
    θ_norm = dot(θ, θ)  # θ₁² + θ₂²
    # Likelihood gradient term: (2/σ²) * ∑(y_i - θ₁² - θ₂²) * θ
    likelihood_∇ = (2 / σ^2) * sum(y[i] - θ_norm for i in 1:n) * θ
    # Prior gradient term: -Σ⁻¹ θ
    prior_∇ = -Σ_inv * θ
    return likelihood_∇ + prior_∇
end

# HMC algorithm
function hmc(y, σ, Σ_inv, θ_init, ε, L, N)
    d = length(θ_init)  # Dimension (2 for θ = [θ₁, θ₂])
    samples = zeros(N + 1, d)  # Store samples (including initial)
    samples[1, :] = θ_init  # Set initial θ
    θ = copy(θ_init)
    accept_count = 0

    for t in 1:N
        # (a) Sample momentum
        p = rand(Normal(0, 1), d)  # p ~ N₂(0, I)

        # (b) Set current state
        θ_current = copy(θ)
        p_current = copy(p)

        # (c) Leapfrog integration
        for l in 1:L
            p += (ε / 2) * grad_log_posterior(θ, y, σ, Σ_inv) # half-step momentum update
            θ += ε * p                                        # full-step position update
            p += (ε / 2) * grad_log_posterior(θ, y, σ, Σ_inv) # half-step momentum update
        end

        # (d) Proposed state is just (θ, p)
        # (e) Compute Hamiltonian
        H_current = -log_posterior(θ_current, y, σ, Σ_inv) + 0.5 * dot(p_current, p_current)
        H_proposed = -log_posterior(θ, y, σ, Σ_inv) + 0.5 * dot(p, p)

        # (f) Metropolis Acceptance and (g) Store sample θ in samples
        α = min(1, exp(-H_proposed + H_current))
        if rand() < α
            samples[t + 1, :] = θ  # Accept proposal
            accept_count += 1
        else
            θ = copy(θ_current)
            samples[t + 1, :] = θ  # Reject proposal and keep current state
        end
    end

    acceptance_rate = accept_count / N
    return samples, acceptance_rate
end