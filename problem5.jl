using LinearAlgebra, Distributions, Plots, Random, Printf
using MAT, StatsBase, Logging
include("./utility_functions.jl")

function run_problem_5()
    verbose(true)
    @info "------------------------------------------------"
    @info "Starting Problem 5: Approximate MLE for Zeta (ζ)"

    # Parameters
    Δt = 0.5
    α = 0.6
    σ_process_noise = 0.5 # Std dev for W (process noise)
    m = 500 # Default, will be adjusted
    N = 10000 # Particles
    ν = 90.0
    η = 3.0
    # ζ (zeta_obs) is estimated

    P_raw = [16.0 1.0 1.0 1.0 1.0;
             1.0 16.0 1.0 1.0 1.0;
             1.0 1.0 16.0 1.0 1.0;
             1.0 1.0 1.0 16.0 1.0;
             1.0 1.0 1.0 1.0 16.0]
    P = P_raw ./ sum(P_raw[1,:]) # Normalize rows

    Z_values = [[0.0, 0.0], [3.5, 0.0], [0.0, 3.5], [-3.5, 0.0], [0.0, -3.5]]

    μ₀ = zeros(6)
    Σ₀ = Diagonal([500.0, 5.0, 5.0, 200.0, 5.0, 5.0])

    # Load Data
    stations_file_path = "stations.mat"
    rssi_file_path = "RSSI-measurements-unknown-sigma.mat"
    local stations, Y_obs, s, m

    try
        stations_data = matread(stations_file_path); stations = stations_data["pos_vec"]
        rssi_data = matread(rssi_file_path)
        data_key = haskey(rssi_data, "Y") ? "Y" : haskey(rssi_data, "y0m") ? "y0m" : error("Cannot find 'Y' or 'y0m' in RSSI data")
        Y_obs = rssi_data[data_key]

        dim_st, s_st = size(stations)
        if s_st == 2 && dim_st >= 1; stations = stations'; s = dim_st; @info "Transposed stations";
        elseif dim_st == 2 && s_st >= 1; s = s_st;
        else error("Stations bad dims") end

        s_loaded, m_plus_1 = size(Y_obs); m_loaded = m_plus_1 - 1
        if m != m_loaded; @info "Adjusting m: $m -> $m_loaded"; m = m_loaded; end
        if s_loaded != s error("Dimension mismatch Y($s_loaded) vs stations($s)") end
        @info "Data loaded: m=$m, s=$s"
    catch e; @error "Data loading error: $e"; return end

    W_distribution = MvNormal(zeros(2), σ_process_noise^2 * I)

    # Grid Search for Zeta
    zeta_min = 0.1; zeta_max = 2.9; num_zeta_points = 50
    zeta_grid = range(zeta_min, stop=zeta_max, length=num_zeta_points)
    log_likelihoods = fill(-Inf, num_zeta_points) # Initialize with -Inf
    base_seed = 12345

    @info "Starting grid search for ζ over [$(zeta_min), $(zeta_max)] ($num_zeta_points points)..."
    # Threads.@threads for parallel execution (start Julia with -t auto)
    Threads.@threads for i in 1:num_zeta_points
        zeta_candidate = zeta_grid[i]
        current_seed = base_seed + i
        tid = Threads.threadid() # Get thread ID for logging if parallel
        @info "  [Thread $tid] Running SISR for ζ = $(@sprintf("%.3f", zeta_candidate)) (seed=$current_seed)..."
        _, log_lik_candidate = run_sisr_for_likelihood(
            m, N, Δt, α, σ_process_noise, P, Z_values, μ₀, Σ₀,
            Y_obs, stations, ν, η, zeta_candidate, W_distribution,
            seed=current_seed
        )
        log_likelihoods[i] = log_lik_candidate
        @info "  [Thread $tid] ζ = $(@sprintf("%.3f", zeta_candidate)), LogLik = $(@sprintf("%.4f", log_lik_candidate))"
    end

    # Find best zeta after loop
    max_log_lik, best_idx = findmax(log_likelihoods)
    if !isfinite(max_log_lik) || isnan(max_log_lik)
        @error "All log-likelihood estimates non-finite/NaN. Cannot determine best zeta."
        # Maybe plot log_likelihoods here to debug
        p_debug = plot(zeta_grid, log_likelihoods, title="Likelihood Debug", xlabel="ζ", ylabel="LogLik")
        display(p_debug)
        return
    end
    best_zeta = zeta_grid[best_idx]
    @info "Grid search complete."
    @info "Approximate MLE: ζ̂ ≈ $(@sprintf("%.4f", best_zeta))"
    @info "Corresponding Max Log-Likelihood ≈ $(@sprintf("%.4f", max_log_lik))"

    # Final run with best zeta
    @info "Running final SISR with ζ̂ = $(best_zeta) for trajectory..."
    final_seed = base_seed + best_idx
    best_tau_hat, _ = run_sisr_for_likelihood(
        m, N, Δt, α, σ_process_noise, P, Z_values, μ₀, Σ₀,
        Y_obs, stations, ν, η, best_zeta, W_distribution,
        seed=final_seed
    )

    # Plotting
    @info "Plotting results..."
    p_likelihood = plot(zeta_grid, log_likelihoods, xlabel="ζ (Observation Noise Std Dev)",
                        ylabel="Estimated Log-Likelihood", title="Log-Likelihood vs ζ",
                        label="LogLik", legend=:bottomleft, marker=:circle, markersize=3)
    vline!(p_likelihood, [best_zeta], label="ζ̂ ≈ $(@sprintf("%.3f", best_zeta))", color=:red, linestyle=:dash)
    display(p_likelihood) # Show likelihood plot

    title_suffix = " (N=$N, SISR, ζ̂ ≈ $(@sprintf("%.3f", best_zeta)))"
    p_trajectory = plot_trajectory(best_tau_hat, stations, title_suffix)
    display(p_trajectory) # Show trajectory plot

    @info "Problem 5 finished."
end

# --- Modified SISR Function (for Likelihood Estimation) ---
function run_sisr_for_likelihood(m::Int, N::Int, Δt::Float64, α::Float64, σ_process::Float64,
                                 P::Matrix{Float64}, Z_values::Vector{Vector{Float64}},
                                 μ₀::Vector{Float64}, Σ₀::Diagonal{Float64},
                                 Y_obs::Matrix{Float64}, # s x (m+1)
                                 stations::Matrix{Float64}, # 2 x s
                                 ν::Float64, η::Float64, ζ_obs::Float64, # Use zeta for observation
                                 W_distribution::MvNormal;
                                 seed::Union{Int, Nothing}=nothing)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    s = size(stations, 2)
    zeta_sq = ζ_obs^2 # Calculate zeta squared for observation density function
    num_Z_states = length(Z_values)

    Φ, Ψz, Ψw = define_model_matrices(Δt, α) # Assumes function is in utility_functions.jl

    # Initialization...
    tau_hat = zeros(2, m + 1)
    X_particles = zeros(6, N)
    Z_idx_particles = zeros(Int, N)
    X_propagated = zeros(6, N)
    Z_idx_propagated = zeros(Int, N)
    total_log_likelihood = 0.0

    # --- Initialization (n=0) ---
    X0_dist = MvNormal(μ₀, Σ₀)
    try X_particles .= rand(X0_dist, N) catch e; @error "Init Error: $e"; return zeros(2,m+1), -Inf end
    Z_idx_particles .= rand(1:num_Z_states, N)

    y0 = Y_obs[:, 1]
    log_p_y0 = calculate_log_observation_density(y0, X_particles, stations, ν, η, zeta_sq)
    weights_norm_0, _, log_sum_w0 = normalize_log_weights(log_p_y0) # Use modified function

    if !isfinite(log_sum_w0)
        @warn "LogLik(n=0) non-finite ($log_sum_w0) for ζ=$ζ_obs. Aborting."
        return tau_hat, -Inf
    end
    total_log_likelihood += log_sum_w0

    tau_hat[1, 1] = sum(weights_norm_0 .* (@view X_particles[1, :]))
    tau_hat[2, 1] = sum(weights_norm_0 .* (@view X_particles[4, :]))

    resampled_indices = systematic_resample(weights_norm_0)
    X_particles = X_particles[:, resampled_indices]
    Z_idx_particles = Z_idx_particles[resampled_indices]

    # --- Iteration (n=1 to m) ---
    for n in 1:m
        X_prev_particles = X_particles
        Z_idx_prev_particles = Z_idx_particles

        # --- Propagate Z ---
        Z_n_minus_1_vectors = [Z_values[idx] for idx in Z_idx_prev_particles]
        for i in 1:N
            prob_vec = P[Z_idx_prev_particles[i], :]
            prob_vec = max.(0.0, prob_vec)
            sum_p = sum(prob_vec)
            if sum_p > 1e-9 && all(isfinite.(prob_vec))
                prob_vec ./= sum_p
                try Z_idx_propagated[i] = rand(Categorical(prob_vec)) catch e; @warn "Z Sample Err (n=$n, i=$i): $e"; Z_idx_propagated[i] = rand(1:num_Z_states) end
            else
                @warn "Z Prob Err (n=$n, i=$i). Uniform sample."
                Z_idx_propagated[i] = rand(1:num_Z_states)
            end
        end

        # --- Propagate X ---
        W_n_samples = rand(W_distribution, N)
        term1 = Φ * X_prev_particles
        term3 = Ψw * W_n_samples
        term2 = hcat([Ψz * Z_n_minus_1_vectors[i] for i in 1:N]...)
        X_propagated .= term1 .+ term2 .+ term3

        # --- Weighting & Normalization ---
        yn = Y_obs[:, n+1]
        log_p_yn = calculate_log_observation_density(yn, X_propagated, stations, ν, η, zeta_sq)
        log_weights_unnorm = fill(-log(N), N) .+ log_p_yn # Previous weights were 1/N
        weights_norm, _, log_sum_w_n = normalize_log_weights(log_weights_unnorm) # Use modified func

        if !isfinite(log_sum_w_n)
            @warn "LogLik(n=$n) non-finite ($log_sum_w_n) for ζ=$ζ_obs. Aborting."
            return tau_hat, -Inf
        end
        total_log_likelihood += log_sum_w_n

        # --- Estimate tau_n ---
        tau_hat[1, n+1] = sum(weights_norm .* (@view X_propagated[1, :]))
        tau_hat[2, n+1] = sum(weights_norm .* (@view X_propagated[4, :]))

        # --- Resampling ---
        if n < m
            resampled_indices = systematic_resample(weights_norm)
            X_particles = X_propagated[:, resampled_indices]
            Z_idx_particles = Z_idx_propagated[resampled_indices]
        end
    end # End loop n

    return tau_hat, total_log_likelihood
end

run_problem_5()