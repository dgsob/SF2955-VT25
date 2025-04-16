using LinearAlgebra, Distributions, Plots, Random, Printf
using MAT, StatsBase 
include("./utility_functions.jl")


function run_problem_4()
    verbose(true) # Whether to print info
    @info "------------------------------------------------"
    @info "Starting Problem 4: Sequential Importance Sampling with Resampling (SISR)"

    # --- Set Parameters ---
    Δt = 0.5
    α = 0.6
    σ = 0.5
    m = 500 # Number of steps - Will be adjusted if data differs
    N = 10000 # Number of particles
    Random.seed!(15) # Use same seed for comparison with SIS

    ν = 90.0
    η = 3.0
    ζ = 1.5

    P_raw = [16.0 1.0 1.0 1.0 1.0;
             1.0 16.0 1.0 1.0 1.0;
             1.0 1.0 16.0 1.0 1.0;
             1.0 1.0 1.0 16.0 1.0;
             1.0 1.0 1.0 1.0 16.0]
    P = P_raw ./ sum(P_raw[1,:])

    Z_values = [[0.0, 0.0], [3.5, 0.0], [0.0, 3.5], [-3.5, 0.0], [0.0, -3.5]]

    μ₀ = zeros(6)
    Σ₀ = Diagonal([500.0, 5.0, 5.0, 200.0, 5.0, 5.0])

    # --- Load Data ---
    stations_file_path = "stations.mat"
    rssi_file_path = "RSSI-measurements.mat"
    local stations, Y_obs, m

    try
        @info "Loading station data from $stations_file_path..."
        stations_data = matread(stations_file_path)
        if !haskey(stations_data, "pos_vec") 
            @error "Variable 'pos_vec' not found in '$stations_file_path'" 
        end
        stations = stations_data["pos_vec"]

        @info "Loading RSSI data from $rssi_file_path..."
        rssi_data = matread(rssi_file_path)
        if !haskey(rssi_data, "Y") 
            @error "Variable 'Y' not found in '$rssi_file_path'" 
        end
        Y_obs = rssi_data["Y"]
        @info "Data loaded successfully."

        local s_st
        dim_st_orig, s_st_orig = size(stations)
        if dim_st_orig == 2 && s_st_orig >= 1
             s_st = s_st_orig
        elseif s_st_orig == 2 && dim_st_orig >= 1
             stations = stations'
             dim_st_new, s_st = size(stations)
        else
             @error "Stations data 'pos_vec' must be 2xS or Sx2 (S>=1), got ($dim_st_orig, $s_st_orig)"
        end

        s_loaded, m_plus_1 = size(Y_obs)
        if m_plus_1 - 1 != m
            original_m = m
            m = m_plus_1 - 1
            @info "Adjusted step count m based on loaded Y data: $original_m -> $m"
        end
        if s_loaded != s_st error("Dimension mismatch: Y has $s_loaded rows, stations implies $s_st stations.") end
        @info "Using data dimensions: m=$m, s=$s_loaded"

    catch e
        @info "Error during data loading or verification: $e"
        return
    end

    # --- Define Noise Distribution ---
    W_distribution = MvNormal(zeros(2), σ^2 * I)

    # --- Run the SISR Simulation ---
    # Now receives both estimates and histogram data
    tau_estimates, weight_hist_data = run_sisr(m, N, Δt, α, σ, P, Z_values, μ₀, Σ₀, Y_obs, stations, ν, η, ζ, W_distribution)

    # --- Plotting ---
    @info "Plotting results..."
    p1 = plot_trajectory(tau_estimates, stations, " (N=$N, SISR)")
    display(p1)

    # Call histogram plotting function
    p2 = plot_weight_histograms(weight_hist_data)
    display(p2)
    
    @info "Problem 4 finished."
end

function run_sisr(m::Int, N::Int, Δt::Float64, α::Float64, σ::Float64,
    P::Matrix{Float64}, Z_values::Vector{Vector{Float64}},
    μ₀::Vector{Float64}, Σ₀::Diagonal{Float64},
    Y_obs::Matrix{Float64}, # s x (m+1)
    stations::Matrix{Float64}, # 2 x s
    ν::Float64, η::Float64, ζ::Float64,
    W_distribution::MvNormal)

    s = size(stations, 2)
    zeta_sq = ζ^2
    num_Z_states = length(Z_values)

    Φ, Ψz, Ψw = define_model_matrices(Δt, α)
    @info "--- Model Matrices Used (SISR) ---"
    @info "\nMatrix Φ (State Transition):\n$(sprint(format_matrix_to_io, Φ))"
    @info "\nMatrix Ψz (Driving Command Input):\n$(sprint(format_matrix_to_io, Ψz))"
    @info "\nMatrix Ψw (Noise Input):\n$(sprint(format_matrix_to_io, Ψw))"
    @info "---------------------------"

    tau_hat = zeros(2, m + 1)
    X_particles = zeros(6, N)
    Z_idx_particles = zeros(Int, N)
    X_propagated = zeros(6, N)
    Z_idx_propagated = zeros(Int, N)
    log_weights_norm = zeros(N)

    # Re-enable histogram storage
    hist_times = [m ÷ 4, m ÷ 2, m]
    weight_histograms = Dict{Int, Vector{Float64}}()

    # --- Initialization (n=0) ---
    @info "Initializing SISR (N=$N)..."
    X0_dist = MvNormal(μ₀, Σ₀)
    X_particles .= rand(X0_dist, N)
    Z_idx_particles .= rand(1:num_Z_states, N)

    log_p_y0 = calculate_log_observation_density(Y_obs[:, 1], X_particles, stations, ν, η, zeta_sq)
    weights_norm, log_weights_norm_current, _ = normalize_log_weights(log_p_y0)

    tau_hat[1, 1] = sum(weights_norm .* (@view X_particles[1, :]))
    tau_hat[2, 1] = sum(weights_norm .* (@view X_particles[4, :]))

    # Store initial weights for histogram (though less useful for SISR)
    if 0 in hist_times 
        weight_histograms[0] = copy(weights_norm) 
    end

    # --- Initial Resample ---
    resampled_indices = systematic_resample(weights_norm)
    X_particles = X_particles[:, resampled_indices]
    Z_idx_particles = Z_idx_particles[resampled_indices]
    log_weights_norm .= -log(N) # Reset weights for start of step 1

    @info "Starting SISR loop (m=$m)..."
    # --- Iteration (n=1 to m) ---
    for n in 1:m
        if n % 50 == 0 
            @info "  Processing step $n/$m..." 
        end

        X_prev_particles = X_particles
        Z_idx_prev_particles = Z_idx_particles

        # --- Propagate Particles ---
        Z_n_minus_1_vectors = [Z_values[idx] for idx in Z_idx_prev_particles]
        for i in 1:N
            prob_vec = P[Z_idx_prev_particles[i], :]
            prob_vec = max.(0.0, prob_vec)
            sum_p = sum(prob_vec)
            if sum_p > 1e-9 && all(isfinite.(prob_vec))
                prob_vec ./= sum_p
                Z_idx_propagated[i] = rand(Categorical(prob_vec))
            else
                Z_idx_propagated[i] = rand(1:num_Z_states)
            end
        end

        W_n_samples = rand(W_distribution, N)

        term1 = Φ * X_prev_particles
        term3 = Ψw * W_n_samples
        term2 = zeros(6, N)
        for i in 1:N
            term2[:, i] = Ψz * Z_n_minus_1_vectors[i]
        end
        X_propagated .= term1 .+ term2 .+ term3

        # --- Weighting ---
        yn = Y_obs[:, n+1]
        log_p_yn = calculate_log_observation_density(yn, X_propagated, stations, ν, η, zeta_sq)
        log_weights_unnorm = log_weights_norm .+ log_p_yn # Start with -log(N)

        # --- Normalization ---
        weights_norm, log_weights_norm_current, _ = normalize_log_weights(log_weights_unnorm)

        # --- Estimate tau_n ---
        tau_hat[1, n+1] = sum(weights_norm .* (@view X_propagated[1, :]))
        tau_hat[2, n+1] = sum(weights_norm .* (@view X_propagated[4, :]))

        # --- Store Weights for Histogram (before resampling) ---
        if n in hist_times
            weight_histograms[n] = copy(weights_norm)
            @info "    Stored weights for histogram at n=$n"
        end

        # --- Resampling ---
        resampled_indices = systematic_resample(weights_norm)
        X_particles = X_propagated[:, resampled_indices]
        Z_idx_particles = Z_idx_propagated[resampled_indices]

        # --- Reset Weights for next iteration ---
        log_weights_norm .= -log(N)

    end # End of loop n=1 to m

    @info "SISR loop finished."
    return tau_hat, weight_histograms
end

run_problem_4()