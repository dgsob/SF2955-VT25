# === PROBLEM 1 ===

using LinearAlgebra, Distributions, Plots, Random, Printf

function prettyprint_matrix(matrix::AbstractMatrix; width::Int=6)
    rows, cols = size(matrix)
    if rows == 0 || cols == 0
        println("[] (Empty Matrix)")
        return
    end

    println("[")
    for i in 1:rows
        print(" ")
        for j in 1:cols
            print(lpad(string(matrix[i, j]), width))
            if j < cols
                print(" ")
            end
        end
        println()
    end
     println("]")
end

function run_simulation(m::Int, Δt::Float64, α::Float64, σ::Float64,
    P::Matrix{Float64}, Z_values::Vector{Vector{Float64}},
    μ₀::Vector{Float64}, Σ₀::Diagonal{Float64},
    W_distribution::MvNormal)

    # --- State Space Matrices ---
    println("\n--- Model's matrices  ---")
    Φ₁ = [1 Δt Δt^2/2; 0 1 Δt; 0 0 α]
    Φ = kron(I(2), Φ₁) # creates the 6x6 block diagonal matrix
    println("Matrix Φ (State Transition):")
    prettyprint_matrix(Φ)

    Ψz = zeros(6, 2)
    Ψz[1, 1] = Δt^2 / 2.0 
    Ψz[2, 1] = Δt        
    Ψz[4, 2] = Δt^2 / 2.0 
    Ψz[5, 2] = Δt        
    println("\nMatrix Ψz (Driving Command Input):")
    prettyprint_matrix(Ψz)

    Ψw = zeros(6, 2)
    Ψw[1, 1] = Δt^2 / 2.0 
    Ψw[2, 1] = Δt        
    Ψw[3, 1] = 1.0        
    Ψw[4, 2] = Δt^2 / 2.0 
    Ψw[5, 2] = Δt        
    Ψw[6, 2] = 1.0        
    println("\nMatrix Ψw (Noise Input):")
    prettyprint_matrix(Ψw)

    num_Z_states = length(Z_values)

    # --- Initial Conditions ---
    X₀_distribution = MvNormal(μ₀, Σ₀)
    X₀ = rand(X₀_distribution)
    z_idx_0 = rand(1:num_Z_states)

    # --- Initialize Trajectory Storage ---
    X_trajectory = zeros(6, m + 1)
    Z_idx_trajectory = zeros(Int, m + 1)

    # Store initial state
    X_trajectory[:, 1] = X₀
    Z_idx_trajectory[1] = z_idx_0

    # This variable holds the command index state across loop iterations
    current_z_idx = z_idx_0

    println("\nStarting simulation...")
    # --- Simulation Loop ---
    for n in 1:m
        # Get current state Xn and command Zn from local variables
        Xn = X_trajectory[:, n]
        Zn = Z_values[current_z_idx] # uses function-local current_z_idx

        # Sample next noise Wn+1
        Wn_plus_1 = rand(W_distribution)

        # Calculate next state Xn+1 (Xn+1 = Φ * Xn + Ψz * Zn + Ψw * Wn_plus_1)
        term_phi_xn = Φ * Xn
        term_psi_zn = Ψz * Zn
        term_psiw_wn1 = Ψw * Wn_plus_1
        Xn_plus_1 = term_phi_xn + term_psi_zn + term_psiw_wn1

        # Store the result
        X_trajectory[:, n+1] = Xn_plus_1

        # Sample next command state index Zn+1 based on P and current index
        probs = P[current_z_idx, :]
        probs = max.(0.0, probs)
        probs /= sum(probs)
        next_z_idx = rand(Categorical(probs))
        Z_idx_trajectory[n+1] = next_z_idx

        current_z_idx = next_z_idx
    end
    println("Simulation finished.")
    return X_trajectory
end


# --- Main Script Area ---
function main_problem_1()
    println("------------------------------------------------")

    # --- Set Parameters ---
    Δt = 0.5
    α = 0.6
    σ = 0.5
    m = 150 # arbitrary trajectory length
    Random.seed!(15)

    # Transition matrix P
    P = (1/20) * [16.0 1.0 1.0 1.0 1.0;
                  1.0 16.0 1.0 1.0 1.0;
                  1.0 1.0 16.0 1.0 1.0;
                  1.0 1.0 1.0 16.0 1.0;
                  1.0 1.0 1.0 1.0 16.0]

    # Driving command options
    Z_values = [
        [0.0, 0.0],  # 1: No command
        [3.5, 0.0],  # 2: East
        [0.0, 3.5],  # 3: North
        [-3.5, 0.0], # 4: West
        [0.0, -3.5]  # 5: South
    ]

    # Initial state distribution parameters
    μ₀ = zeros(6)
    Σ₀ = Diagonal([500.0, 5.0, 5.0, 200.0, 5.0, 5.0])

    # --- Noise Distribution --- 
    W_distribution = MvNormal(zeros(2), σ^2 * I)

    # --- Run the Simulation ---
    X_trajectory = run_simulation(m, Δt, α, σ, P, Z_values, μ₀, Σ₀, W_distribution)

    # --- Extract Position Trajectory from the results ---
    pos_trajectory_x1 = X_trajectory[1, :]
    pos_trajectory_x2 = X_trajectory[4, :]

    # --- Plotting ---
    p = plot(pos_trajectory_x1, pos_trajectory_x2,
            xlabel="X¹ Position (m)",
            ylabel="X² Position (m)",
            title="Simulated Target Trajectory (m=$m steps)",
            legend=false,
            aspect_ratio=:equal,
            marker=:circle,
            markersize=2,
            linewidth=1)
    display(p)
    println("Plotting finished.")

    # --- Reasonableness Assessment ---
    println("\nDoes the trajectory look reasonable?")
    println("Based on visual inspection of the plot, a reasonable trajectory would typically show:")
    println("   > continuous and smooth movement across the 2D plane,")
    println("   > changes in direction, corresponding to the stochastic driving command Zn,")
    println("   > some degree of randomness in the path due to the noise Wn+1.")
    println("   > persistence in direction due to the structure of the transition matrix P.")
end

# main_problem_1()

# =====================================================================================================

# === PROBLEM 3 ===

# --- Main SIS Algorithm Function ---
function run_sis(m::Int, N::Int, Δt::Float64, α::Float64, σ::Float64,
                 P::Matrix{Float64}, Z_values::Vector{Vector{Float64}},
                 μ₀::Vector{Float64}, Σ₀::Diagonal{Float64},
                 Y_obs::Matrix{Float64}, # s x (m+1)
                 stations::Matrix{Float64}, # 2 x s
                 ν::Float64, η::Float64, ζ::Float64,
                 W_distribution::MvNormal) # Added W_distribution argument

    s = size(stations, 2)
    zeta_sq = ζ^2
    num_Z_states = length(Z_values)

    # Define model matrices using the helper
    Φ, Ψz, Ψw = define_matrices(Δt, α)
    println("--- Model Matrices Used ---")
    println("Matrix Φ:"); prettyprint_matrix(Φ) # Call to prettyprint_matrix
    println("\nMatrix Ψz:"); prettyprint_matrix(Ψz) # Call to prettyprint_matrix
    println("\nMatrix Ψw:"); prettyprint_matrix(Ψw) # Call to prettyprint_matrix
    println("---------------------------")

    # Storage
    tau_hat = zeros(2, m + 1)       # Estimates (X1_hat, X2_hat)
    X_particles = zeros(6, N)       # Stores X_n at current step n
    Z_idx_particles = zeros(Int, N) # Stores Z_idx_n at current step n
    X_prev_particles = zeros(6, N)
    Z_idx_prev_particles = zeros(Int, N)
    log_weights_norm = zeros(N)     # Stores log(W_{n-1}) for start of step n

    hist_times = [m ÷ 4, m ÷ 2, m]
    weight_histograms = Dict{Int, Vector{Float64}}()

    # --- Initialization (n=0) ---
    println("Initializing SIS (N=$N)...")
    X0_dist = MvNormal(μ₀, Σ₀)
    X_particles .= rand(X0_dist, N)
    Z_idx_particles .= rand(1:num_Z_states, N)

    log_p_y0 = calculate_log_observation_density(Y_obs[:, 1], X_particles, stations, ν, η, zeta_sq)
    weights_norm, log_weights_norm = normalize_log_weights(log_p_y0)

    tau_hat[1, 1] = sum(weights_norm .* (@view X_particles[1, :]))
    tau_hat[2, 1] = sum(weights_norm .* (@view X_particles[4, :]))

    if 0 in hist_times weight_histograms[0] = copy(weights_norm) end

    println("Starting SIS loop (m=$m)...")
    # --- Iteration (n=1 to m) ---
    for n in 1:m
        if n % 50 == 0 println("  Processing step $n/$m...") end

        X_prev_particles .= X_particles
        Z_idx_prev_particles .= Z_idx_particles

        # --- Propagate Particles ---
        Z_n_minus_1_vectors = [Z_values[idx] for idx in Z_idx_prev_particles]
        for i in 1:N
             prob_vec = P[Z_idx_prev_particles[i], :]
             prob_vec = max.(0.0, prob_vec)
             sum_p = sum(prob_vec)
             if sum_p > 1e-9 && all(isfinite.(prob_vec))
                 prob_vec ./= sum_p
                 Z_idx_particles[i] = rand(Categorical(prob_vec))
             else
                 #@warn "Invalid probability vector for Categorical at n=$n, i=$i. Using uniform."
                 Z_idx_particles[i] = rand(1:num_Z_states) # Fallback
             end
        end

        W_n_samples = rand(W_distribution, N) # Use passed W_distribution

        term1 = Φ * X_prev_particles
        term3 = Ψw * W_n_samples
        term2 = zeros(6, N)
        for i in 1:N
             term2[:, i] = Ψz * Z_n_minus_1_vectors[i]
        end
        X_particles .= term1 .+ term2 .+ term3

        # --- Calculate & Update Weights ---
        yn = Y_obs[:, n+1]
        log_p_yn = calculate_log_observation_density(yn, X_particles, stations, ν, η, zeta_sq)
        log_weights_unnorm = log_weights_norm .+ log_p_yn

        # --- Normalize Weights ---
        weights_norm, log_weights_norm = normalize_log_weights(log_weights_unnorm)

        # --- Estimate tau_n ---
        tau_hat[1, n+1] = sum(weights_norm .* (@view X_particles[1, :]))
        tau_hat[2, n+1] = sum(weights_norm .* (@view X_particles[4, :]))

        # --- Store Weights for Histogram ---
        if n in hist_times
            weight_histograms[n] = copy(weights_norm)
            println("    Stored weights for histogram at n=$n")
        end
    end # End of loop n=1 to m

    println("SIS loop finished.")
    # Return estimates and histograms
    return tau_hat, weight_histograms
end

# --- Plotting Functions ---
function plot_trajectory(tau_hat, stations, title_suffix="")
    p = plot(tau_hat[1,:], tau_hat[2,:],
             label="Estimated Trajectory",
             xlabel="X¹ Position (m)",
             ylabel="X² Position (m)",
             title="SIS Estimated Trajectory" * title_suffix,
             marker=:circle, markersize=2, linewidth=1,
             aspect_ratio=:equal, legend=:outertopright)
    scatter!(p, stations[1,:], stations[2,:],
             label="Base Stations", marker=:star, markersize=8, color=:red)
    return p
end

function plot_weight_histograms(weight_histograms)
    plots_list = []
    sorted_times = sort(collect(keys(weight_histograms)))
    for n in sorted_times
        weights = weight_histograms[n]
        non_zero_weights = weights[weights .> 1e-10]
        if isempty(non_zero_weights)
             @warn "All weights near zero at n=$n. Histogram skipped."
             continue
        else
             p_hist = histogram(non_zero_weights, bins=50, normalize=:probability,
                           title="Weight Histogram at n=$n", xlabel="Normalized Weight", ylabel="Density",
                           label="", xlims=(0, maximum(non_zero_weights)*1.05))
        end
        push!(plots_list, p_hist)
    end
    if !isempty(plots_list)
        plot(plots_list..., layout=(length(plots_list), 1), size=(600, 200 * length(plots_list)))
    else
        plot(title="No valid histograms generated")
    end
end

# --- Main Execution ---
function main()
    println("------------------------------------------------")
    println("Starting Problem 3: Sequential Importance Sampling")

    # --- Set Parameters for Problem 3 Simulation ---
    # These parameters define the HMM and simulation settings used in Problem 3
    Δt = 0.5
    α = 0.6
    σ = 0.5
    m = 500 # Number of steps - will be adjusted based on loaded Y if necessary
    N = 10000 # Number of particles
    Random.seed!(15)

    # Observation model parameters
    ν = 90.0
    η = 3.0
    ζ = 1.5

    # Hidden state model parameters (defined in Problem 1, used in Problem 3)
    P_raw = [16.0 1.0 1.0 1.0 1.0;
             1.0 16.0 1.0 1.0 1.0;
             1.0 1.0 16.0 1.0 1.0;
             1.0 1.0 1.0 16.0 1.0;
             1.0 1.0 1.0 1.0 16.0]
    P = P_raw ./ sum(P_raw[1,:]) # Normalize rows

    Z_values = [[0.0, 0.0], [3.5, 0.0], [0.0, 3.5], [-3.5, 0.0], [0.0, -3.5]]

    μ₀ = zeros(6)
    Σ₀ = Diagonal([500.0, 5.0, 5.0, 200.0, 5.0, 5.0])

    # --- Load Data ---
    stations_file_path = "stations.mat"
    rssi_file_path = "RSSI-measurements.mat"
    local stations, Y_obs, m # Make m modifiable based on loaded data

    try
        println("Loading station data from $stations_file_path...")
        stations_data = matread(stations_file_path)
        if !haskey(stations_data, "pos_vec") error("Variable 'pos_vec' not found in '$stations_file_path'") end
        stations = stations_data["pos_vec"]

        println("Loading RSSI data from $rssi_file_path...")
        rssi_data = matread(rssi_file_path)
        if !haskey(rssi_data, "Y") error("Variable 'Y' not found in '$rssi_file_path'") end
        Y_obs = rssi_data["Y"]
        println("Data loaded successfully.")

        # Verify/Adjust dimensions for stations
        local s_st # Number of stations determined from data
        dim_st_orig, s_st_orig = size(stations)
        if dim_st_orig == 2 && s_st_orig >= 1
             s_st = s_st_orig
        elseif s_st_orig == 2 && dim_st_orig >= 1
             stations = stations'
             dim_st_new, s_st = size(stations)
        else
             error("Stations data 'pos_vec' must be 2xS or Sx2 (S>=1), got ($dim_st_orig, $s_st_orig)")
        end

        # Verify Y_obs dimensions and adjust m if needed
        s_loaded, m_plus_1 = size(Y_obs)
        if m_plus_1 - 1 != m
            original_m = m
            m = m_plus_1 - 1
            @info "Adjusted step count m based on loaded Y data: $original_m -> $m"
        end
        if s_loaded != s_st error("Dimension mismatch: Y has $s_loaded rows, stations implies $s_st stations.") end
        println("Using data dimensions: m=$m, s=$s_loaded")

    catch e
        println("Error during data loading or verification: $e")
        println("Please ensure '$stations_file_path' (containing 'pos_vec') and '$rssi_file_path' (containing 'Y') exist and are valid.")
        return
    end

    # --- Define Noise Distribution for simulation ---
    W_distribution = MvNormal(zeros(2), σ^2 * I)

    # --- Run the SIS Simulation (Core of Problem 3) ---
    # Corrected variable name for the returned trajectory
    X_trajectory, weight_hist_data = run_sis(m, N, Δt, α, σ, P, Z_values, μ₀, Σ₀, Y_obs, stations, ν, η, ζ, W_distribution)

    # --- Plotting Results for Problem 3 ---
    println("Plotting results...")
    # Pass the entire 2x(m+1) trajectory estimate matrix to the plotting function
    p1 = plot_trajectory(X_trajectory, stations, " (N=$N, SIS)") # <-- CORRECTED CALL
    display(p1)

    p2 = plot_weight_histograms(weight_hist_data) # Use helper function
    display(p2)
    println("Plotting finished.")

    # --- Conclusion for Problem 3 ---
    # Text unchanged from user input
    println("\n--- Conclusion ---")
    println("The SIS algorithm was implemented to estimate the trajectory based on RSSI measurements.")
    println("Inspect the 'SIS Estimated Trajectory' plot:")
    println("  - Assess the plausibility of the estimated path relative to base station locations.")
    println("\nInspect the 'Weight Histogram' plots:")
    println("  - Observe the distribution of weights at times n=$(join(sort(collect(keys(weight_hist_data))), ", ")).")
    println("  - Significant skewness towards zero indicates weight degeneracy.")
    println("  - Increasing degeneracy over time highlights the limitations of basic SIS for this m=$m step problem.")
    println("\nOverall: SIS provides an estimate, but degeneracy likely impacts reliability without resampling.")
    println("------------------------------------------------")

end

# Run the main function
# main_problem_3()

# --- Helper: Systematic Resampling ---
function systematic_resample(weights_norm::AbstractVector{Float64})
    N = length(weights_norm)
    indices = zeros(Int, N)
    C = cumsum(weights_norm)
    u1 = rand() / N
    k = 1
    for i in 1:N
        u = u1 + (i - 1) / N
        while k < N && u > C[k]
            k += 1
        end
        indices[i] = k
    end
    return indices
end

# --- Main SISR Algorithm Function ---
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

    Φ, Ψz, Ψw = define_matrices(Δt, α) # Assumes this helper is defined
    println("--- Model Matrices Used (SISR) ---")
    println("Matrix Φ:"); prettyprint_matrix(Φ) # Assumes this helper is defined
    println("\nMatrix Ψz:"); prettyprint_matrix(Ψz) # Assumes this helper is defined
    println("\nMatrix Ψw:"); prettyprint_matrix(Ψw) # Assumes this helper is defined
    println("---------------------------")

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
    println("Initializing SISR (N=$N)...")
    X0_dist = MvNormal(μ₀, Σ₀)
    X_particles .= rand(X0_dist, N)
    Z_idx_particles .= rand(1:num_Z_states, N)

    log_p_y0 = calculate_log_observation_density(Y_obs[:, 1], X_particles, stations, ν, η, zeta_sq) # Assumes helper defined
    weights_norm, log_weights_norm_current = normalize_log_weights(log_p_y0) # Assumes helper defined

    tau_hat[1, 1] = sum(weights_norm .* (@view X_particles[1, :]))
    tau_hat[2, 1] = sum(weights_norm .* (@view X_particles[4, :]))

    # Store initial weights for histogram if requested (though less useful for SISR)
    if 0 in hist_times weight_histograms[0] = copy(weights_norm) end

    # --- Initial Resample ---
    resampled_indices = systematic_resample(weights_norm)
    X_particles = X_particles[:, resampled_indices]
    Z_idx_particles = Z_idx_particles[resampled_indices]
    log_weights_norm .= -log(N) # Reset weights for start of step 1

    println("Starting SISR loop (m=$m)...")
    # --- Iteration (n=1 to m) ---
    for n in 1:m
        if n % 50 == 0 println("  Processing step $n/$m...") end

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
        log_p_yn = calculate_log_observation_density(yn, X_propagated, stations, ν, η, zeta_sq) # Assumes helper defined
        log_weights_unnorm = log_weights_norm .+ log_p_yn # Start with -log(N)

        # --- Normalization ---
        weights_norm, log_weights_norm_current = normalize_log_weights(log_weights_unnorm) # Assumes helper defined

        # --- Estimate tau_n ---
        tau_hat[1, n+1] = sum(weights_norm .* (@view X_propagated[1, :]))
        tau_hat[2, n+1] = sum(weights_norm .* (@view X_propagated[4, :]))

        # --- Store Weights for Histogram (BEFORE resampling) ---
        if n in hist_times
            weight_histograms[n] = copy(weights_norm)
            println("    Stored weights for histogram at n=$n")
        end

        # --- Resampling ---
        resampled_indices = systematic_resample(weights_norm)
        X_particles = X_propagated[:, resampled_indices]
        Z_idx_particles = Z_idx_propagated[resampled_indices]

        # --- Reset Weights for next iteration ---
        log_weights_norm .= -log(N)

    end # End of loop n=1 to m

    println("SISR loop finished.")
    # Return estimates AND histograms
    return tau_hat, weight_histograms
end

# --- Plotting Function for Histograms ---
# (This function was omitted previously, adding it back without docstring)
function plot_weight_histograms(weight_histograms)
    plots_list = []
    sorted_times = sort(collect(keys(weight_histograms)))
    for n in sorted_times
        weights = weight_histograms[n]
        # Filter weights slightly above zero for plotting, adjust threshold if needed
        plot_weights = weights[weights .> 1e-10]
        if isempty(plot_weights)
             @warn "All weights near zero at n=$n. Histogram skipped."
             continue
        else
             # Use enough bins to see detail, normalize to represent density
             p_hist = histogram(plot_weights, bins=100, normalize=:pdf,
                           title="Weight Histogram at n=$n (SISR)", xlabel="Normalized Weight", ylabel="Density",
                           label="", xlims=(0, maximum(plot_weights)*1.05)) # Adjust xlim slightly
        end
        push!(plots_list, p_hist)
    end
    if !isempty(plots_list)
        # Adjust layout and size for better viewing
        num_plots = length(plots_list)
        plot_layout = (num_plots, 1)
        plot(plots_list..., layout=plot_layout, size=(700, 250 * num_plots))
    else
        plot(title="No valid histograms generated")
    end
end


# --- Main Execution for Problem 4 ---
function main_problem_4()
    println("------------------------------------------------")
    println("Starting Problem 4: Sequential Importance Sampling with Resampling (SISR)")

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
        println("Loading station data from $stations_file_path...")
        stations_data = matread(stations_file_path)
        if !haskey(stations_data, "pos_vec") error("Variable 'pos_vec' not found in '$stations_file_path'") end
        stations = stations_data["pos_vec"]

        println("Loading RSSI data from $rssi_file_path...")
        rssi_data = matread(rssi_file_path)
        if !haskey(rssi_data, "Y") error("Variable 'Y' not found in '$rssi_file_path'") end
        Y_obs = rssi_data["Y"]
        println("Data loaded successfully.")

        local s_st
        dim_st_orig, s_st_orig = size(stations)
        if dim_st_orig == 2 && s_st_orig >= 1
             s_st = s_st_orig
        elseif s_st_orig == 2 && dim_st_orig >= 1
             stations = stations'
             dim_st_new, s_st = size(stations)
        else
             error("Stations data 'pos_vec' must be 2xS or Sx2 (S>=1), got ($dim_st_orig, $s_st_orig)")
        end

        s_loaded, m_plus_1 = size(Y_obs)
        if m_plus_1 - 1 != m
            original_m = m
            m = m_plus_1 - 1
            #@info "Adjusted step count m based on loaded Y data: $original_m -> $m"
        end
        if s_loaded != s_st error("Dimension mismatch: Y has $s_loaded rows, stations implies $s_st stations.") end
        println("Using data dimensions: m=$m, s=$s_loaded")

    catch e
        println("Error during data loading or verification: $e")
        return
    end

    # --- Define Noise Distribution ---
    W_distribution = MvNormal(zeros(2), σ^2 * I)

    # --- Run the SISR Simulation ---
    # Now receives both estimates and histogram data
    tau_estimates, weight_hist_data = run_sisr(m, N, Δt, α, σ, P, Z_values, μ₀, Σ₀, Y_obs, stations, ν, η, ζ, W_distribution)

    # --- Plotting ---
    println("Plotting results...")
    p1 = plot_trajectory(tau_estimates, stations, " (N=$N, SISR)") # Assumes plot_trajectory defined
    display(p1)

    # Call histogram plotting function
    p2 = plot_weight_histograms(weight_hist_data)
    display(p2)
    println("Plotting finished.")

    # --- Conclusion for Problem 4 ---
    println("\n--- Conclusion ---")
    println("The SISR (Sequential Importance Sampling with Resampling) algorithm was implemented.")
    println("Inspect the 'SISR Estimated Trajectory' plot:")
    println("  - Compare this trajectory to the one obtained with basic SIS (Problem 3).")
    println("  - Does it appear more stable or less noisy? Does it follow the likely path better?")
    println("  - Resampling mitigates weight degeneracy, often leading to more robust estimates.")
    println("\nInspect the 'Weight Histogram' plots (showing weights *before* resampling at selected steps):")
    println("  - How does the weight distribution look compared to the SIS histograms?")
    println("  - While degeneracy can still occur *between* resampling steps, it should be significantly less severe overall than in SIS.")
    println("\nOverall: SISR addresses the main limitation of SIS (weight degeneracy) by incorporating resampling.")
    println("This generally leads to more stable and reliable state estimates, especially for longer time series like m=$m.")
    println("The trade-off is slightly increased computational cost per step due to resampling and potential loss of diversity if resampling is done too aggressively or inappropriately.")
    println("------------------------------------------------")

end

# Optional: Call the main function for Problem 4
# main_problem_4()

