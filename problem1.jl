using LinearAlgebra, Distributions, Plots, Random, Printf
include("./utility_functions.jl")

function run_problem_1()
    verbose(true) # Whether to print info
    @info "------------------------------------------------"
    @info "Starting Problem 1: Motion Model Trajectory Simulation"

    # --- Set Parameters ---
    Δt = 0.5
    α = 0.6
    σ = 0.5
    m = 500
    Random.seed!(83)

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
    Wₙ = MvNormal(zeros(2), σ^2 * I)

    # --- Run the Simulation ---
    X_trajectory = run_simulation(m, Δt, α, P, Z_values, μ₀, Σ₀, Wₙ)

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
    @info "Problem 1 finished."
end

function run_simulation(m::Int, Δt::Float64, α::Float64,
    P::Matrix{Float64}, Z_values::Vector{Vector{Float64}},
    μ₀::Vector{Float64}, Σ₀::Diagonal{Float64},
    Wₙ::MvNormal)

    # --- State Space Matrices ---
    Φ, Ψz, Ψw = define_model_matrices(Δt, α)

    @info "\nMatrix Φ (State Transition):\n$(sprint(format_matrix_to_io, Φ))"
    @info "\nMatrix Ψz (Driving Command Input):\n$(sprint(format_matrix_to_io, Ψz))"
    @info "\nMatrix Ψw (Noise Input):\n$(sprint(format_matrix_to_io, Ψw))"

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

    @info "Starting simulation..."
    # --- Simulation Loop ---
    for n in 1:m
        # Get current state Xn and command Zn from local variables
        Xn = X_trajectory[:, n]
        Zn = Z_values[current_z_idx] # uses function-local current_z_idx

        # Sample next noise Wn+1
        Wn_plus_1 = rand(Wₙ)

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
    @info "Simulation finished."
    return X_trajectory
end

run_problem_1()