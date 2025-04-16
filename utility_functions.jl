using Printf, Distributions, Logging, Statistics

# Just prints a matrix row by row with sprint
function format_matrix_to_io(io::IO, matrix::AbstractMatrix; width::Int=6)
    rows, cols = size(matrix)
    if rows == 0 || cols == 0
        print(io, "[] (Empty Matrix)")
        return
    end

    println(io, "[")
    for i in 1:rows
        print(io, " ")
        for j in 1:cols
            print(io, lpad(string(matrix[i, j]), width))
            if j < cols
                print(io, " ")
            end
        end
        println(io)
    end
    print(io, "]")
end

# Defines a minimal logger type
struct PlainLogger <: AbstractLogger
    stream::IO
    min_level::LogLevel
end
PlainLogger(stream=stdout, level=Logging.Debug) = PlainLogger(stream, level)
Logging.shouldlog(logger::PlainLogger, level, _module, group, id) = level >= logger.min_level
Logging.min_enabled_level(logger::PlainLogger) = logger.min_level
Logging.catch_exceptions(logger::PlainLogger) = false # Basic logger doesn't handle exceptions
function Logging.handle_message(logger::PlainLogger, level, message, _module, group, id, file, line; kwargs...)
    println(logger.stream, message)
end

# Wraps the global logger
function verbose(VERBOSE)
    if VERBOSE
        global_logger(PlainLogger(stdout, Logging.Debug))
    else
        global_logger(PlainLogger(stdout, Logging.Warn))
    end
end

# Creates the state space matrices 
function define_model_matrices(Δt::Float64, α::Float64)
    Φ₁ = [1 Δt Δt^2/2; 0 1 Δt; 0 0 α]
    Φ = kron(I(2), Φ₁) # creates the 6x6 block diagonal matrix

    Ψz = zeros(6, 2)
    Ψz[1, 1] = Δt^2 / 2.0 
    Ψz[2, 1] = Δt        
    Ψz[4, 2] = Δt^2 / 2.0 
    Ψz[5, 2] = Δt

    Ψw = zeros(6, 2)
    Ψw[1, 1] = Δt^2 / 2.0 
    Ψw[2, 1] = Δt        
    Ψw[3, 1] = 1.0        
    Ψw[4, 2] = Δt^2 / 2.0 
    Ψw[5, 2] = Δt        
    Ψw[6, 2] = 1.0

    return Φ, Ψz, Ψw
end

# --- Helper: Calculate Log Observation Density (Vectorized) ---
function calculate_log_observation_density(yn_col::AbstractVector,
                                           Xn_particles::AbstractMatrix, # 6xN
                                           stations::AbstractMatrix, # 2xs
                                           ν::Float64, η::Float64, ζ_sq::Float64)
    N = size(Xn_particles, 2)
    s = size(stations, 2)
    pos_n_particles = @view Xn_particles[[1, 4], :] # Efficient view (2xN)

    log_const_term_single = -0.5 * log(2 * pi * ζ_sq)
    inv_2_zeta_sq = 1.0 / (2.0 * ζ_sq)
    epsilon = 1e-12 # For numerical stability

    pos_b = reshape(pos_n_particles, 2, 1, N)
    stat_b = reshape(stations, 2, s, 1)
    dist_sq_b = sum((pos_b .- stat_b).^2, dims=1) # 1 x s x N
    distances_b = sqrt.(max.(dist_sq_b, epsilon)) # 1 x s x N
    mu_b = ν .- (10 * η * log10.(distances_b)) # 1 x s x N

    yn_b = reshape(yn_col, 1, s, 1)
    sq_errors_b = (yn_b .- mu_b).^2 # 1 x s x N
    sum_sq_err_vec = vec(sum(sq_errors_b, dims=2)) # Vector N

    log_p = s * log_const_term_single .- inv_2_zeta_sq .* sum_sq_err_vec # Vector N
    return log_p
end

# Calculates Log Observation Density (Vectorized)
function calculate_log_observation_density(yn_col::AbstractVector,
    Xn_particles::AbstractMatrix, # 6xN
    stations::AbstractMatrix, # 2xs
    ν::Float64, η::Float64, ζ_sq::Float64)

    N = size(Xn_particles, 2)
    s = size(stations, 2)
    pos_n_particles = @view Xn_particles[[1, 4], :] # Efficient view (2xN)

    log_const_term_single = -0.5 * log(2 * pi * ζ_sq)
    inv_2_zeta_sq = 1.0 / (2.0 * ζ_sq)
    epsilon = 1e-12 # For numerical stability

    pos_b = reshape(pos_n_particles, 2, 1, N)
    stat_b = reshape(stations, 2, s, 1)
    dist_sq_b = sum((pos_b .- stat_b).^2, dims=1) # 1 x s x N
    distances_b = sqrt.(max.(dist_sq_b, epsilon)) # 1 x s x N
    mu_b = ν .- (10 * η * log10.(distances_b)) # 1 x s x N

    yn_b = reshape(yn_col, 1, s, 1)
    sq_errors_b = (yn_b .- mu_b).^2 # 1 x s x N
    sum_sq_err_vec = vec(sum(sq_errors_b, dims=2)) # Vector N

    log_p = s * log_const_term_single .- inv_2_zeta_sq .* sum_sq_err_vec # Vector N
    return log_p
end

# Normalizes Log Weights
function normalize_log_weights(log_weights_unnorm::AbstractVector)
    N = length(log_weights_unnorm)
    log_weights_norm = zeros(N)
    weights_norm = zeros(N)
    log_sum_w = -Inf # Initialize log_sum_w

    non_finite_idx = .!isfinite.(log_weights_unnorm)
    if any(non_finite_idx)
         @warn "Non-finite values found in unnormalized log weights. Handling them."
         finite_max = -Inf
         if any(.!non_finite_idx)
             finite_max = maximum(log_weights_unnorm[.!non_finite_idx])
         else
             finite_max = 0.0
         end
         log_weights_unnorm[non_finite_idx] .= finite_max - 700.0
    end

    if isempty(log_weights_unnorm)
        return weights_norm, log_weights_norm, log_sum_w # Return -Inf for log_sum_w
    end

    L_max = maximum(log_weights_unnorm)
    if !isfinite(L_max)
        @warn "Maximum log weight is not finite. Assigning uniform weights."
        log_weights_norm .= -log(N)
        weights_norm .= 1.0 / N
        log_sum_w = -Inf
        return weights_norm, log_weights_norm, log_sum_w
    end

    w_tilde = exp.(log_weights_unnorm .- L_max)
    sum_w_tilde = sum(w_tilde)

    if sum_w_tilde > 1e-100 && isfinite(sum_w_tilde)
        log_sum_w = L_max + log(sum_w_tilde) # Log of sum of unnormalized weights
        log_weights_norm .= log_weights_unnorm .- log_sum_w # Normalize
        weights_norm .= exp.(log_weights_norm)
        weights_norm ./= sum(weights_norm) # Ensure sum to 1 numerically
    else
         @warn "Weight normalization failed (sum approx zero or non-finite). Assigning uniform weights."
         log_weights_norm .= -log(N)
         weights_norm .= 1.0 / N
         # log_sum_w remains -Inf
    end

    return weights_norm, log_weights_norm, log_sum_w
end

# Old plotting
# function plot_trajectory(tau_hat, stations, title_suffix="")
#     p = plot(tau_hat[1,:], tau_hat[2,:],
#              label="Estimated Trajectory",
#              xlabel="X¹ Position (m)",
#              ylabel="X² Position (m)",
#              title="Estimated Trajectory" * title_suffix,
#              marker=:circle, markersize=2, linewidth=1,
#              aspect_ratio=:equal, legend=:outertopright)
#     scatter!(p, stations[1,:], stations[2,:],
#              label="Base Stations", marker=:x, markersize=8, color=:red)
#     return p
# end 

function plot_trajectory(tau_hat, stations, title_suffix="", padding_factor=0.2)
    # Calculate bounds of the trajectory
    min_x1, max_x1 = extrema(tau_hat[1,:])
    min_x2, max_x2 = extrema(tau_hat[2,:])

    # Calculate the range (span) in each dimension
    range_x1 = max_x1 - min_x1
    range_x2 = max_x2 - min_x2

    # Determine the center of the trajectory
    center_x1 = mean(tau_hat[1,:])
    center_x2 = mean(tau_hat[2,:])

    # Determine the maximum range needed to maintain aspect ratio
    max_range = max(range_x1, range_x2) * (1 + 2 * padding_factor) # Add padding

    # Calculate limits centered around the trajectory, respecting aspect ratio
    lims_x1 = (center_x1 - max_range / 2, center_x1 + max_range / 2)
    lims_x2 = (center_x2 - max_range / 2, center_x2 + max_range / 2)

    p = plot(tau_hat[1,:], tau_hat[2,:],
             label="Estimated Trajectory",
             xlabel="X¹ Position (m)",
             ylabel="X² Position (m)",
             title="Estimated Trajectory" * title_suffix,
             marker=:circle, markersize=4, 
             linewidth=1,
             aspect_ratio=:equal, legend=:outertopright,
             xlims=lims_x1,  # Set x-limits
             ylims=lims_x2,  # Set y-limits
             size=(1000, 800))

    scatter!(p, stations[1,:], stations[2,:],
             label="Base Stations", marker=:x, markersize=8, color=:red)
    return p
end

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
        @error "No valid histograms generated"
    end
end

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
