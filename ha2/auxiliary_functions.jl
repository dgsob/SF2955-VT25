using StatsBase

function get_breakpoints_string(d, caps=false)
    string = ""
    if d == 1  
        caps ? string = "Breakpoint" : string = "breakpoint"
    else
        caps ? string = "Breakpoints" : string = "breakpoints"
    end
    return string
end

function get_label(d, sign)
    if sign == "λ"
        return ["λ$i" for i in 1:d]
    elseif sign == "t"
        return ["t$i" for i in 1:(d-1)]
    end
end

# Function to generate labels based on breakpoint modes with overlap correction
function generate_labels(t_s, d, t1, t_d1, bin_length=77, closeness_threshold=5, min_distance=5)
    # Compute the modes (highest peaks) for each breakpoint
    bin_edges_t = range(t1, t_d1, length=bin_length)
    bin_centers = (bin_edges_t[1:end-1] + bin_edges_t[2:end]) / 2  # Midpoints of bins for labeling
    modes = zeros(Int, d-1)  # Store the mode (year) for each breakpoint
    for i in 1:(d-1)
        hist = fit(Histogram, t_s[1001:end, i+1], bin_edges_t)
        max_idx = argmax(hist.weights)
        modes[i] = round(Int, bin_centers[max_idx])
    end
    
    # Adjust modes if they are too close
    adjusted = false
    while !adjusted
        adjusted = true
        for i in 1:(d-2)  # Check adjacent pairs
            if abs(modes[i+1] - modes[i]) < closeness_threshold
                adjusted = false
                # Find the next highest peak for the second breakpoint
                hist = fit(Histogram, t_s[1001:end, i+1+1], bin_edges_t)
                weights = hist.weights
                sorted_indices = sortperm(weights, rev=true)  # Sort by frequency descending
                current_mode_idx = findfirst(x -> bin_centers[x] ≈ modes[i+1], 1:length(bin_centers))
                for idx in sorted_indices
                    new_mode = round(Int, bin_centers[idx])
                    if abs(new_mode - modes[i]) >= min_distance &&  # Far from previous mode
                       all(abs(new_mode - modes[j]) >= min_distance for j in 1:(i-1) if j+1 <= d-1) &&  # Far from all previous modes
                       new_mode >= t1 && new_mode <= t_d1  # Within bounds
                        modes[i+1] = new_mode
                        break
                    end
                end
            end
        end
    end
    
    # Generate t labels
    t_labels = ["t$i: $(modes[i])" for i in 1:(d-1)]
    
    # Generate λ labels
    λ_labels = Vector{String}(undef, d)
    for i in 1:d
        if i == 1
            λ_labels[i] = "λ$i: $(Int(t1))-$(modes[i])"
        elseif i == d
            λ_labels[i] = "λ$i: $(modes[i-1])-$(Int(t_d1))"
        else
            λ_labels[i] = "λ$i: $(modes[i-1])-$(modes[i])"
        end
    end
    
    return t_labels, λ_labels
end