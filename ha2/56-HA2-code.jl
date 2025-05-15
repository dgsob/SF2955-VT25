include("./hybrid_mcmc.jl")
include("./compare_hmc_to_mh.jl")

function main()
    println("\nPart 1: Complex MCMC for coal mine disasters data\n")
    run_part_one()
    println("\nPart 2: Hamiltonian MC for circle-shaped posterior\n")
    run_part_two()
end

main()