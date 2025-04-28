# All other .jl files must be in the same directory as this file
include("./problem1.jl")
include("./problem3.jl")
include("./problem4.jl")
include("./problem5.jl")

function main()
    run_problem_1() # Produces one figure: trajectory plot
    run_problem_3() # Produces two figures: trajectory plot, histograms plot
    run_problem_4() # Produces two figures: trajectory plot, histograms plot
    run_problem_5() # Produces two figures: log-likelihood for zeta values plot, trajectory plot
end

main()